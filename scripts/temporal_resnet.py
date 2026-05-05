#!/usr/bin/env python3
"""Add ResNet temporal validation results to an existing upgrade_eval directory."""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = "/home/zihend1/Genesis/TargetSage2"
os.chdir(ROOT)


def normalize_tdl(x):
    if pd.isna(x): return np.nan
    return {"tclin":"Tclin","tchem":"Tchem","tbio":"Tbio","tdark":"Tdark"}.get(
        str(x).strip().lower(), str(x).strip())


class ResNetBlock(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d, d), nn.BatchNorm1d(d),
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(x + self.net(x)))


class ResNetMLP(nn.Module):
    def __init__(self, d_in, d_hidden=256, n_blocks=4, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResNetBlock(d_hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(d_hidden, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(-1)


def train_resnet(X_tr, y_tr, X_score, device, epochs=100, lr=1e-3, bs=512, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    model = ResNetMLP(X_tr.shape[1], d_hidden=256, n_blocks=4, dropout=0.1).to(device)
    pos_w = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)],
                         dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xs = torch.from_numpy(X_score.astype(np.float32)).to(device)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            loss = crit(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        scores = []
        for i in range(0, len(Xs), 4096):
            scores.append(torch.sigmoid(model(Xs[i:i+4096])).cpu().numpy())
        return np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bio",    default="data/gene_features.tsv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DATA] loading... device={device}")

    bio = pd.read_csv(args.bio, sep="\t").set_index("Gene_Symbol")
    bio = bio.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    ldf = pd.read_csv(args.labels, sep="\t", dtype=str)
    ldf["old_n"] = ldf["idgTDL_old"].apply(normalize_tdl)
    ldf["new_n"] = ldf["idgTDL_new"].apply(normalize_tdl)
    ldf = ldf.set_index("Gene_Symbol")

    genes = sorted(set(bio.index) & set(ldf.index))
    X = SimpleImputer(strategy="median").fit_transform(bio.reindex(genes).values)
    X = StandardScaler().fit_transform(X)
    print(f"  {len(genes)} genes, X shape: {X.shape}")

    old = ldf.reindex(genes)["old_n"]
    new = ldf.reindex(genes)["new_n"]

    tasks = {
        "upgrade_to_Tclin": {
            "y": (old == "Tclin").values.astype(int),
            "upgraded": (old != "Tclin") & (new == "Tclin"),
        },
        "upgrade_to_TclinOrTchem": {
            "y": ((old == "Tclin") | (old == "Tchem")).values.astype(int),
            "upgraded": (~old.isin(["Tclin", "Tchem"])) & (new.isin(["Tclin", "Tchem"])),
        },
    }

    old_t = old.values
    new_t = new.values

    for task_key, info in tasks.items():
        y = info["y"]
        upgraded_genes = [g for g, u in zip(genes, info["upgraded"].values) if u]
        print(f"\n[{task_key}]  pos={y.sum()}, upgraded={len(upgraded_genes)}")

        up_csv = os.path.join(args.out_dir, f"{task_key}__upgraded_genes.csv")
        if not os.path.exists(up_csv):
            pd.DataFrame({"Gene_Symbol": upgraded_genes}).to_csv(up_csv, index=False)

        out_path = os.path.join(args.out_dir, f"{task_key}__ResNet.csv")
        if os.path.exists(out_path):
            print(f"  [SKIP] {out_path} already exists")
            continue

        print(f"  Training ResNet (epochs={args.epochs}, seed={args.seed})...")
        scores = train_resnet(X, y, X, device, epochs=args.epochs, lr=1e-3, bs=512, seed=args.seed)

        df = pd.DataFrame({
            "Gene_Symbol": genes, "score": scores,
            "idgTDL_old": old_t, "idgTDL_new": new_t,
        }).sort_values("score", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"
        df.to_csv(out_path)
        print(f"  -> {out_path}")

        # Print median rank percentile for upgraded genes
        N = len(df)
        g2rank = dict(zip(df["Gene_Symbol"].astype(str), df.index.astype(int)))
        pcts = np.array([g2rank[g]/N*100.0 for g in upgraded_genes if g in g2rank])
        print(f"  Median rank percentile: {np.median(pcts):.1f}%")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
