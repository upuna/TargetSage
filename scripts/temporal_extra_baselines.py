#!/usr/bin/env python3
"""Add SVM, TabNet, and FT-Transformer rankings to existing temporal validation eval dir.

Trains each baseline on 2021 PHAROS labels (Tclin or TclinOrTchem) using bio-only features (d=482),
scores all ~19,032 genes, and writes ranking CSVs in the same format expected by
plot_temporal_validation.py.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

ROOT = "/home/zihend1/Genesis/TargetSage2"
os.chdir(ROOT)

def normalize_tdl(x):
    if pd.isna(x): return np.nan
    return {"tclin":"Tclin","tchem":"Tchem","tbio":"Tbio","tdark":"Tdark"}.get(
        str(x).strip().lower(), str(x).strip())

class FTTransformer(nn.Module):
    def __init__(self, d_in, d_emb=64, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.emb = nn.Linear(1, d_emb)
        self.pos = nn.Parameter(torch.randn(1, d_in, d_emb) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_emb) * 0.02)
        layer = nn.TransformerEncoderLayer(d_emb, n_heads, d_emb * 2, dropout=dropout,
                                            norm_first=True, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_emb, 1)
    def forward(self, x):
        B = x.shape[0]
        z = self.emb(x.unsqueeze(-1)) + self.pos
        z = torch.cat([self.cls.expand(B, -1, -1), z], 1)
        z = self.enc(z)
        return self.head(z[:, 0]).squeeze(-1)

class TabNetSimple(nn.Module):
    """Simplified TabNet-like attention-based tabular model."""
    def __init__(self, d_in, d_h=128, n_steps=3, dropout=0.1):
        super().__init__()
        self.attn_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_in, d_in), nn.BatchNorm1d(d_in), nn.GELU(),
            nn.Linear(d_in, d_in), nn.Sigmoid(),
        ) for _ in range(n_steps)])
        self.feat_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_in, d_h), nn.BatchNorm1d(d_h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_h, d_h), nn.GELU(),
        ) for _ in range(n_steps)])
        self.head = nn.Linear(d_h, 1)
    def forward(self, x):
        out = 0
        for attn, feat in zip(self.attn_layers, self.feat_layers):
            mask = attn(x)
            out = out + feat(x * mask)
        return self.head(out).squeeze(-1)

def train_torch(model, X_tr, y_tr, X_score, device, epochs=80, lr=1e-3, bs=512):
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
    ap.add_argument("--out_dir", required=True, help="Existing upgrade_eval_* directory to add results to")
    ap.add_argument("--bio", default="data/gene_features.tsv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[DATA] loading...")
    bio = pd.read_csv(args.bio, sep="\t").set_index("Gene_Symbol")
    bio = bio.apply(pd.to_numeric, errors='coerce').fillna(0.0)
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

    for task_key, info in tasks.items():
        y = info["y"]
        upgraded_genes = [g for g, u in zip(genes, info["upgraded"].values) if u]
        print(f"\n[{task_key}]  pos={y.sum()}, upgraded={len(upgraded_genes)}")

        # Save upgraded genes file (in case missing)
        up_csv = os.path.join(args.out_dir, f"{task_key}__upgraded_genes.csv")
        if not os.path.exists(up_csv):
            pd.DataFrame({"Gene_Symbol": upgraded_genes}).to_csv(up_csv, index=False)

        old_t = old.values
        new_t = new.values

        def save_ranking(scores, name):
            df = pd.DataFrame({
                "Gene_Symbol": genes, "score": scores,
                "idgTDL_old": old_t, "idgTDL_new": new_t,
            }).sort_values("score", ascending=False).reset_index(drop=True)
            df.index = df.index + 1
            df.index.name = "rank"
            out = os.path.join(args.out_dir, f"{task_key}__{name}.csv")
            df.to_csv(out)
            print(f"    {name}: top-1={df.iloc[0].Gene_Symbol} ({df.iloc[0].score:.4f})  -> {out}")

        # TabNet
        print("  Training TabNet ...")
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        tabnet = TabNetSimple(X.shape[1]).to(device)
        s_tab = train_torch(tabnet, X, y, X, device, epochs=80, lr=1e-3, bs=512)
        save_ranking(s_tab, "TabNet")

        # FT-Transformer
        print("  Training FT-Transformer ...")
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        ft = FTTransformer(X.shape[1]).to(device)
        s_ft = train_torch(ft, X, y, X, device, epochs=80, lr=1e-3, bs=512)
        save_ranking(s_ft, "FT-Trans")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
