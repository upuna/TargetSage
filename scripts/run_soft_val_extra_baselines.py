#!/usr/bin/env python3
"""Train TabNet and FT-Transformer on 2025 Tclin labels and save genome-wide rankings
for soft validation (ChEMBL Phase 2/3 enrichment analysis).

Output: results/baseline_inference_2025/FT-Trans_ranking.csv
        results/baseline_inference_2025/TabNet_ranking.csv
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = "/home/zihend1/Genesis/TargetSage2"
os.chdir(ROOT)

TASK_COL = "task_pharos_tclin_vs_others"
OUT_DIR  = "results/baseline_inference_2025"
BIO_FILE = "data/gene_features.tsv"
LAB_FILE = "data/gene_labels.tsv"
SEED     = 42
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


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
        return self.head(self.enc(z)[:, 0]).squeeze(-1)


class TabNetSimple(nn.Module):
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
            out = out + feat(x * attn(x))
        return self.head(out).squeeze(-1)


def train_and_score(model, X, y, device, epochs=80, lr=1e-3, bs=512):
    pos_w = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32, device=device)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    n  = len(Xt)
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            crit(model(Xt[b]), yt[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, n, 4096):
            chunks.append(torch.sigmoid(model(Xt[i:i+4096])).cpu().numpy())
        return np.concatenate(chunks)


def save_ranking(genes, scores, name, out_dir):
    df = (pd.DataFrame({"Gene_Symbol": genes, "score": scores})
            .sort_values("score", ascending=False)
            .reset_index(drop=True))
    df.index = df.index + 1
    df.index.name = "rank"
    out = os.path.join(out_dir, f"{name}_ranking.csv")
    df.to_csv(out)
    print(f"  [{name}] saved -> {out}  top-1={df.iloc[0].Gene_Symbol} ({df.iloc[0].score:.4f})")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)

    print("[DATA] loading features and labels...")
    bio  = pd.read_csv(BIO_FILE, sep="\t").set_index("Gene_Symbol")
    bio  = bio.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    ldf  = pd.read_csv(LAB_FILE, sep="\t")
    ldf[TASK_COL] = pd.to_numeric(ldf[TASK_COL], errors="coerce").fillna(0).astype(int)
    ldf  = ldf.set_index("Gene_Symbol")

    genes = sorted(set(bio.index) & set(ldf.index))
    X = SimpleImputer(strategy="median").fit_transform(bio.reindex(genes).values)
    X = StandardScaler().fit_transform(X)
    y = ldf.reindex(genes)[TASK_COL].values.astype(int)
    print(f"  {len(genes)} genes, d={X.shape[1]}, pos={y.sum()}, device={DEVICE}")

    print("\n[TabNet] training...")
    tabnet = TabNetSimple(X.shape[1]).to(DEVICE)
    s_tab = train_and_score(tabnet, X, y, DEVICE)
    save_ranking(genes, s_tab, "TabNet", OUT_DIR)

    print("\n[FT-Trans] training...")
    np.random.seed(SEED); torch.manual_seed(SEED)
    ft = FTTransformer(X.shape[1]).to(DEVICE)
    s_ft = train_and_score(ft, X, y, DEVICE)
    save_ranking(genes, s_ft, "FT-Trans", OUT_DIR)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
