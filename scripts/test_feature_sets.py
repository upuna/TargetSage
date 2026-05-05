#!/usr/bin/env python3
import os, sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd, numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from targetsage import load_bio_features, load_llm_embeddings, load_labels

bio_df = load_bio_features('data/gene_features.tsv')
emb_df = load_llm_embeddings('data/features_llm_embedding.csv')
label_df = load_labels('data/gene_labels.tsv')
merged = bio_df.merge(emb_df, on='Gene_Symbol').merge(label_df[['Gene_Symbol']], on='Gene_Symbol')
genes = merged['Gene_Symbol'].astype(str).values

bio_cols = [c for c in bio_df.columns if c != 'Gene_Symbol']
emb_cols = [c for c in emb_df.columns if c != 'Gene_Symbol']
Xb = SimpleImputer(strategy='median').fit_transform(merged[bio_cols].values)
Xe = SimpleImputer(strategy='median').fit_transform(merged[emb_cols].values)
Xe_pca = PCA(n_components=256, random_state=42).fit_transform(Xe)
Xb = StandardScaler().fit_transform(Xb)
Xe_pca = StandardScaler().fit_transform(Xe_pca)

X_bio = Xb
X_bioemb = np.hstack([Xb, Xe_pca])

labels_raw = pd.read_csv('data/gene_labels.tsv', sep='\t', dtype=str)
def norm(x):
    if pd.isna(x): return None
    return {'tclin':'Tclin','tchem':'Tchem','tbio':'Tbio','tdark':'Tdark'}.get(str(x).strip().lower(), str(x).strip())

old = np.array([norm(labels_raw.set_index('Gene_Symbol')['idgTDL_old'].to_dict().get(g)) for g in genes], dtype=object)
new = np.array([norm(labels_raw.set_index('Gene_Symbol')['idgTDL_new'].to_dict().get(g)) for g in genes], dtype=object)
mask = np.array([o in ['Tclin','Tchem','Tbio','Tdark'] for o in old])
upgraded_A = set(genes[np.isin(old, ['Tchem','Tbio','Tdark']) & (new == 'Tclin')])
upgraded_B = set(genes[np.isin(old, ['Tbio','Tdark']) & np.isin(new, ['Tclin','Tchem'])])
N = len(genes)

methods = {
    'GB': GradientBoostingClassifier(random_state=42),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    'LR': LogisticRegression(max_iter=5000),
    'KNN': KNeighborsClassifier(n_neighbors=10, n_jobs=1),
    'NB': GaussianNB(),
    'MLP': MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200, random_state=42),
}

for task_name, y_task, upgraded in [
    ('TaskA', (old=='Tclin').astype(int), upgraded_A),
    ('TaskB', np.isin(old, ['Tclin','Tchem']).astype(int), upgraded_B),
]:
    print(f'\n{task_name}:')
    for fs_name, X in [('bio_only', X_bio), ('bio+emb256', X_bioemb)]:
        res = []
        for m_name, clf_template in methods.items():
            clf = clone(clf_template)
            clf.fit(X[mask], y_task[mask])
            scores = clf.predict_proba(X)[:, 1]
            ranks = rankdata(-scores)
            g2r = dict(zip(genes, ranks))
            pcts = [g2r[g]/N*100 for g in upgraded if g in g2r]
            res.append(f'{m_name}={np.median(pcts):.1f}%')
        print(f'  {fs_name:15s} {"  ".join(res)}')
