"""
Elbow + Silhouette / Davies-Bouldin / Calinski-Harabasz sweep for K-Means,
run on the corrected behavioral feature vectors (see FeatureEngineering.py).

Produces:
  - validity_<dataset>_<entity>.csv   (K, inertia, silhouette, davies_bouldin, calinski_harabasz)
  - Figure_Elbow_<dataset>_<entity>.png
  - Figure_Silhouette_<dataset>_<entity>.png
  - a printed summary row for K=36 (the value used in the paper)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import FeatureEngineering as FE

K_RANGE = list(range(4, 61, 4)) + [36]
K_RANGE = sorted(set(K_RANGE))


def sweep(X, entity_name, dataset_name, sample_size=3000):
    rows = []
    n = X.shape[0]
    sil_sample = sample_size if n > sample_size else None
    for k in K_RANGE:
        if k >= n:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        sil = silhouette_score(X, labels, sample_size=sil_sample, random_state=42)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        rows.append({'K': k, 'inertia': inertia, 'silhouette': sil,
                     'davies_bouldin': db, 'calinski_harabasz': ch})
        print(f"{dataset_name}-{entity_name} K={k:3d}  inertia={inertia:10.1f}  "
              f"silhouette={sil:+.4f}  DB={db:.4f}  CH={ch:9.2f}")
    df = pd.DataFrame(rows).sort_values('K')
    df.to_csv(f'validity_{dataset_name}_{entity_name}.csv', index=False)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(df['K'], df['inertia'], marker='o', color='tab:blue')
    ax1.axvline(36, color='gray', linestyle='--', linewidth=1)
    ax1.set_xlabel('Number of clusters (K)')
    ax1.set_ylabel('Inertia (within-cluster SSE)')
    ax1.set_title(f'Elbow curve — {dataset_name} {entity_name}')
    fig.tight_layout()
    fig.savefig(f'Figure_Elbow_{dataset_name}_{entity_name}.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df['K'], df['silhouette'], marker='o', label='Silhouette (higher=better)')
    ax2 = ax.twinx()
    ax2.plot(df['K'], df['davies_bouldin'], marker='s', color='tab:red', label='Davies–Bouldin (lower=better)')
    ax.axvline(36, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Number of clusters (K)')
    ax.set_ylabel('Silhouette score')
    ax2.set_ylabel('Davies–Bouldin index')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax.set_title(f'Cluster validity indices — {dataset_name} {entity_name}')
    fig.tight_layout()
    fig.savefig(f'Figure_Silhouette_{dataset_name}_{entity_name}.png', dpi=150)
    plt.close(fig)

    return df


if __name__ == '__main__':
    dfa = pd.read_csv('amazon test set.csv', header=None, names=['User No.', 'Product ID code', 'Rating'])
    Xu, uid, Xp, pid = FE.amazon_features(dfa, rating_col='Rating')
    print("\n=== Amazon: users ===")
    sweep(Xu, 'users', 'Amazon')
    print("\n=== Amazon: products ===")
    sweep(Xp, 'products', 'Amazon')

    dfm = pd.read_csv('Movielens100k.csv')
    Xu2, uid2, Xp2, pid2 = FE.movielens_features(dfm)
    print("\n=== MovieLens: users ===")
    sweep(Xu2, 'users', 'MovieLens')
    print("\n=== MovieLens: products ===")
    sweep(Xp2, 'products', 'MovieLens')
