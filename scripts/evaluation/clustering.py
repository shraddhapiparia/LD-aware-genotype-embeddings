#!/usr/bin/env python3
"""
clustering.py — shared clustering routine for the per-block baseline ladder
(per_block_PCA_concat.py, per_block_vae_concat.py).

No model-specific (PCA/VAE) logic lives here.
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# ──────────────────────────────────────────────────────────────────
# clustering — identical recipe for every condition in the ladder
# ──────────────────────────────────────────────────────────────────
def cluster(concat, k_range, seed, out):
    """GMM+BIC (primary) and KMeans+silhouette (parity). Standardize first."""
    out.mkdir(parents=True, exist_ok=True)
    Z = StandardScaler().fit_transform(concat).astype(np.float64)   # matches attention_phase2.run_clustering

    gmm_labels, gmm_rows = {}, []
    for k in k_range:
        gm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=1e-6,
                n_init=10,
                random_state=seed
            )
        lab = gm.fit_predict(Z)
        bic = float(gm.bic(Z))
        sil = float(silhouette_score(Z, lab)) if k > 1 else 0.0
        gmm_labels[f"gmm_k{k}"] = lab
        gmm_rows.append({"k": k, "bic": round(bic, 2), "silhouette": round(sil, 4)})
        print(f"    GMM    k={k:2d}  BIC={bic:12.1f}  sil={sil:.4f}")
    gmm_df = pd.DataFrame(gmm_rows)
    best_k_gmm = int(gmm_df.loc[gmm_df["bic"].idxmin(), "k"])
    print(f"    GMM best k (min BIC) = {best_k_gmm}")

    km_labels, km_rows = {}, []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        lab = km.fit_predict(Z)
        sil = float(silhouette_score(Z, lab)) if k > 1 else 0.0
        km_labels[f"kmeans_k{k}"] = lab
        km_rows.append({"k": k, "silhouette": round(sil, 4)})
        print(f"    KMeans k={k:2d}  sil={sil:.4f}")
    km_df = pd.DataFrame(km_rows)
    best_k_km = int(km_df.loc[km_df["silhouette"].idxmax(), "k"])
    print(f"    KMeans best k (max silhouette) = {best_k_km}")

    pd.DataFrame(gmm_labels).to_csv(out / "gmm_labels.csv", index=False)
    gmm_df.to_csv(out / "gmm_bic.csv", index=False)
    pd.DataFrame(km_labels).to_csv(out / "kmeans_labels.csv", index=False)
    km_df.to_csv(out / "kmeans_silhouette.csv", index=False)
    return best_k_gmm, best_k_km, gmm_df, km_df
