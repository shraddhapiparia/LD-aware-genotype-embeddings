#!/usr/bin/env python3
"""
per_block_pca_baseline.py — Baseline: per-block PCA + concat + clustering.

Role in the ladder
-------------------
  raw-PCA + GMM          -> ancestry null (separate diagnostic, not this file)
  per-block PCA + concat -> THIS FILE: the linear floor to beat
  per-block VAE + concat -> reuse Phase-1 all_blocks.npy through the SAME clustering
  per-block VAE + attn   -> existing VAE_phase1.py + attention_phase2.py

Design invariants (must be identical across every condition, or the
encoder comparison is confounded with something else):
  * per-block latent dim is set by ONE rule, shared verbatim with the VAE
    baseline (scripts/baselines/latent_rule.py): p<=49 -> d=2, 50-199 -> d=4,
    p>=200 -> d=8. Neither baseline sets its own dim; both read the same
    function on the same (p, n_train).
  * PCA basis fit on TRAIN subjects only, then applied to all subjects (no leakage;
    mirrors the VAE training only on train).
  * every per-block latent is zero-padded to latent_rule.MAX_D (8) before
    concatenation, so blocks at different size classes/dims still produce a
    uniform-width concat/all_blocks tensor.

Fixed-split design (audited; see docs/baselines.md "Split semantics"):
  PCA is DETERMINISTIC given a split (sklearn's random_state only affects
  the sign/rotation of near-degenerate components), so it is fit EXACTLY
  ONCE, on the ONE split created from `shared.split_seed`. There is no
  seed loop and no median-across-seed aggregation here -- that would just
  re-measure the same deterministic fit and call the noise "seed
  variance." The five `shared.model_seeds` are VAE-only (weight init,
  minibatch order, latent sampling); every VAE model_seed reuses THIS
  split byte-identically (per_block_vae_concat.py fails loudly rather than
  silently regenerating a split if it can't).

Slide reconciliation
--------------------
  * Point 4 (per-block d by size class): implemented via latent_rule.py, shared
    with the VAE. At a given block's fixed d, per-block z-scoring and the
    sqrt(d_b) trick (point 3) reduce to a per-size-class constant with no
    effect on cluster geometry, so StandardScaler on the concat is sufficient
    here. eff_haps and a variable-dim (eff_haps) ablation are explicitly
    deferred to Phase 2 — not wired in on purpose.
  * Point 5 (80%-variance PCA): computed per block as a REPORTED diagnostic only
    (column eff_dims_80pct). It is NOT used to choose the latent dim — doing so
    would give PCA a different budget than the VAE and confound the comparison.
  * var_explained_at_d quantifies how much signal each block loses at fixed d
    (low-LD blocks will explain less with 2 PCs — point 4's "lose some signal").
    Also summarized by size_class (variance_by_size_class.csv) and separately
    for blocks >500 SNPs (large_blocks_gt500_summary.csv) — a reporting slice,
    not a fourth size class. Run scripts/baselines/pca_plausibility_summary.py
    against block_order.csv for a by-block/by-size-class plausibility check.

Outputs (under pca.output_root -- flat, no seed subdirectory: one split, one fit)
  subjects.csv, train_idx.npy, val_idx.npy, split_meta.json (split_seed, val_frac, n_train, n_val)
  block_order.csv                                  (pos, block_id, n_snps, size_class,
                                                    nominal_d, d_used, var_explained_at_d,
                                                    eff_dims_80pct)
  variance_by_size_class.csv                       (mean/median var_explained_at_d, eff_dims_80pct by size_class)
  large_blocks_gt500_summary.csv                   (n_snps>500 blocks only; reporting slice)
  embeddings/concat_embedding.npy                  (N, B*MAX_D)  -> clustering input
  embeddings/all_blocks.npy                        (N, B, MAX_D) padded, drop-in for Phase 2
  embeddings/all_blocks_latent_dims.npy            (B,) actual d_used per block (<=nominal_d<=MAX_D)
  embeddings/per_block/<block>.npy                 per-block latents (for per-block probe)
  clustering/gmm_labels.csv, gmm_bic.csv
  clustering/kmeans_labels.csv, kmeans_silhouette.csv
  baseline_summary.csv, baseline_meta.json          (both record split_seed)

Config-driven (config/baseline.yaml): no experiment CLI flags, only an
optional --config path override.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from data_utils import (
    load_block_defs, load_all_blocks, create_or_reuse_pca_split, load_baseline_config, DEFAULT_CONFIG,
    resolve_run_clustering,
)
from latent_rule import latent_dim_for_block, pad_to_max_d, MAX_D
from clustering import cluster


# ──────────────────────────────────────────────────────────────────
# per-block PCA → latents + diagnostics
# ──────────────────────────────────────────────────────────────────
def block_pca(G, tr_ix, d_used, split_seed):
    """
    Fit PCA on train subjects only, transform ALL. Plain PCA on raw dosages
    (matches the VAE input and Tas's PCA-reconstruction comparison).

    Returns latent (N, MAX_D) zero-padded, plus diagnostics. `split_seed`
    is passed as sklearn's random_state purely for bitwise reproducibility
    of near-degenerate components -- it does not make this fit "a seed's
    fit"; PCA is otherwise deterministic given the split.
    """
    N, p = G.shape
    n_tr = len(tr_ix)

    pca = PCA(n_components=d_used, random_state=split_seed)
    pca.fit(G[tr_ix])
    Z = pca.transform(G)                      # (N, d_used)
    var_at_d = float(pca.explained_variance_ratio_[:d_used].sum())

    # reported-only diagnostic: dims to reach 80% variance (slide point 5)
    full = PCA(n_components=int(min(p, n_tr - 1)), random_state=split_seed).fit(G[tr_ix])
    cum = np.cumsum(full.explained_variance_ratio_)
    eff80 = int(np.searchsorted(cum, 0.80) + 1)

    Z = pad_to_max_d(Z.astype(np.float32), d_used)
    return Z, var_at_d, eff80


# ──────────────────────────────────────────────────────────────────
# main -- runs the ONE fixed-split PCA fit (never looped per seed)
# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Per-block PCA baseline (linear floor); config-driven")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help=f"config YAML (default: {DEFAULT_CONFIG})")
    args = ap.parse_args()
    cfg = load_baseline_config(args.config, "pca")
    split_seed = int(cfg["split_seed"])
    run_clustering_flag = resolve_run_clustering(cfg)   # item 8: default True, shared with VAE

    t0 = time.time()
    out = Path(cfg["output_root"]); out.mkdir(parents=True, exist_ok=True)
    (out / "embeddings").mkdir(exist_ok=True)

    print(f"\n══════ Loading data (split_seed={split_seed}) ══════")
    bdf = load_block_defs(cfg["block_def"])
    blocks, subjects, block_ids = load_all_blocks(
        bdf, cfg["raw_dir"], use_manifest_path=cfg["use_manifest_path"])

    tr_ix, va_ix = create_or_reuse_pca_split(subjects, out, cfg.get("phase1_dir"), cfg["val_frac"], split_seed)
    with open(out / "split_meta.json", "w") as f:
        json.dump({"split_seed": split_seed, "val_frac": cfg["val_frac"],
                    "n_train": len(tr_ix), "n_val": len(va_ix), "n_subjects": len(subjects)}, f, indent=2)

    print("\n══════ Per-block PCA (fit on train, transform all) ══════")
    N, B = len(subjects), len(block_ids)
    padded = np.zeros((N, B, MAX_D), dtype=np.float32)
    meta_rows = []
    d_used_list = []
    if cfg["save_per_block"]:
        (out / "embeddings" / "per_block").mkdir(exist_ok=True)

    for j, bid in enumerate(block_ids):
        G = blocks[bid]["geno"]
        p = G.shape[1]
        size_class, nominal_d, d_used = latent_dim_for_block(p, len(tr_ix))
        Z, var_at_d, eff80 = block_pca(G, tr_ix, d_used, split_seed)
        padded[:, j, :] = Z
        d_used_list.append(d_used)
        if cfg["save_per_block"]:
            np.save(out / "embeddings" / "per_block" / f"{bid}.npy", Z[:, :d_used])
        meta_rows.append({
            "pos": j, "block_id": bid, "gene": blocks[bid]["gene"],
            "group": blocks[bid]["group"], "region_id": blocks[bid]["region_id"],
            "n_snps": p, "size_class": size_class, "nominal_d": nominal_d, "d_used": d_used,
            "n_components": d_used,  # kept for backward compat with earlier reports
            "var_explained_at_d": round(var_at_d, 4),
            "eff_dims_80pct": eff80,          # REPORTED ONLY — not used for dim
            "split_seed": split_seed,
        })
        print(f"  {bid:30s} p={p:5d} class={size_class:6s} d={d_used}  var@d={var_at_d:.3f}  eff80={eff80}")

    block_order = pd.DataFrame(meta_rows)
    block_order.to_csv(out / "block_order.csv", index=False)

    # block_order.csv is the canonical mapping between all_blocks.npy axis 1
    # (and concat_embedding.npy's per-block column groups) and block ID —
    # manifest row order, preserved unchanged through filtering, is never re-sorted.
    assert block_order["block_id"].tolist() == block_ids
    assert block_order["pos"].tolist() == list(range(len(block_ids)))
    # all_blocks_latent_dims.npy (saved below) must align 1:1, in order, with
    # block_order.csv's d_used column -- asserted explicitly (see matching
    # assertion in per_block_vae_concat.py).
    assert d_used_list == block_order["d_used"].tolist()

    # size-class variance summary (point 7: "cumulative variance explained
    # at prescribed d ... by size class")
    var_by_class = (block_order.groupby("size_class", as_index=False)
                     .agg(n_blocks=("block_id", "count"),
                          mean_var_explained_at_d=("var_explained_at_d", "mean"),
                          median_var_explained_at_d=("var_explained_at_d", "median"),
                          mean_eff_dims_80pct=("eff_dims_80pct", "mean"),
                          median_eff_dims_80pct=("eff_dims_80pct", "median")))
    var_by_class.to_csv(out / "variance_by_size_class.csv", index=False)

    # reporting slice, not a fourth class: blocks >500 SNPs
    large_gt500 = block_order[block_order["n_snps"] > 500].copy()
    large_gt500.to_csv(out / "large_blocks_gt500_summary.csv", index=False)

    concat = padded.reshape(N, B * MAX_D)
    np.save(out / "embeddings" / "concat_embedding.npy", concat)
    np.save(out / "embeddings" / "all_blocks.npy", padded)             # drop-in for Phase 2
    np.save(out / "embeddings" / "all_blocks_latent_dims.npy",
            np.array(d_used_list, dtype=np.int32))                     # actual d_used per block
    print(f"\n  concat embedding {concat.shape} | padded {padded.shape}")

    # flag how far d sits from real block structure (diagnostic, not an error)
    n_under = int((block_order["eff_dims_80pct"] > block_order["nominal_d"]).sum())
    print(f"  [diag] {n_under}/{B} blocks need more PCs than their class's nominal_d for 80% variance. "
          f"Expected; d is a deliberately matched floor per size class, not optimal.")

    if run_clustering_flag:
        print("\n══════ Clustering (GMM+BIC primary, KMeans+silhouette parity) ══════")
        best_k_gmm, best_k_km, gmm_df, km_df = cluster(
            concat, cfg["k_range"], split_seed, out / "clustering")
        clustering_metrics = {
            "best_k_gmm_bic": best_k_gmm, "best_gmm_bic": float(gmm_df["bic"].min()),
            "best_k_kmeans_sil": best_k_km, "best_kmeans_silhouette": float(km_df["silhouette"].max()),
        }
    else:
        print("\n══════ Clustering SKIPPED (shared.run_clustering: false) ══════")
        clustering_metrics = {
            "best_k_gmm_bic": np.nan, "best_gmm_bic": np.nan,
            "best_k_kmeans_sil": np.nan, "best_kmeans_silhouette": np.nan,
        }

    summary = pd.DataFrame([{
        "condition": "per_block_pca_concat",
        "n_subjects": N, "n_blocks": B, "max_d": MAX_D,
        **clustering_metrics, "clustering_run": run_clustering_flag,
        "blocks_needing_more_than_nominal_d_pcs": n_under,
        "split_seed": split_seed, "seconds": round(time.time() - t0, 1),
    }])
    summary.to_csv(out / "baseline_summary.csv", index=False)
    with open(out / "baseline_meta.json", "w") as f:
        json.dump({
            "condition": "per_block_pca_concat", "max_d": MAX_D, "split_seed": split_seed,
            "n_subjects": N, "n_blocks": B, "k_range": cfg["k_range"], "clustering_run": run_clustering_flag,
            "pca_fit_on": "train_only", "clustering": "GMM+BIC,KMeans+silhouette",
            "latent_rule": "p<=49 -> d=2, 50-199 -> d=4, p>=200 -> d=8 (scripts/baselines/latent_rule.py)",
            "fit_mode": "single fixed-split fit -- PCA is deterministic given a split, "
                        "no seed loop, no median-across-seed aggregation here (see docs/baselines.md).",
            "note": "eff_dims_80pct is reported-only; dim is fixed per size class (latent_rule.py, "
                    "matched to VAE). all_blocks_latent_dims.npy records the actual per-block d_used.",
        }, f, indent=2)

    print(f"\n{'═'*55}")
    print(summary.to_string(index=False))
    print(f"[done] split_seed={split_seed}  {time.time()-t0:.1f}s → {out}")


if __name__ == "__main__":
    main()
