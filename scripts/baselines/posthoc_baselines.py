#!/usr/bin/env python3
"""
posthoc_baselines.py -- Tier-0 diagnostics on a COMPLETED per-block run.
No training, no model checkpoints required: everything here is computed from
the genotypes, the saved latents (all_blocks.npy), and the PCA split.

Answers, per block:
  1. MARGINAL BASELINE (plan item 5)
     Does the latent add anything over predicting each SNP from its
     train-set genotype frequencies? Reports ordinal NLL and dosage MSE for
     the marginal predictor, and the VAE's improvement over it.
     NLL convention matches training_summary.va_ordinal_nll exactly:
     SUM over SNPs, MEAN over validation subjects.

  2. SUBJECT DISTANCE PRESERVATION (plan item 4)
     Spearman correlation between pairwise standardized-genotype Euclidean
     distances and pairwise latent Euclidean distances, validation subjects
     only. Replaces the degenerate row-wise `ld_corr`. Computed for the VAE
     latents and (if available) the PCA latents at matched d.

  3. CLASS SUPPORT (plan item 6)
     n_validation_g0/g1/g2 and n_snps_with_validation_g2, so a recall_g2 of
     0.11 can be read against how many minor homozygotes actually exist.

  4. MATCHED-d PCA RECONSTRUCTION (plan: "matched PCA comparison")
     Refits per-block PCA on TRAIN at the same d_used and reports validation
     dosage MSE, against the VAE's va_dosage_mse and the marginal's MSE.
     PCA is continuous, so it has no ordinal NLL -- dosage MSE is the only
     like-for-like scale. Do NOT compare PCA MSE to VAE NLL.

Latents are read from all_blocks.npy (N, B, MAX_D) and truncated to
all_blocks_latent_dims.npy, so no per-block .npy files are needed.

Usage:
  python posthoc_baselines.py \
      --config config/baseline.yaml \
      --vae-dir results/GACRS/baseline_vae/ORD/beta0.5_e200_pilot/seed_42 \
      --pca-dir results/GACRS/baseline_pca \
      --out results/GACRS/posthoc/tier0_seed42.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import (
    load_block_defs, load_all_blocks, load_baseline_config, DEFAULT_CONFIG,
    subject_distance_preservation, class_support, maf_bin_labels,
)
from per_block_vae_concat import ordinal_probs, ordinal_loss, ordinal_nll_per_subject


# ──────────────────────────────────────────────────────────────────
# 1. marginal baseline
# ──────────────────────────────────────────────────────────────────
def marginal_baseline(G, tr_ix, va_ix, alpha=1.0):
    """Per-SNP categorical predictor fit on TRAIN genotype frequencies only.

    Returns (nll, mse, mae) on validation subjects.
      nll : SUM over SNPs, MEAN over subjects -- identical convention to
            training_summary.va_ordinal_nll, so the two are directly
            subtractable.
      mse : on EXPECTED dosage E[G] = P1 + 2*P2, matching how the VAE's
            va_dosage_mse is computed from its ordinal head (not on a hard
            argmax call).

    alpha is Laplace smoothing, which also guarantees log(0) can't occur for
    a genotype class absent from train but present in validation.
    """
    Gtr, Gva = G[tr_ix], G[va_ix]
    Gtr_i = np.rint(Gtr).astype(np.int64)
    Gva_i = np.rint(Gva).astype(np.int64)

    counts = np.stack([(Gtr_i == g).sum(axis=0) for g in (0, 1, 2)]).astype(np.float64)
    P = (counts + alpha) / (counts + alpha).sum(axis=0, keepdims=True)   # (3, p)
    # item 11 safeguard: P must be a genuine probability simplex per SNP --
    # finite, nonnegative, sums to one -- before anything downstream logs it.
    assert np.isfinite(P).all(), "marginal_baseline: non-finite class probability"
    assert (P >= 0).all(), "marginal_baseline: negative class probability"
    assert np.allclose(P.sum(axis=0), 1.0, atol=1e-6), "marginal_baseline: per-SNP class probabilities don't sum to 1"
    logP = np.log(P)

    # NLL: pick each validation subject's observed class per SNP
    ll = np.take_along_axis(logP[:, None, :], Gva_i[None, :, :], axis=0)[0]  # (n_va, p)
    nll = float(-ll.sum(axis=1).mean())

    exp_dos = P[1] + 2.0 * P[2]                                          # (p,)
    err = Gva - exp_dos[None, :]
    return nll, float((err ** 2).mean()), float(np.abs(err).mean())


# ──────────────────────────────────────────────────────────────────
# 2. subject distance preservation -- item 5: train-fitted scaling, shared
# with per_block_vae_concat.py via data_utils.subject_distance_preservation
# (NOT reimplemented here; the old validation-only-scaled version has been
# removed to avoid two divergent definitions of the same statistic).
# ──────────────────────────────────────────────────────────────────

# 3. class support -- item 2/item 3: shared with per_block_vae_concat.py via
# data_utils.class_support (NOT reimplemented here).


# ──────────────────────────────────────────────────────────────────
# 4. matched-d PCA reconstruction (PCA's OWN objective: squared error)
# ──────────────────────────────────────────────────────────────────
def pca_fit_transform(G, tr_ix, va_ix, d_used, seed):
    """Fit PCA on TRAIN at the SAME d_used the VAE used; return the fitted
    model plus train/validation scores, so callers needing both the
    reconstruction-MSE diagnostic (this function's caller) and the linear
    ordinal-head baseline (pca_ordinal_head, item 6) can reuse ONE fit
    instead of two."""
    d = int(min(d_used, G.shape[1], len(tr_ix) - 1))
    pca = PCA(n_components=d, random_state=seed).fit(G[tr_ix])
    Ztr, Zva = pca.transform(G[tr_ix]), pca.transform(G[va_ix])
    return pca, Ztr, Zva, d


def pca_recon_mse(pca, G_va, Zva):
    """PCA-objective reconstruction (inverse_transform of its OWN scores):
    validation dosage MSE/MAE. Cheap: only d components, no full spectrum
    (unlike the eff80 diagnostic in per_block_PCA_concat.py)."""
    rec = pca.inverse_transform(Zva)
    err = G_va - rec
    return float((err ** 2).mean()), float(np.abs(err).mean())


# ──────────────────────────────────────────────────────────────────
# 6. PCA + linear ordinal decoder -- a cheap, objective-MATCHED comparison
# (PCA's raw inverse-transform MSE above is not, since PCA directly
# optimizes squared error while the VAE optimizes ordinal NLL). Freezes the
# PCA scores from pca_fit_transform above, then fits ONLY a linear
# d_used -> 2*p ordinal head on them by full-batch gradient descent,
# reusing the exact ordinal_probs/ordinal_loss/ordinal_nll_per_subject the
# VAE's own decoder head uses (scripts/baselines/per_block_vae_concat.py) --
# so the reconstruction math is identical, only decoder CAPACITY differs.
# Model selection is by validation ordinal NLL, explicit/reproducible
# settings below (epochs, lr, patience, seed=split_seed).
# ──────────────────────────────────────────────────────────────────
def _g2_calibration_from_probs(p_cls, y, maf_train, maf_bins):
    """Calibration summaries for a precomputed (N,p,3) probability tensor."""
    p_cls = np.asarray(p_cls)
    y = np.asarray(y, dtype=np.int64)
    p_g2 = p_cls[..., 2]
    chosen = np.take_along_axis(p_cls, y[..., None], axis=-1)[..., 0]
    nll = -np.log(np.clip(chosen, 1e-6, None))
    is_g2 = (y == 2).astype(np.float64)
    labels = maf_bin_labels(maf_bins)
    obs, pred, brier, nll_bin, n_obs = {}, {}, {}, {}, {}
    for lo, hi, lab in zip(maf_bins[:-1], maf_bins[1:], labels):
        snp_mask = (maf_train >= lo) & (maf_train < hi if hi < maf_bins[-1] else maf_train <= hi)
        if not snp_mask.any():
            obs[lab] = pred[lab] = brier[lab] = nll_bin[lab] = float("nan")
            n_obs[lab] = 0
            continue
        obs[lab] = float(is_g2[:, snp_mask].mean())
        pred[lab] = float(p_g2[:, snp_mask].mean())
        brier[lab] = float(np.mean((p_g2[:, snp_mask] - is_g2[:, snp_mask]) ** 2))
        nll_bin[lab] = float(nll[:, snp_mask].mean())
        n_obs[lab] = int(is_g2[:, snp_mask].size)
    return {
        "observed_g2_frequency_by_maf_bin": json.dumps(obs),
        "mean_predicted_p_g2_by_maf_bin": json.dumps(pred),
        "g2_brier_score_by_maf_bin": json.dumps(brier),
        "ordinal_nll_per_snp_by_maf_bin": json.dumps(nll_bin),
        "n_observations_by_maf_bin": json.dumps(n_obs),
    }


def pca_ordinal_head(Ztr, Zva, ytr, yva, d, p, seed, maf_train=None, maf_bins=None,
                     epochs=300, lr=0.05, patience=20):
    """'PCA + linear ordinal head' (label it exactly that in any report --
    NOT 'the VAE', and not a claim that its decoder capacity matches the
    VAE's nonlinear 2-layer decoder: a worse score here means a LINEAR
    readout of the PCA representation is worse, not necessarily that the
    representation itself is worse)."""
    # Fit scaling on TRAIN scores only. Without this, a fixed learning rate
    # behaves differently across PCs/blocks solely because score variances
    # differ, making the post-hoc comparison an optimizer-scale test.
    z_mean = Ztr.mean(axis=0)
    z_sd = Ztr.std(axis=0)
    keep = z_sd > 1e-8
    if not keep.any():
        raise ValueError("PCA ordinal head has no nonconstant training score dimension")
    Ztr = (Ztr[:, keep] - z_mean[keep]) / z_sd[keep]
    Zva = (Zva[:, keep] - z_mean[keep]) / z_sd[keep]
    d_fit = int(keep.sum())

    torch.manual_seed(seed)
    head = nn.Linear(d_fit, p * 2)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Ztr_t = torch.tensor(Ztr, dtype=torch.float32)
    Zva_t = torch.tensor(Zva, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    yva_t = torch.tensor(yva, dtype=torch.long)

    best_nll, best_sd, best_epoch, wait = float("inf"), None, None, 0
    for epoch in range(1, epochs + 1):
        head.train()
        logits = head(Ztr_t).view(-1, p, 2)
        loss = ordinal_loss(logits, ytr_t)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite PCA ordinal training loss at epoch {epoch}")
        opt.zero_grad()
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            va_logits = head(Zva_t).view(-1, p, 2)
            va_nll = ordinal_loss(va_logits, yva_t).item()
        if not np.isfinite(va_nll):
            raise FloatingPointError(f"non-finite PCA ordinal validation NLL at epoch {epoch}")
        if va_nll < best_nll:
            best_nll, best_epoch, wait = va_nll, epoch, 0
            best_sd = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_sd is None:
        raise RuntimeError("PCA ordinal head produced no eligible finite checkpoint")
    head.load_state_dict(best_sd)
    head.eval()
    with torch.no_grad():
        va_logits = head(Zva_t).view(-1, p, 2)
    p_cls, _, _ = ordinal_probs(va_logits)
    nll_per_subj = ordinal_nll_per_subject(va_logits, yva_t)
    exp_dosage = p_cls[..., 1].numpy() + 2.0 * p_cls[..., 2].numpy()
    err = exp_dosage - yva.astype(np.float32)
    result = {
        "pca_ord_va_nll": float(nll_per_subj.mean().item()),
        "pca_ord_va_nll_per_snp": float(nll_per_subj.mean().item() / p),
        "pca_ord_va_dosage_mse": float(np.mean(err ** 2)),
        "pca_ord_va_dosage_mae": float(np.mean(np.abs(err))),
        "pca_ord_best_epoch": int(best_epoch),
        "pca_ord_d_fit": d_fit,
    }
    if maf_train is not None and maf_bins is not None:
        cal = _g2_calibration_from_probs(p_cls.numpy(), yva, maf_train, maf_bins)
        result.update({f"pca_ord_{k}": v for k, v in cal.items()})
    return result


# ──────────────────────────────────────────────────────────────────
def load_latents(run_dir):
    """(N, B, MAX_D) latents + per-block d_used from a completed run dir."""
    emb = Path(run_dir) / "embeddings"
    Z = np.load(emb / "all_blocks.npy")
    dims = np.load(emb / "all_blocks_latent_dims.npy")
    return Z, dims


def compute_tier0(vae_dir, pca_dir, blocks, block_ids, tr_ix, va_ix, split_seed,
                  maf_bins, max_subjects_dist=None):
    """One model_seed's worth of tier-0 diagnostics. Genotypes/split/block
    order are loaded ONCE by the caller and passed in, since they are
    identical across every model_seed (item 7) -- only vae_dir's own
    embeddings/training_summary differ per call."""
    Zv, dims_v = load_latents(vae_dir)
    order = pd.read_csv(vae_dir / "block_order.csv")
    if order["block_id"].astype(str).tolist() != block_ids:
        raise ValueError(f"{vae_dir}: block_order.csv does not match freshly loaded block order")
    model_seed = int(order["model_seed"].iloc[0])

    have_pca = (pca_dir / "embeddings" / "all_blocks.npy").exists()
    if have_pca:
        Zp, dims_p = load_latents(pca_dir)
        if not np.array_equal(dims_p, dims_v):
            raise ValueError("PCA and VAE d_used differ -- the latent rule drifted")
    print(f"[latents] {vae_dir.name}: VAE {Zv.shape}" + (f" | PCA {Zp.shape}" if have_pca else " | no PCA latents found"))

    rows = []
    for j, bid in enumerate(block_ids):
        G = blocks[bid]["geno"]
        p = G.shape[1]
        d = int(dims_v[j])
        Gtr_i = np.rint(G[tr_ix]).astype(np.int64)
        Gva_i = np.rint(G[va_ix]).astype(np.int64)

        m_nll, m_mse, m_mae = marginal_baseline(G, tr_ix, va_ix)
        r = {
            "block_id": bid, "p": p, "d_used": d, "model_seed": model_seed,
            "marginal_nll": round(m_nll, 4),
            "marginal_dosage_mse": round(m_mse, 6),
            "marginal_dosage_mae": round(m_mae, 6),
            # item 5: train-fitted scaling (data_utils.subject_distance_preservation),
            # NOT the old validation-only-scaled statistic.
            "dist_pres_vae": subject_distance_preservation(
                G[tr_ix], G[va_ix], Zv[tr_ix, j, :d], Zv[va_ix, j, :d], max_subjects=max_subjects_dist),
        }
        # item 2/3: shared class_support (data_utils), all three classes --
        # keeps this script's established n_validation_g*/frac_snps_with_validation_g* names.
        cs = class_support(Gva_i)
        for g in (0, 1, 2):
            r[f"n_validation_g{g}"] = cs[f"n_g{g}"]
            r[f"n_snps_with_validation_g{g}"] = cs[f"n_snps_with_g{g}"]
            r[f"frac_snps_with_validation_g{g}"] = cs[f"fraction_snps_with_g{g}"]

        # item 6: PCA's OWN objective (inverse-transform MSE) -- one fit, reused below.
        pca, Ztr_pca, Zva_pca, d_pca = pca_fit_transform(G, tr_ix, va_ix, d, split_seed)
        p_mse, p_mae = pca_recon_mse(pca, G[va_ix], Zva_pca)
        r["pca_dosage_mse"] = round(p_mse, 6)
        r["pca_dosage_mae"] = round(p_mae, 6)
        if have_pca:
            r["dist_pres_pca"] = subject_distance_preservation(
                G[tr_ix], G[va_ix], Zp[tr_ix, j, :d], Zp[va_ix, j, :d], max_subjects=max_subjects_dist)

        # item 6: PCA + LINEAR ordinal head -- objective-matched (ordinal NLL)
        # comparison, freezing the same PCA fit above.
        maf_train = G[tr_ix].mean(axis=0) / 2.0
        maf_train = np.minimum(maf_train, 1.0 - maf_train)
        head = pca_ordinal_head(
            Ztr_pca, Zva_pca, Gtr_i, Gva_i, d_pca, p, seed=split_seed,
            maf_train=maf_train, maf_bins=maf_bins)
        r.update(head)
        r["pca_ord_nll_improvement_over_marginal"] = round(m_nll - head["pca_ord_va_nll"], 4)

        rows.append(r)
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(block_ids)} blocks")

    df = pd.DataFrame(rows)

    # ---- join the VAE's own (training-time) diagnostics -- NOT recomputed
    # here (item 4's dims_above_free_bits/all_dims_below_free_bits/
    # any_dim_above_free_bits/raw_kl_minus_free_capacity/kl_to_recon_ratio
    # and item 5's subject_distance_preservation already live in
    # training_summary.csv; this script only reads them). ----
    ts = pd.read_csv(vae_dir / "training_summary.csv")
    keep = ["block_id", "size_class", "epochs_run", "best_epoch", "va_ordinal_nll", "va_dosage_mse",
            "va_dosage_mae", "va_recall_g2", "raw_total_kl", "free_bits_rate", "active_latent_dims",
            "complete_collapse", "dims_above_free_bits", "fraction_dims_above_free_bits",
            "all_dims_below_free_bits", "any_dim_above_free_bits", "raw_kl_minus_free_capacity",
            "kl_to_recon_ratio", "subject_distance_preservation"]
    df = df.merge(ts[[c for c in keep if c in ts.columns]], on="block_id", how="left")

    # THE headline columns: positive = beats predicting marginals (item 3);
    # negative vae_minus_pca_ord_nll = VAE better than PCA+linear head (item 6).
    df["vae_nll_improvement_over_marginal"] = (df["marginal_nll"] - df["va_ordinal_nll"]).round(4)
    df["vae_nll_improvement_per_snp"] = (df["vae_nll_improvement_over_marginal"] / df["p"]).round(5)
    df["vae_mse_improvement_over_marginal"] = (df["marginal_dosage_mse"] - df["va_dosage_mse"]).round(6)
    df["pca_mse_improvement_over_marginal"] = (df["marginal_dosage_mse"] - df["pca_dosage_mse"]).round(6)
    df["vae_minus_pca_mse"] = (df["pca_dosage_mse"] - df["va_dosage_mse"]).round(6)
    df["vae_minus_pca_ord_nll"] = (df["va_ordinal_nll"] - df["pca_ord_va_nll"]).round(4)
    return df


def block_quality_stability(dfs):
    """item 7: per-block stability of the marginal-improvement signal ACROSS
    model_seeds -- NOT an exclusion rule (primary outputs keep every block;
    any later filtering is a separately specified sensitivity analysis)."""
    if len(dfs) < 2:
        raise ValueError("block_quality_stability requires at least two model-seed results")
    expected_blocks = set(dfs[0]["block_id"].astype(str))
    seen_seeds = []
    for i, df in enumerate(dfs):
        blocks_i = set(df["block_id"].astype(str))
        if blocks_i != expected_blocks:
            raise ValueError(f"model-seed result {i} has different block coverage")
        seeds_i = df["model_seed"].dropna().unique()
        if len(seeds_i) != 1:
            raise ValueError(f"model-seed result {i} must contain exactly one model_seed")
        seen_seeds.append(int(seeds_i[0]))
    if len(set(seen_seeds)) != len(seen_seeds):
        raise ValueError(f"duplicate model_seed inputs are not allowed: {seen_seeds}")

    all_df = pd.concat(dfs, ignore_index=True)
    g = all_df.groupby("block_id")
    out = g.agg(
        n_model_seeds=("model_seed", "nunique"),
        n_seeds_vae_beats_marginal=("vae_nll_improvement_over_marginal", lambda x: int((x > 0).sum())),
        median_nll_improvement_over_marginal=("vae_nll_improvement_over_marginal", "median"),
        min_nll_improvement_over_marginal=("vae_nll_improvement_over_marginal", "min"),
        max_nll_improvement_over_marginal=("vae_nll_improvement_over_marginal", "max"),
        iqr_nll_improvement_over_marginal=(
            "vae_nll_improvement_over_marginal", lambda x: float(x.quantile(0.75) - x.quantile(0.25))),
        median_nll_improvement_per_snp=("vae_nll_improvement_per_snp", "median"),
        median_subject_distance_preservation=("dist_pres_vae", "median"),
    ).reset_index()
    out["fraction_seeds_vae_beats_marginal"] = (
        out["n_seeds_vae_beats_marginal"] / out["n_model_seeds"]).round(4)
    return out.round(4)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--vae-dir", required=True, nargs="+",
                     help="one or more completed VAE model_seed dirs (contains embeddings/, training_summary.csv); "
                          "pass 2+ to also write block_quality_stability.csv (item 7)")
    ap.add_argument("--pca-dir", required=True, help="PCA output_root (split source; PCA latents used if present)")
    ap.add_argument("--out", default=None, help="posthoc_tier0.csv path (single --vae-dir only)")
    ap.add_argument("--stability-out", default=None,
                     help="block_quality_stability.csv path (default: common parent of --vae-dir / this name)")
    ap.add_argument("--max-subjects-dist", type=int, default=None,
                    help="subsample validation subjects for the distance stat (default: use all)")
    args = ap.parse_args()
    vae_dirs = [Path(v) for v in args.vae_dir]
    pca_dir = Path(args.pca_dir)

    cfg = load_baseline_config(args.config, "vae")
    split_seed = int(cfg["split_seed"])
    maf_bins = cfg.get("maf_bins", [0.0, 0.05, 0.10, 0.20, 0.50])

    # ---- split (authoritative: PCA's, exactly as every VAE model_seed reused it) ----
    subjects = pd.read_csv(pca_dir / "subjects.csv")["IID"].astype(str).values
    tr_ix = np.load(pca_dir / "train_idx.npy")
    va_ix = np.load(pca_dir / "val_idx.npy")
    print(f"[split] train {len(tr_ix)} / val {len(va_ix)}")

    # ---- genotypes -- loaded ONCE, shared across every vae_dir (item 7) ----
    bdf = load_block_defs(cfg["block_def"])
    blocks, subj_loaded, block_ids = load_all_blocks(
        bdf, cfg["raw_dir"], use_manifest_path=cfg["use_manifest_path"])
    if not np.array_equal(subj_loaded, subjects):
        raise ValueError("subject order differs from the PCA split -- wrong raw_dir/manifest?")

    dfs = []
    for vae_dir in vae_dirs:
        print(f"\n{'#'*78}\n# {vae_dir}\n{'#'*78}")
        df = compute_tier0(vae_dir, pca_dir, blocks, block_ids, tr_ix, va_ix, split_seed, maf_bins,
                            max_subjects_dist=args.max_subjects_dist)
        out = Path(args.out) if (args.out and len(vae_dirs) == 1) else vae_dir / "posthoc_tier0.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        dfs.append(df)

        # ──────────────────────────────────────────────────────────
        print("\n" + "=" * 78)
        print(f"TIER-0 POST-HOC DIAGNOSTICS — {vae_dir}")
        print("=" * 78)

        print("\n[1] Improvement over marginal baseline, by size class")
        print(df.groupby("size_class").agg(
            n=("block_id", "size"),
            nll_impr_med=("vae_nll_improvement_over_marginal", "median"),
            nll_impr_per_snp=("vae_nll_improvement_per_snp", "median"),
            n_no_better=("vae_nll_improvement_over_marginal", lambda x: int((x <= 0).sum())),
            dist_pres=("dist_pres_vae", "median"),
        ).round(4).to_string())

        below = df[df["all_dims_below_free_bits"] == True]    # noqa: E712
        above = df[df["any_dim_above_free_bits"] == True]     # noqa: E712
        print(f"\n[2] THE KEY TEST -- do the {len(below)} all_dims_below_free_bits blocks beat marginal?")
        if len(below):
            print(f"    all_dims_below_free_bits: median NLL improvement "
                  f"{below['vae_nll_improvement_over_marginal'].median():.4f} "
                  f"({below['vae_nll_improvement_per_snp'].median():.5f}/SNP), "
                  f"{int((below['vae_nll_improvement_over_marginal'] <= 0).sum())}/{len(below)} no better than marginal")
            print(f"    any_dim_above_free_bits: median NLL improvement "
                  f"{above['vae_nll_improvement_over_marginal'].median():.4f} "
                  f"({above['vae_nll_improvement_per_snp'].median():.5f}/SNP)")
            print(f"    all_dims_below_free_bits median distance preservation "
                  f"{below['dist_pres_vae'].median():.4f} vs any_dim_above_free_bits {above['dist_pres_vae'].median():.4f}")
            print("    -> improvement ~0 for all_dims_below_free_bits blocks: the KL/free-bits crossing (a")
            print("       stochastic-posterior property) agrees with the reconstruction-utility read.")
            print("       improvement > 0 there: mu still carries signal the KL floor alone hides -- this is")
            print("       exactly why dims_above_free_bits must never redefine collapse (see item 4).")

        print("\n[3] Matched-d reconstruction: VAE vs PCA (inverse-transform) vs marginal (dosage MSE, lower better)")
        print(df.groupby("size_class").agg(
            marginal=("marginal_dosage_mse", "median"),
            pca=("pca_dosage_mse", "median"),
            vae=("va_dosage_mse", "median"),
            vae_beats_pca=("vae_minus_pca_mse", lambda x: int((x > 0).sum())),
            n=("block_id", "size"),
        ).round(5).to_string())

        print("\n[3b] Objective-matched: VAE vs PCA + linear ordinal head (ordinal NLL, lower better)")
        print(df.groupby("size_class").agg(
            marginal=("marginal_nll", "median"),
            pca_linear_head=("pca_ord_va_nll", "median"),
            vae=("va_ordinal_nll", "median"),
            vae_beats_pca_head=("vae_minus_pca_ord_nll", lambda x: int((x < 0).sum())),
            n=("block_id", "size"),
        ).round(4).to_string())

        print("\n[4] Class support vs recall_g2")
        print(df.groupby("size_class").agg(
            n_g2_median=("n_validation_g2", "median"),
            frac_snps_with_g2=("frac_snps_with_validation_g2", "median"),
            recall_g2=("va_recall_g2", "median"),
        ).round(4).to_string())

        print(f"\n[written] {out}")

    if len(vae_dirs) > 1:
        stab = block_quality_stability(dfs)
        stab_out = Path(args.stability_out) if args.stability_out else (
            Path(*vae_dirs[0].parts[:-1]) / "block_quality_stability.csv")
        stab_out.parent.mkdir(parents=True, exist_ok=True)
        stab.to_csv(stab_out, index=False)
        print(f"\n{'='*78}\nBLOCK QUALITY STABILITY across {len(vae_dirs)} model_seeds "
              f"({stab['n_model_seeds'].iloc[0] if len(stab) else 0} seeds/block)\n{'='*78}")
        print(f"  fraction_seeds_vae_beats_marginal: median {stab['fraction_seeds_vae_beats_marginal'].median():.4f}, "
              f"{int((stab['fraction_seeds_vae_beats_marginal'] < 1.0).sum())}/{len(stab)} blocks not unanimous "
              f"across seeds (reported only -- primary outputs retain every block).")
        print(f"[written] {stab_out}")


if __name__ == "__main__":
    main()
