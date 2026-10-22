#!/usr/bin/env python3
"""
data_utils.py — shared block-loading and train/val split helpers for the
per-block baseline ladder (per_block_PCA_concat.py, per_block_vae_concat.py).

No model-specific (PCA/VAE) logic lives here.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

DEFAULT_CONFIG = "config/baseline.yaml"


def load_baseline_config(config_path, section):
    """Load config/baseline.yaml (or an override path) and merge the
    dataset + shared settings with the model-specific `section` ('pca' or
    'vae') into one flat dict. Shared keys (val_frac, k_range, split_seed,
    model_seeds, ...) live only under `shared:`, so PCA and VAE always read
    the identical value -- never a duplicated one. Per-block latent dim is
    derived from latent_rule.py, not read from config.

    split_seed vs. model_seeds (fixed-split design): split_seed is the ONE
    seed used to create the train/validation split (PCA fits once on it;
    every VAE model_seed reuses it verbatim, never regenerates it).
    model_seeds is VAE-only -- it seeds weight init, minibatch order, and
    latent sampling, and has no effect on which subjects are in train/val."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    merged = {}
    merged.update(cfg["dataset"])
    merged.update(cfg["shared"])
    merged.update(cfg[section])
    return merged


# ──────────────────────────────────────────────────────────────────
# loader — matched to read_raw()/load_all_blocks() in VAE_phase1.py
# ──────────────────────────────────────────────────────────────────
def read_raw(path):
    """Return (subject_ids, snp_names, geno float32 in {0,1,2}). PLINK --recodeA."""
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    sids = df["IID"].astype(str).values
    dup_mask = pd.Series(sids).duplicated()
    if dup_mask.any():
        dups = sorted(set(sids[dup_mask.to_numpy()]))
        raise ValueError(f"{path}: {int(dup_mask.sum())} duplicate IID value(s), e.g. {dups[:5]}")
    snp_cols = list(df.columns[6:])  # after FID IID PAT MAT SEX PHENOTYPE
    G = df[snp_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(G).any():
        raise ValueError(f"{path}: {int(np.isnan(G).sum())} NaNs after parsing.")
    gmin, gmax = float(G.min()), float(G.max())
    if gmin < -1e-6 or gmax > 2 + 1e-6:
        raise ValueError(f"{path}: dosages out of [0,2]: min={gmin}, max={gmax}")
    return sids, snp_cols, G


def load_block_defs(tsv):
    """
    Read the block manifest and normalize column names so the rest of the
    pipeline can rely on: block_id, gene, group, region_id.

    Accepts the GACRS schema (haploblock_id, region_id, group, raw_file, ...)
    as well as the older block_id/gene schema. Only status==ok rows are kept.
    """
    df = pd.read_csv(tsv, sep="\t", header=0)

    # id column: prefer block_id, else haploblock_id
    if "block_id" not in df.columns:
        if "haploblock_id" in df.columns:
            df = df.rename(columns={"haploblock_id": "block_id"})
        else:
            raise KeyError("manifest needs 'block_id' or 'haploblock_id'")
    df["block_id"] = df["block_id"].astype(str)

    # carry-through metadata, with safe fallbacks
    if "gene" not in df.columns:
        df["gene"] = df["region_id"] if "region_id" in df.columns else ""
    if "group" not in df.columns:
        df["group"] = ""
    if "region_id" not in df.columns:
        df["region_id"] = ""

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")].copy()

    dup_mask = df["block_id"].duplicated()
    if dup_mask.any():
        dups = sorted(set(df.loc[dup_mask, "block_id"]))
        raise ValueError(
            f"{tsv}: {int(dup_mask.sum())} duplicate block_id value(s) after "
            f"schema normalization/status filtering, e.g. {dups[:5]}"
        )

    print(f"[blocks] {len(df)} OK entries from {tsv}")
    if "group" in df.columns and df["group"].astype(bool).any():
        print(f"[blocks] group counts: {df['group'].value_counts().to_dict()}")
    return df


def load_all_blocks(block_df, raw_dir, fail_on_missing=True, use_manifest_path=False):
    """
    Load one .raw per block, align to sorted subject intersection, keep TSV order.

    Path resolution (portable by default):
      * default: <raw_dir>/<block_id>.raw  — works regardless of where the
        manifest was generated (the manifest's absolute raw_file may be a path
        from another machine).
      * use_manifest_path=True: trust the manifest's 'raw_file' column verbatim.
    """
    blocks, common = {}, None
    raw_dir = str(raw_dir)
    ordered_ids = block_df["block_id"].astype(str).tolist()

    for bid in ordered_ids:
        row = block_df.loc[block_df["block_id"].astype(str) == bid].iloc[0]
        if use_manifest_path and "raw_file" in block_df.columns:
            fp = str(row["raw_file"])
        else:
            fp = os.path.join(raw_dir, f"{bid}.raw")
        if not os.path.isfile(fp):
            msg = f"[missing] {fp}"
            if fail_on_missing:
                raise FileNotFoundError(
                    msg + ("\n  (manifest raw_file paths look absolute/foreign; "
                           "run with --raw-dir pointing at the .raw folder, "
                           "or pass --use-manifest-path if they are valid here.)")
                )
            print("  " + msg + " (skipping)")
            continue
        sids, snps, G = read_raw(fp)
        blocks[bid] = {
            "sids": sids, "geno": G,
            "gene": str(row.get("gene", "")),
            "group": str(row.get("group", "")),
            "region_id": str(row.get("region_id", "")),
        }
        common = set(sids) if common is None else (common & set(sids))
        print(f"  {bid:30s} {G.shape[1]:>4d} SNPs  {G.shape[0]} subj")

    if not blocks:
        raise RuntimeError("No blocks loaded. Check raw_dir / block_ids.")
    common = sorted(common)
    if len(common) < 10:
        raise RuntimeError(f"Subject intersection too small (n={len(common)}). Check IID sets.")

    common_arr = np.array(common, dtype=str)
    for bid, d in blocks.items():
        sid2i = {s: i for i, s in enumerate(d["sids"])}
        idx = [sid2i[s] for s in common]
        d["geno"] = d["geno"][idx]
        d["n_snps"] = d["geno"].shape[1]
    kept = [b for b in ordered_ids if b in blocks]
    print(f"[data] {len(common)} subjects × {len(blocks)} blocks")
    return blocks, common_arr, kept


# ──────────────────────────────────────────────────────────────────
# split — reuse Phase 1 if present, else create and persist
# ──────────────────────────────────────────────────────────────────
def _validate_split(subjects, tr, va):
    """Lightweight integrity checks for a train/val split, run before it is
    saved or returned -- whether just created or reused from --phase1-dir."""
    N = len(subjects)
    tr = np.asarray(tr)
    va = np.asarray(va)
    if not np.issubdtype(tr.dtype, np.integer) or not np.issubdtype(va.dtype, np.integer):
        raise ValueError(f"split indices must be integer-typed (got tr={tr.dtype}, va={va.dtype})")
    if tr.size and (tr.min() < 0 or tr.max() >= N):
        raise ValueError(f"train indices out of range [0,{N})")
    if va.size and (va.min() < 0 or va.max() >= N):
        raise ValueError(f"val indices out of range [0,{N})")
    tr_list, va_list = tr.tolist(), va.tolist()
    if len(set(tr_list)) != len(tr_list):
        raise ValueError("duplicate indices within train split")
    if len(set(va_list)) != len(va_list):
        raise ValueError("duplicate indices within val split")
    overlap = set(tr_list) & set(va_list)
    if overlap:
        raise ValueError(f"train/val overlap: {len(overlap)} shared indices, e.g. {sorted(overlap)[:5]}")
    covered = set(tr_list) | set(va_list)
    if len(covered) != N:
        raise ValueError(f"train+val must cover every subject exactly once: covered {len(covered)} of {N}")


def _save_split(out, subjects, tr, va):
    """Write subjects.csv/train_idx.npy/val_idx.npy into `out` (always, whether
    the split was just created or reused from --phase1-dir), so every
    experiment's own output directory is self-contained."""
    np.save(out / "train_idx.npy", tr)
    np.save(out / "val_idx.npy", va)
    pd.DataFrame({"IID": subjects}).to_csv(out / "subjects.csv", index=False)


def resolve_model_seeds(cfg):
    """Return the list of VAE model_seeds to run (init/minibatch/latent-
    sampling stochasticity only -- never the split). `shared.model_seeds`
    (a list) is the finalized Phase-1 mode; single-element lists work too."""
    seeds = cfg.get("model_seeds")
    if not seeds:
        raise ValueError("config shared.model_seeds is required (e.g. [42, 43, 44, 45, 46])")
    return [int(s) for s in seeds]


def model_seed_output_dir(output_root, model_seed):
    """Per-model_seed VAE output subdirectory: <output_root>/seed_<model_seed>/.
    Always nested (unlike the old ambiguous multi-seed flag) -- VAE always
    has at least one model_seed and its outputs always live under seed_<n>/."""
    return Path(output_root) / f"seed_{model_seed}"


def compute_maf(G, tr_ix):
    """Per-SNP minor allele frequency from TRAIN subjects only (no leakage),
    used to stratify ORD recall by MAF bin. G is dosage {0,1,2}, (N, p)."""
    freq = G[tr_ix].mean(axis=0) / 2.0
    return np.minimum(freq, 1.0 - freq)


def maf_bin_labels(maf_bins):
    return [f"[{maf_bins[i]:.2f},{maf_bins[i+1]:.2f})" for i in range(len(maf_bins) - 1)]


def create_or_reuse_pca_split(subjects, out, phase1_dir, val_frac, split_seed):
    """PCA's OWN split resolution -- the authoritative split every VAE
    model_seed must later reuse verbatim. Reuses an explicit phase1_dir if
    given and alignable (e.g. to lock onto a split from elsewhere), else
    creates ONE fresh split from split_seed. PCA is deterministic given a
    split, so this is called exactly once (never per-seed)."""
    if phase1_dir:
        p1 = Path(phase1_dir)
        need = [p1 / "subjects.csv", p1 / "train_idx.npy", p1 / "val_idx.npy"]
        if all(f.exists() for f in need):
            p1_sub = pd.read_csv(p1 / "subjects.csv")["IID"].astype(str).values
            if np.array_equal(p1_sub, subjects):
                tr = np.load(p1 / "train_idx.npy")
                va = np.load(p1 / "val_idx.npy")
                _validate_split(subjects, tr, va)
                print(f"[split] reused external split from {phase1_dir}: train {len(tr)} / val {len(va)} — saved to {out}")
                _save_split(out, subjects, tr, va)
                return tr, va
            print(f"[split] {phase1_dir} subject order differs from this run — creating fresh split from split_seed={split_seed}.")
    N = len(subjects)
    n_val = int(N * val_frac)
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(N)
    va, tr = perm[:n_val], perm[n_val:]
    _validate_split(subjects, tr, va)
    print(f"[split] created split (split_seed={split_seed}): train {len(tr)} / val {len(va)} — saved.")
    _save_split(out, subjects, tr, va)
    return tr, va


def require_pca_split(subjects, out, phase1_dir):
    """VAE's ONLY split resolution: MUST reuse the split PCA already
    created, byte-identical, for EVERY model_seed. Never falls back to
    creating a new split -- fails loudly instead, since a silently
    regenerated split would break the PCA/VAE fairness comparison without
    any visible error."""
    if not phase1_dir:
        raise ValueError(
            "vae.phase1_dir is required: VAE must reuse the PCA baseline's split, "
            "never create its own. Run per_block_PCA_concat.py first, then point "
            "vae.phase1_dir at its output_root."
        )
    p1 = Path(phase1_dir)
    need = [p1 / "subjects.csv", p1 / "train_idx.npy", p1 / "val_idx.npy"]
    missing = [str(f) for f in need if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"PCA split files missing at {phase1_dir}: {missing}. "
            f"Run per_block_PCA_concat.py first to create the split VAE must reuse."
        )
    p1_sub = pd.read_csv(p1 / "subjects.csv")["IID"].astype(str).values
    if not np.array_equal(p1_sub, subjects):
        raise ValueError(
            f"Subject set/order at {phase1_dir} does not match this run's subjects "
            f"(different manifest/raw_dir?) -- refusing to silently create a new split. "
            f"Re-run per_block_PCA_concat.py against the same data this VAE run sees."
        )
    tr = np.load(p1 / "train_idx.npy")
    va = np.load(p1 / "val_idx.npy")
    _validate_split(subjects, tr, va)
    print(f"[split] reusing PCA split from {phase1_dir} (required, not optional): train {len(tr)} / val {len(va)} — saved to {out}")
    _save_split(out, subjects, tr, va)
    return tr, va


def resolve_run_clustering(cfg):
    """`shared.run_clustering` (default True if absent) -- functional switch
    shared verbatim by PCA and VAE so a diagnostics-only run (e.g. the long-
    training pilot, item 10) can skip the GMM/KMeans step on both without
    the two baselines drifting on the default."""
    value = cfg.get("run_clustering", True)
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            "shared.run_clustering must be a YAML boolean (true/false), "
            f"not {value!r}"
        )
    return bool(value)


def class_support(y_int):
    """Per-class subject-level support from an (N, p) integer-genotype
    matrix in {0,1,2}: n_g{0,1,2} (subject-SNP observation counts) and
    n_snps_with_g{0,1,2} / fraction_snps_with_g{0,1,2} (how many SNPs have
    at least one observation of that class in this split). Shared by
    per_block_vae_concat.py (train + validation, at the restored best
    checkpoint) and posthoc_baselines.py (validation only, against the
    marginal baseline) so the definition can't drift between the two."""
    y_int = np.asarray(y_int)
    out = {}
    for g in (0, 1, 2):
        mask = y_int == g
        out[f"n_g{g}"] = int(mask.sum())
        per_snp_g = mask.sum(axis=0)
        out[f"n_snps_with_g{g}"] = int((per_snp_g > 0).sum())
        out[f"fraction_snps_with_g{g}"] = round(float((per_snp_g > 0).mean()), 4)
    return out


def subject_distance_preservation(G_tr, G_va, mu_tr, mu_va, max_subjects=None, seed=0):
    """Spearman correlation between pairwise validation-subject genotype
    distance and pairwise validation-subject latent (mu) distance --
    replaces the old rowwise np.corrcoef 'ld_corr' statistic (deprecated,
    see compute_ld_corr in per_block_vae_concat.py), which standardized on
    validation subjects themselves (a train/val-leakage-adjacent, circular
    scaling choice). Scaling here is fit on TRAIN only, exactly as the
    model itself was fit, then applied to validation:
      1. Fit per-SNP genotype mean/SD on TRAIN subjects only.
      2. Drop train-zero-variance SNPs (can't be standardized).
      3/4. Fit per-dimension mu mean/SD on TRAIN mu only.
      5. Drop train-zero-variance latent dimensions.
      6. Transform VALIDATION genotype and latent matrices with the
         train-fitted scaling (never refit on validation).
      7. Pairwise Euclidean distance (scipy pdist) on each, validation only.
      8. Spearman rho between the two validation distance vectors.
    Returns NaN whenever undefined (no train-informative SNPs/dims, a
    degenerate <3-subject validation set, or a constant distance vector --
    the honest answer for a fully collapsed block), rather than raising.
    """
    G_tr, G_va = np.asarray(G_tr), np.asarray(G_va)
    mu_tr, mu_va = np.asarray(mu_tr), np.asarray(mu_va)

    def _fit(X_tr):
        mean, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
        return mean, sd, sd > 1e-8

    g_mean, g_sd, g_keep = _fit(G_tr)
    z_mean, z_sd, z_keep = _fit(mu_tr)
    if g_keep.sum() == 0 or z_keep.sum() == 0:
        return float("nan")

    Gva_s = (G_va[:, g_keep] - g_mean[g_keep]) / g_sd[g_keep]
    Zva_s = (mu_va[:, z_keep] - z_mean[z_keep]) / z_sd[z_keep]

    n = Gva_s.shape[0]
    if max_subjects and n > max_subjects:
        rng = np.random.RandomState(seed)
        sel = rng.choice(n, max_subjects, replace=False)
        Gva_s, Zva_s = Gva_s[sel], Zva_s[sel]
    if Gva_s.shape[0] < 3:
        return float("nan")

    dG, dZ = pdist(Gva_s), pdist(Zva_s)
    if np.std(dG) < 1e-12 or np.std(dZ) < 1e-12:
        return float("nan")
    return float(spearmanr(dG, dZ).correlation)


def read_split_seed(phase1_dir, default=None):
    """Read the split_seed PCA recorded in its own split_meta.json, for
    provenance in VAE outputs (more robust than trusting config.split_seed
    might match what PCA actually used, in case of drift)."""
    p = Path(phase1_dir) / "split_meta.json" if phase1_dir else None
    if p and p.exists():
        with open(p) as f:
            return json.load(f).get("split_seed", default)
    return default
