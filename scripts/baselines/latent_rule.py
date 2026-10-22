#!/usr/bin/env python3
"""
latent_rule.py — the ONE latent-dimension rule shared by per_block_PCA_concat.py
and per_block_vae_concat.py, plus the VAE hidden-width formula that is keyed
off the same (p, d) pair. Centralized so the two baselines can never drift
apart on class boundaries or dimension.

Rule (finalized Phase-1 design; eff_haps and a fourth size class are
explicitly out of scope for Phase 1 — see docs/baselines.md):
    p <= 49          -> size_class="small",  nominal_d=2
    50 <= p <= 199    -> size_class="medium", nominal_d=4
    p >= 200          -> size_class="large",  nominal_d=8

`actual_d_used` clamps nominal_d to what the data can support:
min(nominal_d, p, n_train - 1). All padded tensors (all_blocks.npy,
concat_embedding.npy) use MAX_D=8 so block-to-block dimension differences
don't change array shapes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

MAX_D = 8

_BOUNDARIES = (
    (49, "small", 2),
    (199, "medium", 4),
)
_LARGE = ("large", 8)


def size_class_and_nominal_d(p):
    """p (int, SNP count) -> (size_class, nominal_d). The single source of
    truth for the class boundaries — do not duplicate this logic elsewhere."""
    for hi, name, d in _BOUNDARIES:
        if p <= hi:
            return name, d
    return _LARGE


def latent_dim_for_block(p, n_train):
    """(p, n_train) -> (size_class, nominal_d, actual_d_used).

    actual_d_used <= nominal_d <= MAX_D; also clamped to p and n_train-1 so a
    block can never be asked to fit more latent dims than it has SNPs or
    training subjects to support."""
    size_class, nominal_d = size_class_and_nominal_d(p)
    actual_d_used = int(min(nominal_d, p, n_train - 1))
    if actual_d_used < 1:
        raise ValueError(
            f"block with p={p} SNPs, n_train={n_train} subjects cannot "
            f"support even d=1 (need n_train>=2 and p>=1)"
        )
    return size_class, nominal_d, actual_d_used


def hidden_widths(p, d):
    """Two-layer VAE width formula, keyed on (p, actual_d_used):
        h1 = round(p^(2/3) * d^(1/3))
        h2 = round(p^(1/3) * d^(2/3))
    encoder p->h1->h2 (mirrored decoder d->h2->h1)."""
    h1 = max(1, round(float(p) ** (2.0 / 3.0) * float(d) ** (1.0 / 3.0)))
    h2 = max(1, round(float(p) ** (1.0 / 3.0) * float(d) ** (2.0 / 3.0)))
    return int(h1), int(h2)


def pad_to_max_d(Z, d_used):
    """Zero-pad a (N, d_used) latent array to (N, MAX_D)."""
    N = Z.shape[0]
    if d_used == MAX_D:
        return Z
    if d_used > MAX_D:
        raise ValueError(f"d_used={d_used} exceeds MAX_D={MAX_D}")
    pad = np.zeros((N, MAX_D - d_used), dtype=Z.dtype)
    return np.hstack([Z, pad])


def assert_matching_block_order_and_d(other_block_order_csv, block_ids, nominal_d_list, label="PCA"):
    """Cross-check this run's block order/nominal_d against another
    condition's block_order.csv (e.g. VAE checking against PCA's, or vice
    versa), so the two baselines can never silently drift apart on which
    blocks were used, in what order, or at what class/dimension.

    No-op (prints a note) if `other_block_order_csv` doesn't exist yet —
    e.g. the very first run in a fresh pipeline has nothing to compare to.
    """
    other_block_order_csv = str(other_block_order_csv)
    try:
        other = pd.read_csv(other_block_order_csv)
    except (FileNotFoundError, OSError):
        print(f"[latent_rule] no {label} block_order.csv at {other_block_order_csv} yet — skipping cross-check.")
        return
    if "nominal_d" not in other.columns:
        print(f"[latent_rule] {other_block_order_csv} has no nominal_d column (pre-Phase-1 format) — skipping cross-check.")
        return
    other_ids = other["block_id"].astype(str).tolist()
    other_d = other["nominal_d"].astype(int).tolist()
    assert other_ids == list(block_ids), (
        f"block order mismatch vs {label} ({other_block_order_csv}): "
        f"this run has {len(block_ids)} blocks, {label} has {len(other_ids)}; "
        f"first divergence at {_first_diff(other_ids, list(block_ids))}"
    )
    assert other_d == list(nominal_d_list), (
        f"nominal_d mismatch vs {label} ({other_block_order_csv}) for one or more blocks — "
        f"the latent-dimension rule must be identical across PCA and VAE."
    )
    print(f"[latent_rule] block order and nominal_d verified to match {label} ({other_block_order_csv}).")


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    return min(len(a), len(b)), None, None


def median_across_seeds(csv_paths, key_cols, out_path, seeds=None):
    """Read the same-shaped CSV from multiple seed runs, matched by
    key_cols (e.g. block_id) -- NOT by row position or pooled raw
    observations -- and write:
      * out_path: median PLUS range (_min/_max) and IQR (_iqr = Q75-Q25) for
        every numeric metric column, grouped by key_cols. Boolean/flag
        columns (e.g. complete_collapse, partial_collapse) are NOT
        numerically averaged into a fractional float under their own name --
        they get a separately, clearly named `<col>_frac_true` column
        instead. Every other categorical/provenance column (size_class,
        loss_type, gene, JSON-string diagnostics, ...) is carried through
        via `.first()` (verbatim from one seed), never aggregated.
      * a companion "<out_path stem>_by_seed.csv" with EVERY seed's raw
        value for every column, long-format (one row per key x seed), so
        seed-level values are always available alongside the median.
    """
    if not csv_paths:
        raise ValueError("no seed CSVs to aggregate")
    if seeds is None:
        seeds = list(range(len(csv_paths)))
    frames = []
    for seed, p in zip(seeds, csv_paths):
        df = pd.read_csv(p)
        # baseline_summary.csv already carries its own 'seed' column (the
        # run's seed) -- overwrite it with the authoritative label passed in
        # here rather than colliding on insert; training_summary.csv (no
        # prior 'seed' column) gets a fresh one inserted up front.
        if "seed" in df.columns:
            df["seed"] = seed
        else:
            df.insert(0, "seed", seed)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    by_seed_path = Path(str(out_path)).with_name(Path(str(out_path)).stem + "_by_seed.csv")
    all_df.to_csv(by_seed_path, index=False)

    data_cols = [c for c in all_df.columns if c not in key_cols and c != "seed"]
    bool_cols = [c for c in data_cols if pd.api.types.is_bool_dtype(all_df[c])]
    numeric_cols = [c for c in data_cols if c not in bool_cols and pd.api.types.is_numeric_dtype(all_df[c])]
    categorical_cols = [c for c in data_cols if c not in bool_cols and c not in numeric_cols]

    g = all_df.groupby(key_cols, as_index=False)

    # Establish one row per key, aligned with groupby's default sorted order.
    med = (
        g[numeric_cols].median()
        if numeric_cols
        else (
            all_df[key_cols]
            .drop_duplicates()
            .sort_values(key_cols)
            .reset_index(drop=True)
        )
    )

    parts = [med.reset_index(drop=True)]

    if numeric_cols:
        mn = g[numeric_cols].min()
        mx = g[numeric_cols].max()
        q25 = (
            all_df.groupby(key_cols)[numeric_cols]
            .quantile(0.25)
            .reset_index()
        )
        q75 = (
            all_df.groupby(key_cols)[numeric_cols]
            .quantile(0.75)
            .reset_index()
        )

        numeric_stats = pd.concat(
            [
                mn[numeric_cols].add_suffix("_min"),
                mx[numeric_cols].add_suffix("_max"),
                (q75[numeric_cols] - q25[numeric_cols]).add_suffix("_iqr"),
            ],
            axis=1,
        )
        parts.append(numeric_stats.reset_index(drop=True))

    if bool_cols:
        frac_true = g[bool_cols].mean()
        bool_stats = frac_true[bool_cols].add_suffix("_frac_true")
        parts.append(bool_stats.reset_index(drop=True))

    if categorical_cols:
        first_vals = g[categorical_cols].first()
        parts.append(
            first_vals[categorical_cols].reset_index(drop=True)
        )

    # Concatenate all generated columns once, avoiding DataFrame fragmentation.
    med = pd.concat(parts, axis=1)
    med["n_seeds"] = len(csv_paths)

    med.to_csv(out_path, index=False)
    return med
