#!/usr/bin/env python3
"""
pca_plausibility_summary.py — sanity check on the (single, fixed-split)
per_block_PCA_concat.py run: does the fixed per-size-class latent dim
explain a plausible fraction of each block's variance, or is `d` so far
below the block's real structure that the PCA floor (and therefore the
VAE comparison built on top of it) is not meaningful for that block?

Fixed-split design (see docs/baselines.md "Split semantics"): PCA is fit
EXACTLY ONCE, on the one split from `shared.split_seed` -- there is no
seed loop and nothing to vary across, so this reports the single run's
numbers directly (no median/IQR/range -- there is only one value per
block). Aggregating over VAE model_seeds is a separate, VAE-specific
concept (per_block_vae_concat.py's median_summary.csv /
median_training_summary.csv) and has no PCA analogue.

Reads one block_order.csv (pos, block_id, size_class, nominal_d, d_used,
var_explained_at_d, eff_dims_80pct) from the PCA run and prints/writes:
  * by-size-class var_explained_at_d / eff_dims_80pct summary (mirrors
    variance_by_size_class.csv, recomputed here so this script is a
    standalone plausibility check, not just a re-print)
  * blocks flagged LOW PLAUSIBILITY: var_explained_at_d below
    --min-var-explained (default 0.50) -- i.e. the assigned d captures less
    than half that block's variance.

The >500-SNP slice (large_blocks_gt500_summary.csv, written by
per_block_PCA_concat.py) and eff_dims_80pct stay purely descriptive here
too -- this script never uses either to choose or change `d`.

Not part of the PCA/VAE fairness comparison itself -- this is a diagnostic
report only, run manually against the completed PCA output directory.
"""
import argparse
from pathlib import Path

import pandas as pd


def summarize(block_order_csv, min_var_explained=0.50):
    df = pd.read_csv(block_order_csv)
    by_class = (df.groupby("size_class", as_index=False)
                .agg(n_blocks=("block_id", "count"),
                     nominal_d=("nominal_d", "first"),
                     mean_var_explained_at_d=("var_explained_at_d", "mean"),
                     median_var_explained_at_d=("var_explained_at_d", "median"),
                     mean_eff_dims_80pct=("eff_dims_80pct", "mean"),
                     median_eff_dims_80pct=("eff_dims_80pct", "median")))
    low = df[df["var_explained_at_d"] < min_var_explained].copy()
    low = low.sort_values("var_explained_at_d")
    return by_class, low


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("block_order_csv", help="path to the PCA run's block_order.csv "
                                             "(e.g. results/GACRS/baseline_pca/block_order.csv)")
    ap.add_argument("--min-var-explained", type=float, default=0.50,
                     help="flag blocks below this var_explained_at_d as low plausibility (default 0.50)")
    ap.add_argument("--out", default=None,
                     help="optional CSV path to write the by-size-class summary (default: alongside input)")
    args = ap.parse_args()

    by_class, low = summarize(args.block_order_csv, args.min_var_explained)

    print("═" * 70)
    print("PCA plausibility summary — variance explained by size class (single fixed-split run)")
    print("═" * 70)
    print(by_class.to_string(index=False))

    print(f"\nBlocks below var_explained_at_d={args.min_var_explained:.2f} (low plausibility): {len(low)}")
    if len(low):
        print(low[["block_id", "size_class", "nominal_d", "var_explained_at_d", "eff_dims_80pct"]].to_string(index=False))

    out = Path(args.out) if args.out else Path(args.block_order_csv).parent / "pca_plausibility_summary.csv"
    by_class.to_csv(out, index=False)
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
