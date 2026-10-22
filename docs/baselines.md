# Phase 1 baselines: per-block PCA and β-VAE

This document describes how the two per-block encoders are built, trained and compared. The README gives the scientific question and current status; this is the detailed reference.

The two encoders are designed so that the encoder is the only thing that differs between them:

| Step | Script | Encoder |
|---|---|---|
| 1 | `scripts/baselines/per_block_PCA_concat.py` | Per-block PCA (linear baseline) |
| 2 | `scripts/baselines/per_block_vae_concat.py` | Per-block β-VAE |

Both use the same blocks, the same train/validation split, the same latent dimension per block, and the same output layout. A cross-block model that combines block embeddings is planned for Phase 2 and is not in this repository yet.

## Blocks and units

- 464 LD blocks, defined with the Gabriel method in PLINK on all 997 subjects.
- Blocks come from 25 genomic regions, labeled as asthma-associated or control. 254 blocks fall in asthma regions and 210 in control regions.
- Size class is computed from each block's SNP count `p`. The manifest's own `effective_haps` and `block_size_class` columns are not used.

## Latent dimension

One rule, in `scripts/baselines/latent_rule.py`, is imported by both encoders so they can't diverge:

| SNP count `p` | Size class | Nominal `d` |
|---|---|---:|
| ≤ 49 | small | 2 |
| 50–199 | medium | 4 |
| ≥ 200 | large | 8 |

The dimension actually used is `d_used = min(nominal_d, p, n_train − 1)`. It drops below `nominal_d` only for a degenerate block or split.

Each block's embedding is zero-padded to `MAX_D = 8` so the concatenated outputs have a fixed width. `all_blocks_latent_dims.npy` records the real `d_used` for each block. Both encoders write `size_class`, `nominal_d` and `d_used` to `block_order.csv`, keep the manifest's row order, and every VAE run checks its block order and `nominal_d` against the PCA run (`latent_rule.assert_matching_block_order_and_d`).

Two alternatives are out of scope for Phase 1:

- **Setting `d` from effective haplotype count.** On these blocks it correlates 0.95 with 1/r̄² and 0.10 with `p`, so it mostly reflects mean LD rather than dimensionality. A variable-`d` comparison is planned for Phase 2.
- **A fourth size class for very large blocks.** `large_blocks_gt500_summary.csv` reports blocks above 500 SNPs separately so this can be revisited.

## PCA

- Fit on training subjects only, then applied to all subjects.
- Fit once. PCA has no random initialization, so with a fixed split there is nothing to vary across seeds. `random_state` only affects the sign of near-degenerate components.
- `block_order.csv` reports `var_explained_at_d` and `eff_dims_80pct` (dimensions needed for 80% of variance on training subjects). `eff_dims_80pct` is reported but never used to choose `d`; doing so would give PCA a different budget from the VAE.
- `variance_by_size_class.csv` summarizes both by size class.
- `pca_plausibility_summary.py` flags blocks whose `var_explained_at_d` falls below a threshold (default 0.50).

## β-VAE

### Data flow

Manifest (blocks with `status == ok`) → PLINK `.raw` per block → subjects common to all blocks → one VAE per block, trained on training subjects → posterior mean μ for all subjects → zero-padded stack → concatenation.

Data loading and the split come from `scripts/baselines/data_utils.py`; the latent rule and layer widths from `scripts/baselines/latent_rule.py`.

### Architecture

Two `Linear → LayerNorm → GELU → Dropout(0.30)` layers on each side, with widths set from the block's `p` and `d_used`:

```
h1 = round(p^(2/3) · d^(1/3))
h2 = round(p^(1/3) · d^(2/3))
```

Encoder: `p → h1 → h2 → (μ, log σ²)`. Decoder: `d → h2 → h1 → ordinal head`. LayerNorm is used instead of BatchNorm so that a final minibatch of size 1 trains normally. Optimizer: Adam.

### Ordinal decoder and reconstruction loss

Genotypes are treated as ordered categories. For each SNP the decoder predicts two thresholds `t0 ≤ t1` (the gap is kept positive with a softplus), giving `P(G=0)`, `P(G=1)` and `P(G=2)`. Probabilities are clamped only immediately before the log.

The loader accepts only finite integer genotypes in {0, 1, 2} and raises an error on anything else, such as imputed dosages, instead of rounding them.

The ordinal NLL is summed over SNPs and averaged over subjects, without dividing by `p`. Larger blocks therefore have a larger raw NLL; per-SNP metrics such as dosage MSE are reported separately.

An MSE decoder on raw dosage is still in the code for comparison but is not part of the Phase 1 results.

### KL term and free bits

KL is computed in closed form for each subject and latent dimension, averaged over the minibatch for each dimension, and floored per dimension:

```
rate = Σ_j max(0.5, mean_batch_KL_j)
loss = ordinal_NLL + β · rate
```

The floor of 0.5 nats per dimension means KL below that level isn't penalized. It doesn't force the model to use that capacity, so both `raw_total_kl` (no floor) and `free_bits_rate` (with floor) are reported.

Neither the NLL nor the KL is divided by `p`. Averaging reconstruction per SNP while summing KL over dimensions would weight KL roughly in proportion to `p` and push large blocks toward the prior.

### β schedule

β rises linearly from 0 to 0.5 over the first 50 epochs, then stays fixed. β = 0.5 was chosen before training and is not tuned. Other values are run as a sensitivity analysis and reported, but never used to change β.

### Checkpointing and early stopping

- Checkpoints are selected on validation ordinal NLL only, never on the β-weighted loss, so a higher β isn't penalized for KL pressure unrelated to reconstruction.
- The learning-rate scheduler, if used, also steps on validation ordinal NLL.
- Epochs 1–50 are warm-up. No checkpoint from this period can be selected, and early-stopping patience starts counting at epoch 51.
- The saved model produces the posterior mean μ, which is used for all downstream embeddings.

## Split and seeds

**One fixed split.** `shared.split_seed` creates the train/validation split once (798 / 199 subjects). The PCA script creates it through `data_utils.create_or_reuse_pca_split()` and writes `subjects.csv`, `train_idx.npy`, `val_idx.npy` and `split_meta.json` to the PCA output directory.

**The VAE reuses that split exactly.** The VAE script never creates a split. It reads PCA's split through `data_utils.require_pca_split()` and raises an error if the files are missing or the subject set differs.

**Model seeds affect training only.** Each entry in `shared.model_seeds` (five in the main configuration) sets weight initialization, minibatch order and latent sampling. It does not change which subjects are in training or validation. Tests check both directions:

- `test_changing_model_seed_does_not_change_split_indices`: every model seed loads the same split as PCA.
- `test_changing_model_seed_changes_rng_state`: model seeds do change initialization and minibatch order.
- `test_fixed_split_end_to_end_pca_once_vae_two_model_seeds`: runs both scripts on synthetic data and checks the split files and recorded seeds in every output.

Because the split is fixed, spread across model seeds measures how stable VAE training is, not how results depend on the sample of subjects. Median summaries report the median, min, max and IQR of each metric, and the fraction of seeds for which each boolean flag is true. A `_by_seed.csv` file keeps each seed's raw values, matched by `block_id`.

Sampling uncertainty is handled separately in the phenotype evaluation: a paired bootstrap over subjects, and a permutation test over the 25 regions.

Every VAE output records both `split_seed` and `model_seed`. PCA outputs record `split_seed` only.

## Diagnostics

Reported per block in `training_summary.csv`, on validation subjects:

| Group | Columns |
|---|---|
| Block | `p`, `size_class`, `nominal_d`, `d_used`, `h1`, `h2` |
| Latent use | `active_latent_dims`, `active_fraction`, `mu_var_per_dim`, `raw_kl_per_dim`, `raw_total_kl`, `free_bits_rate` |
| Collapse | `complete_collapse`, `partial_collapse` |
| Reconstruction | `va_ordinal_nll`, `va_dosage_mse`, `va_dosage_mae`, `va_exact_acc`, `va_balanced_acc` |
| Per-class recall | `va_recall_g0`, `va_recall_g1`, `va_recall_g2`, `va_maf_stratified_recall` |
| Structure | `subject_distance_preservation` |

Definitions:

- A latent dimension is **active** if Var(μ) on validation subjects exceeds 0.01.
- **Complete collapse:** no active dimensions. **Partial collapse:** fewer active dimensions than `nominal_d`.
- **Expected dosage** is `P(G=1) + 2·P(G=2)`, compared with the true genotype.
- **MAF-stratified recall** uses bins [0, 0.05), [0.05, 0.10), [0.10, 0.20), [0.20, 0.50], with MAF computed from training subjects.
- **Subject-distance preservation** measures how well pairwise distances between validation subjects in μ match their pairwise distances in the raw block genotypes, with scaling fit on training subjects (see `data_utils.py`). It replaces an earlier `ld_corr` column, which is now always NaN.

All diagnostics are computed after checkpoint selection and are never used for checkpointing, early stopping or choosing between models. Thresholds and MAF bins are set in the config file.

## Sensitivity analysis

β and the free-bits floor λ were varied one at a time on 30 blocks chosen in advance to cover block size, functional group and LD complexity, using the fixed split. Only genotype-level metrics were used.

| Condition | Median NLL/SNP | Median MSE | Distance preservation | Median raw KL | Partial collapse |
|---|---|---|---|---|---|
| β = 0.5, λ = 0.5 (main) | 0.258 | 0.080 | 0.756 | 4.10 | 0/30 |
| β = 0.1, λ = 0.5 | 0.230 | 0.071 | 0.733 | 7.21 | 0/30 |
| β = 1, λ = 0.5 | 0.259 | 0.085 | 0.719 | 3.28 | 0/30 |
| β = 0.5, λ = 0 | 0.224 | 0.071 | 0.709 | 3.79 | 9/30 |
| β = 0.5, λ = 1 | 0.265 | 0.084 | 0.709 | 4.75 | 0/30 |
| Autoencoder (β = 0) | 0.185 | 0.058 | 0.681 | – | – |

## Clustering

Both scripts also run GMM (diagonal covariance, chosen by BIC) and K-means (chosen by silhouette) on the standardized concatenated embedding, using `scripts/evaluation/clustering.py`. This is kept as a descriptive diagnostic. It is not the evaluation of either encoder; that is the held-out phenotype analysis described in the README.

## Running

Both scripts read `config/baseline_phase1.yaml` by default and accept `--config` to use another file. Run PCA first; the VAE script needs PCA's split.

```bash
# PCA: one fit on the fixed split
python scripts/baselines/per_block_PCA_concat.py

# VAE: β = 0.5, five model seeds, median summary
python scripts/baselines/per_block_vae_concat.py

# β and free-bits sensitivity on the 30-block panel
python scripts/baselines/per_block_vae_concat.py --config config/sensitivity.yaml

# PCA plausibility summary
python scripts/baselines/pca_plausibility_summary.py results/GACRS/baseline_pca/block_order.csv
```

Each VAE configuration writes to its own output directory, with one `seed_<model_seed>/` subdirectory per seed. All configurations share the same PCA output directory, so PCA only needs to run once.

## Outputs

**PCA** (one directory, no seed subdirectories): split files (`subjects.csv`, `train_idx.npy`, `val_idx.npy`, `split_meta.json`), `block_order.csv`, `variance_by_size_class.csv`, `large_blocks_gt500_summary.csv`, `embeddings/` (`concat_embedding.npy`, `all_blocks.npy`, `all_blocks_latent_dims.npy`), `clustering/`, `baseline_summary.csv`, `baseline_meta.json`.

**VAE** (`seed_<model_seed>/` for each seed): the same embedding and clustering files, plus `training_summary.csv`, per-epoch logs in `logs/`, and checkpoints in `models/`. At the top level: `median_summary.csv`, `median_training_summary.csv`, and their `_by_seed.csv` versions.

## Limitations

- Block boundaries were defined on all 997 subjects. Model fitting and all reported statistics use training subjects for fitting and validation subjects for evaluation.
- Asthma and control blocks differ in size. The phenotype comparison adjusts for SNP count, but the imbalance can't be fully removed.
- At n ≈ 1,000, the analysis can only detect effects within a block. Interactions between blocks are out of reach.