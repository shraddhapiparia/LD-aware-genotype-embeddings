# LD-aware genotype embeddings

Does a nonlinear encoder trained on genotypes alone capture more phenotype-relevant signal from an LD block than PCA at the same dimensionality, and is any gain specific to asthma-associated regions?

## Status

Phase 1 is complete at the genotype level: per-block PCA and β-VAE encoders are built, trained and compared on reconstruction and latent-use diagnostics. Phenotype evaluation is next. No phenotype results are reported yet.

## Why LD blocks

SNPs within an LD block are strongly correlated, so a block carries far fewer independent signals than its SNP count suggests. Encoding each block separately keeps local haplotype structure intact, gives a natural unit for comparing methods, and keeps results traceable to a genomic region.

## Data

- GACRS pediatric asthma cohort, 997 subjects, one ancestry stratum.
- 464 LD blocks (Gabriel method) from 25 regions: 254 blocks in asthma-associated regions, 210 in control regions.
- Primary phenotype: IgE. Secondary: eosinophils, FEV, age of onset, G20D.
- Genotype and phenotype data are access-restricted and not included. `data/Synthetic/create_synthetic_data.py` generates synthetic data for testing.

## Design

An unsupervised encoder captures the largest sources of genotype variation, which in candidate-region data includes highly polymorphic regions like HLA whether or not they affect the phenotype. Two comparisons address this. Per-block PCA at the same latent dimension tests whether the VAE captures more than PCA does. Control regions test whether any advantage is specific to asthma regions.

Key decisions:

- **Matched capacity.** One rule sets the latent dimension for both methods: d = 2, 4 or 8 for blocks of ≤ 49, 50–199 and ≥ 200 SNPs.
- **Ordinal decoder.** Genotypes are modeled as ordered categories {0, 1, 2} with an ordinal likelihood.
- **Fixed in advance.** β = 0.5 and a free-bits floor of 0.5 nats per dimension were set before training. Other values are reported as sensitivity, not used for selection.
- **No phenotype in model building.** Checkpoints are chosen on validation reconstruction only, and no phenotype result informed any modeling choice.
- **One fixed split.** Both methods use the same 798 / 199 split. The VAE is trained with five seeds to measure training stability.
- **Continuous evaluation.** Asthma traits are polygenic and vary continuously, so the evaluation uses held-out partial R² per block instead of clustering subjects.

Full details of the architecture, loss, split handling and diagnostics are in [docs/baselines.md](docs/baselines.md).

## Results so far

β and free-bits sensitivity on 30 prespecified blocks, genotype-level metrics only:

| Condition | Median NLL/SNP | Median MSE | Distance preservation | Partial collapse |
|---|---|---|---|---|
| β = 0.5, λ = 0.5 (main) | 0.258 | 0.080 | 0.756 | 0/30 |
| β = 0.1, λ = 0.5 | 0.230 | 0.071 | 0.733 | 0/30 |
| β = 1, λ = 0.5 | 0.259 | 0.085 | 0.719 | 0/30 |
| β = 0.5, λ = 0 | 0.224 | 0.071 | 0.709 | 9/30 |
| β = 0.5, λ = 1 | 0.265 | 0.084 | 0.709 | 0/30 |
| Autoencoder (β = 0) | 0.185 | 0.058 | 0.681 | – |

The main setting kept every latent dimension active and preserved subject distances best. Removing free bits caused partial collapse in 9 of 30 blocks. The autoencoder reconstructed best but preserved subject structure worst.

## Next: phenotype evaluation

1. Held-out partial R² from adding each block's embedding to a model with ancestry PCs, with ancestry adjustment fit inside each training fold.
2. Asthma-minus-control difference for each method, adjusted for block size.
3. VAE versus PCA at matched dimension, with a paired bootstrap over subjects and a permutation test over the 25 regions, reported overall and by size class.
4. A difference of 0.01 in partial R² is treated as the smallest meaningful effect, set in advance.

## Repository layout

```
config/                    Run configurations
docs/baselines.md          Detailed methods
scripts/
  preprocessing/           Cohort QC and regional block extraction
  haploblock_derivation/   Gabriel block definition
  baselines/               Per-block PCA and VAE, latent-dimension rule, summaries
  evaluation/              Ancestry evaluation, clustering diagnostic
tests/                     Unit and end-to-end tests on synthetic data
data/Synthetic/            Synthetic data generator
```

`data/`, `metadata/` and `results/` hold restricted or generated files and are not tracked.

## Running

```bash
python scripts/baselines/per_block_PCA_concat.py   # run first; creates the split
python scripts/baselines/per_block_vae_concat.py   # five seeds, median summary
pytest tests/
```

See [docs/baselines.md](docs/baselines.md#running) for sensitivity runs and outputs.

## Limitations

- Block boundaries were defined on all subjects; model fitting and statistics are train-only.
- Asthma and control blocks differ in size, which the evaluation adjusts for but can't fully remove.
- Single cohort and ancestry stratum, with no external validation yet.
- At this sample size, only within-block effects are detectable.

## License

MIT