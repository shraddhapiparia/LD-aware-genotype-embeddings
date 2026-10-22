#!/usr/bin/env python3
"""
per_block_vae_concat.py — Baseline: per-block VAE + concat + clustering.

Role in the ladder (see per_block_PCA_concat.py header for the full ladder
and fairness invariants — this file follows the same ones):
  per-block PCA + concat -> per_block_PCA_concat.py (the linear floor)
  per-block VAE + concat -> THIS FILE
  per-block VAE + attn   -> existing VAE_phase1.py + attention_phase2.py (blockbased-genotype-embedding-analysis)

Phase-1 finalized design
-------------------------
* **Latent dim.** One rule, shared verbatim with the PCA baseline
  (scripts/baselines/latent_rule.py): p<=49 -> d=2, 50-199 -> d=4, p>=200 -> d=8.
  Every per-block latent is zero-padded to latent_rule.MAX_D (8) for the
  concat/all_blocks tensors; all_blocks_latent_dims.npy still records the
  real (unpadded) d_used.
* **Architecture.** Exactly two Linear/LayerNorm/GELU/Dropout(0.30) layers
  per side, widths from latent_rule.hidden_widths(p, d) — NOT hard-coded:
  h1=round(p^(2/3) d^(1/3)), h2=round(p^(1/3) d^(2/3)). Encoder p->h1->h2->
  {mu,logvar}; decoder mirrors it, d->h2->h1->(reconstruction head). Adam.
* **ORD is the finalized primary objective** (loss_type MSE remains for
  sensitivity/legacy comparison only, see docs/baselines.md):
    - genotype input must be finite integers in {0,1,2} (validated, not
      silently rounded).
    - ordinal NLL, summed over SNPs, averaged over subjects — no /p.
    - KL is computed RAW per subject/dimension, averaged over the minibatch
      per dimension, then free-bits floored: rate = sum_j max(0.5, mean_KL_j).
    - loss = ordinal_NLL + beta * rate. beta linearly warms up 0->beta_max
      through epoch `beta_warmup` (main beta_max=0.5, warmup=50), then held.
* **Early stopping / checkpointing** monitor validation ordinal NLL ONLY
  (never the beta-weighted loss). Eligibility and patience counting do not
  start until epoch `checkpoint_start_epoch` (default 51) — epochs 1-50 are
  a pure KL warm-up runway and are never candidates for "best" or an early
  stop. The LR scheduler (if kept) also steps on validation ordinal NLL.
  The saved checkpoint holds the encoder's posterior MEAN (mu), used for
  every downstream embedding.
* **Fixed-split design, five model_seeds** (audited; see docs/baselines.md
  "Split semantics"). `shared.split_seed` is the ONE seed used to create the
  train/validation split — PCA fits it once; this script NEVER creates or
  regenerates a split, it only reuses PCA's byte-identically (fails loudly,
  via `require_pca_split()`, if PCA's split files are missing or the
  subject set doesn't match — no silent fallback). `shared.model_seeds`
  (five in the finalized config) is VAE-ONLY: it seeds weight init,
  minibatch order, and latent sampling (`set_seed(model_seed)`), and has
  no effect on which subjects land in train/val. Each model_seed runs this
  whole pipeline once into `<output_root>/seed_<model_seed>/`, never
  overwriting another model_seed's outputs, then writes a
  median-across-model_seed `<output_root>/median_training_summary.csv` and
  `<output_root>/median_summary.csv` (median + min/max/IQR + a `_by_seed.csv`
  companion, matched by block_id — quantifying VAE TRAINING/init stability,
  not split variability, since the split never changes). beta in {0.25, 1.0}
  are REPORTED sensitivity runs around the main beta=0.5; beta=0 is an AE
  ablation. Neither is ever used to select beta — that stays fixed at 0.5.

Fairness vs. PCA (must hold or the comparison is confounded):
  * latent dim per block from the SAME rule as PCA (latent_rule.py), same
    class boundaries, same MAX_D padding.
  * VAE fit on TRAIN subjects only, mu extracted for ALL subjects (no leakage).
  * train/val split is REQUIRED to be reused from vae.phase1_dir (the PCA
    baseline's output_root) via require_pca_split(), imported from
    scripts/baselines/data_utils.py (shared with per_block_PCA_concat.py) —
    every model_seed reuses the exact same split; block order and nominal_d
    are asserted to match the PCA baseline's block_order.csv.
  * clustering is the identical cluster() routine (GMM+BIC primary,
    KMeans+silhouette parity), imported from scripts/evaluation/clustering.py
    rather than reimplemented.

Outputs (under vae.output_root/seed_<model_seed>/ -- always nested, one dir per model_seed)
  subjects.csv, train_idx.npy, val_idx.npy   (copied from PCA's split; never regenerated)
  block_order.csv                             (pos, block_id, n_snps, size_class,
                                               nominal_d, d_used, h1, h2, loss_type,
                                               split_seed, model_seed)
  training_summary.csv                        (per-block training + full Phase-1 eval report;
                                               includes split_seed, model_seed columns)
  logs/<block>.csv                            per-epoch training history (split_seed, model_seed columns)
  models/<block>.pt                           best checkpoint + metadata (state_dict, loss_type,
                                               split_seed, model_seed, ...)
  embeddings/concat_embedding.npy             (N, B*MAX_D)  -> clustering input
  embeddings/all_blocks.npy                   (N, B, MAX_D) padded, drop-in for Phase 2
  embeddings/all_blocks_latent_dims.npy       (B,) actual d_used per block (<=nominal_d<=MAX_D)
  embeddings/per_block/<block>.npy            per-block mu (if save_per_block)
  clustering/gmm_labels.csv, gmm_bic.csv
  clustering/kmeans_labels.csv, kmeans_silhouette.csv
  baseline_summary.csv, baseline_meta.json    (both record split_seed AND model_seed)
  median_summary.csv (+_by_seed.csv), median_training_summary.csv (+_by_seed.csv)  (top-level output_root only)

Config-driven (config/baseline.yaml): no experiment CLI flags, only an
optional --config path override.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from data_utils import (
    load_block_defs, load_all_blocks, require_pca_split, load_baseline_config, DEFAULT_CONFIG,
    resolve_model_seeds, model_seed_output_dir, read_split_seed, compute_maf, maf_bin_labels,
    resolve_run_clustering, class_support, subject_distance_preservation,
)
from latent_rule import (
    latent_dim_for_block, hidden_widths, pad_to_max_d, MAX_D,
    assert_matching_block_order_and_d, median_across_seeds,
)
from clustering import cluster


# ──────────────────────────────────────────────────────────────────
# seed / device
# ──────────────────────────────────────────────────────────────────
def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.zeros(2, device="mps")
            print("[device] Apple MPS")
            return torch.device("mps")
        except Exception:
            pass
    if torch.cuda.is_available():
        print("[device] CUDA")
        return torch.device("cuda")
    print("[device] CPU")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────
# ordinal head — ported verbatim (parameterization + NLL) from
# VAE_phase1.py's ORD loss_type, restricted to the ORD case only.
# ──────────────────────────────────────────────────────────────────
def ordinal_probs(logits2):
    """
    logits2: (..., P, 2) — [:,:,0]=t0, [:,:,1]=raw delta; t1 = t0 + softplus(delta)
    enforces t1 >= t0 (monotonic thresholds). Returns (p_cls, t0, t1) where
    p_cls stacks the UNCLAMPED [P(G=0), P(G=1), P(G=2)] and sums to 1 by
    construction (up to float precision) — no clamping here, so callers that
    need the true probability simplex (e.g. check_ordinal_probs) see it
    unperturbed. Callers that take log() of these must clamp immediately
    before doing so (see ordinal_loss).
    """
    t0 = logits2[..., 0]
    t1 = t0 + F.softplus(logits2[..., 1])
    p0 = torch.sigmoid(t0)
    p1 = torch.sigmoid(t1)
    p_cls = torch.stack([p0, p1 - p0, 1 - p1], dim=-1)
    return p_cls, t0, t1


def ordinal_nll_per_subject(logits2, targets):
    """Numerically stable ordinal NLL, summed over SNPs, PER SUBJECT (B,) —
    no /p, no /B here; callers reduce as needed (mean over subjects for the
    training objective, as-is for per-subject reporting)."""
    p_cls, _, _ = ordinal_probs(logits2)
    log_p = p_cls.clamp_min(1e-6).log().gather(-1, targets.clamp(0, 2).unsqueeze(-1)).squeeze(-1)
    return -log_p.sum(dim=-1)  # sum over SNPs -> (B,)


def ordinal_loss(logits2, targets):
    """Ordinal NLL summed over SNPs, AVERAGED over subjects — no /p."""
    return ordinal_nll_per_subject(logits2, targets).mean()


def check_ordinal_probs(p_cls, t0, t1, tol=1e-3):
    """Sanity checks on the UNCLAMPED probability simplex from ordinal_probs():
    finite, nonnegative, sum-to-one, threshold ordering."""
    arr = p_cls.detach().cpu().numpy()
    assert np.isfinite(arr).all(), "ordinal probs contain NaN/Inf"
    assert (arr >= -1e-6).all(), "ordinal probs have negative entries"
    s = arr.sum(axis=-1)
    assert np.allclose(s, 1.0, atol=tol), f"ordinal probs don't sum to 1 (max dev {np.max(np.abs(s-1)):.2e})"
    t0a, t1a = t0.detach().cpu().numpy(), t1.detach().cpu().numpy()
    assert (t1a >= t0a - 1e-6).all(), "ordinal thresholds not ordered (t1 < t0)"


def validate_integer_genotypes(G, block_id, tol=1e-6):
    """ORD requires finite, integer-valued genotype calls in {0,1,2} —
    refuse to silently round soft/imputed dosages or accept NaN/Inf."""
    if not np.isfinite(G).all():
        raise ValueError(f"{block_id}: non-finite genotype value(s) found; ORD requires finite {{0,1,2}} calls.")
    if not np.allclose(G, np.round(G), atol=tol):
        bad = int((np.abs(G - np.round(G)) > tol).sum())
        raise ValueError(
            f"{block_id}: vae.loss_type ORD requires integer-valued genotypes in {{0,1,2}}; "
            f"found {bad} non-integer dosage value(s) (soft-called/imputed data is not supported here)."
        )
    if G.min() < -tol or G.max() > 2 + tol:
        raise ValueError(f"{block_id}: genotype values out of {{0,1,2}}: min={G.min()}, max={G.max()}")


# ──────────────────────────────────────────────────────────────────
# free-bits KL — raw per-subject/dimension KL, averaged per-dimension
# over the minibatch, THEN floored per dimension (point 3).
# ──────────────────────────────────────────────────────────────────
def raw_kl_per_dim(mu, lv):
    """(B, d) raw KL per subject/dimension — no free-bits clamp."""
    return -0.5 * (1 + lv - mu.pow(2) - lv.exp())


def free_bits_rate(mu, lv, free_bits):
    """mean_batch_KL_j = mean over subjects of raw KL, per dimension j.
    rate = sum_j max(free_bits, mean_batch_KL_j) — the training objective's
    KL term. raw_kl_total = sum_j mean_batch_KL_j with NO floor — a
    diagnostic only (point 6), always <= rate."""
    kl_bd = raw_kl_per_dim(mu, lv)                 # (B, d)
    mean_kl_dim = kl_bd.mean(dim=0)                # (d,) average each dim over the minibatch
    raw_kl_total = mean_kl_dim.sum()
    rate = torch.clamp(mean_kl_dim, min=free_bits).sum()
    return rate, mean_kl_dim, raw_kl_total


# ──────────────────────────────────────────────────────────────────
# model — two Linear/LayerNorm/GELU/Dropout layers per side, widths
# from latent_rule.hidden_widths(p, d). LayerNorm (not BatchNorm1d)
# so training/eval batches — including the final batch of size 1 —
# don't depend on other samples in the batch.
# ──────────────────────────────────────────────────────────────────
class BlockVAE(nn.Module):
    def __init__(self, p, d, drop=0.30, loss_type="ORD", objective_mode="VAE"):
        super().__init__()
        if objective_mode not in ("VAE", "AE"):
            raise ValueError(f"objective_mode must be VAE or AE (got {objective_mode!r})")
        self.p, self.d, self.loss_type, self.objective_mode = p, d, loss_type, objective_mode
        h1, h2 = hidden_widths(p, d)
        self.h1, self.h2 = h1, h2

        self.enc = nn.Sequential(
            nn.Linear(p, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(drop),
        )
        self.mu = nn.Linear(h2, d)
        self.lv = nn.Linear(h2, d)

        out_width = p * 2 if loss_type == "ORD" else p
        self.dec = nn.Sequential(
            nn.Linear(d, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h2, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h1, out_width),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.lv(h).clamp(-20, 2)

    def reparam(self, mu, lv):
        if self.objective_mode == "AE":
            # true deterministic autoencoder: z=mu always, train and eval alike
            # -- no reparameterization noise injected at any point.
            return mu
        return mu + torch.randn_like(mu) * (0.5 * lv).exp() if self.training else mu

    def decode(self, z):
        out = self.dec(z)
        if self.loss_type == "ORD":
            return out.view(out.size(0), self.p, 2)
        return out

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return self.decode(z), mu, lv

    def compute_loss(self, x, recon, mu, lv, beta, free_bits, y=None):
        if self.loss_type == "ORD":
            recon_metric = ordinal_loss(recon, y)     # ordinal NLL, sum-over-SNPs/mean-over-subjects, no /p
        else:
            # sum over SNPs / mean over subjects -- same scale as ordinal_loss,
            # so beta imposes the same KL/reconstruction tradeoff in both loss types
            recon_metric = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        # rate/mean_kl_dim/raw_kl_total are always computed (diagnostic value
        # for AE mode too), but AE's optimized loss is reconstruction ONLY --
        # no KL/free-bits term, no beta warm-up pressure. The KL numbers
        # reported for an AE run describe an un-optimized posterior and are
        # not directly interpretable the way they are for VAE (see docs).
        rate, mean_kl_dim, raw_kl_total = free_bits_rate(mu, lv, free_bits)
        if self.objective_mode == "AE":
            loss = recon_metric
        else:
            loss = recon_metric + beta * rate
        return loss, recon_metric, rate, mean_kl_dim, raw_kl_total


# ──────────────────────────────────────────────────────────────────
# training loop (single block)
# ──────────────────────────────────────────────────────────────────
def train_block(model, tr_x, va_x, device, cfg, log_csv, model_seed, tr_y=None, va_y=None, split_seed=None,
                 maf=None, maf_bins=None):
    """`model_seed` seeds ONLY the minibatch-shuffle generator here (weight
    init/latent sampling are seeded by the caller's set_seed(model_seed)
    before this is called). `split_seed` is recorded in the log for
    provenance only -- it plays no role in training.

    Every `cfg["diagnostic_interval"]` epochs (item 1; NaN on the epochs in
    between), a full model.eval()/no_grad pass over ALL of train and ALL of
    validation (not the stochastic per-minibatch training loss) is logged
    via evaluate_split() -- so tr_* and va_* columns are actually
    comparable, plus nll_per_snp_gap/balanced_acc_gap. This read-only pass
    happens strictly AFTER the optimizer step for that epoch and never
    influences gradients, the LR scheduler, checkpoint selection, or
    patience (all of which read only tl/v_recon, computed exactly as
    before). At the end, the row for `best_epoch` is overwritten with a
    fresh evaluate_split() pass on the RESTORED best checkpoint, so that
    row's diagnostics are never stale/NaN even if best_epoch didn't land on
    a diagnostic_interval multiple."""
    diagnostic_interval = int(cfg.get("diagnostic_interval", 10)) or None
    tr_ds = TensorDataset(tr_x) if tr_y is None else TensorDataset(tr_x, tr_y)
    va_ds = TensorDataset(va_x) if va_y is None else TensorDataset(va_x, va_y)
    g = torch.Generator()
    g.manual_seed(model_seed)
    tr_dl = DataLoader(tr_ds, cfg["batch_size"], shuffle=True, generator=g)
    va_dl = DataLoader(va_ds, cfg["batch_size"])

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-6)

    checkpoint_start = cfg["checkpoint_start_epoch"]
    free_bits = cfg["free_bits"]
    best_val_recon, best_epoch, wait, best_sd, log = float("inf"), None, 0, None, []
    for ep in range(1, cfg["epochs"] + 1):
        beta = min(cfg["beta_max"], cfg["beta_max"] * ep / max(cfg["beta_warmup"], 1))

        model.train()
        tl = tn = 0.0
        for batch in tr_dl:
            x, y = (batch[0], batch[1]) if tr_y is not None else (batch[0], None)
            x = x.to(device)
            y = y.to(device) if y is not None else None
            rec, mu, lv = model(x)
            loss, recon_metric, rate, mean_kl_dim, raw_kl_total = model.compute_loss(x, rec, mu, lv, beta, free_bits, y=y)
            assert torch.isfinite(loss), f"non-finite training loss at epoch {ep}"
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tl += loss.item() * x.size(0)
            tn += x.size(0)
        tl /= tn

        model.eval()
        vl = vn = 0.0
        v_recon = 0.0
        v_raw_kl_total = 0.0
        with torch.no_grad():
            for batch in va_dl:
                x, y = (batch[0], batch[1]) if va_y is not None else (batch[0], None)
                x = x.to(device)
                y = y.to(device) if y is not None else None
                rec, mu, lv = model(x)
                loss, recon_metric, rate, mean_kl_dim, raw_kl_total = model.compute_loss(x, rec, mu, lv, beta, free_bits, y=y)
                assert torch.isfinite(loss), f"non-finite validation loss at epoch {ep}"
                vl += loss.item() * x.size(0)
                v_recon += recon_metric.item() * x.size(0)
                v_raw_kl_total += raw_kl_total.item() * x.size(0)
                vn += x.size(0)
        vl /= vn
        v_recon /= vn            # validation "recon_metric" — ordinal NLL for ORD, MSE for MSE mode
        v_raw_kl_total /= vn
        sched.step(v_recon)      # LR scheduler also monitors the primary recon metric, not the beta-weighted loss

        # -- item 1: comparable train/val diagnostics, every diagnostic_interval
        # epochs, on a full model.eval()/no_grad pass (never the stochastic
        # per-minibatch tr_loss above). Read-only: computed AFTER the
        # optimizer step and AFTER sched.step()/checkpoint bookkeeping use
        # v_recon, so it cannot affect either. NaN on non-diagnostic epochs.
        if diagnostic_interval and (ep % diagnostic_interval == 0):
            model.cpu()
            tr_diag = evaluate_split(model, tr_x, tr_y, model.loss_type, free_bits, maf, maf_bins)
            va_diag = evaluate_split(model, va_x, va_y, model.loss_type, free_bits, maf, maf_bins)
            model.to(device)
            diag_row = {f"tr_{k}": v for k, v in tr_diag.items() if k in DIAG_KEYS}
            diag_row.update({f"va_{k}": v for k, v in va_diag.items() if k in DIAG_KEYS})
            diag_row["nll_per_snp_gap"] = va_diag["ordinal_nll_per_snp"] - tr_diag["ordinal_nll_per_snp"]
            diag_row["balanced_acc_gap"] = tr_diag["balanced_acc"] - va_diag["balanced_acc"]
        else:
            diag_row = _nan_diag_row()

        log.append(dict(epoch=ep, beta=round(beta, 4), tr_loss=round(tl, 4),
                         va_loss=round(vl, 4), va_recon_metric=round(v_recon, 4),
                         va_raw_kl_total=round(v_raw_kl_total, 4), lr=opt.param_groups[0]["lr"],
                         split_seed=split_seed, model_seed=model_seed,
                         is_best_checkpoint=False, **diag_row))

        # checkpoint eligibility / patience counting only start at epoch
        # checkpoint_start (default 51) — epochs before that are a pure KL
        # warm-up runway, never a candidate "best" and never counted against patience.
        if ep >= checkpoint_start:
            if v_recon < best_val_recon:
                best_val_recon, best_epoch, wait = v_recon, ep, 0
                best_sd = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            else:
                wait += 1
            if wait >= cfg["patience"]:
                break

    if best_sd is None:
        # epochs never reached checkpoint_start (e.g. cfg["epochs"] < checkpoint_start)
        # -- fall back to the final epoch's weights so training still yields a usable model.
        best_sd = {k: w.cpu().clone() for k, w in model.state_dict().items()}
        best_val_recon, best_epoch = v_recon, ep
        print(f"  [warn] training ended before checkpoint_start_epoch={checkpoint_start}; "
              f"using final-epoch weights (no eligible checkpoint was ever selected).")

    model.load_state_dict(best_sd)          # restore best checkpoint before any eval/embedding
    model.to("cpu")

    # -- item 1: evaluate the RESTORED best checkpoint (never a stale
    # mid-training snapshot) and overwrite that epoch's log row so it is
    # never left NaN just because best_epoch didn't land on a
    # diagnostic_interval multiple.
    final_tr_diag = evaluate_split(model, tr_x, tr_y, model.loss_type, free_bits, maf, maf_bins)
    final_va_diag = evaluate_split(model, va_x, va_y, model.loss_type, free_bits, maf, maf_bins)
    for row in log:
        if row["epoch"] == best_epoch:
            row.update({f"tr_{k}": v for k, v in final_tr_diag.items() if k in DIAG_KEYS})
            row.update({f"va_{k}": v for k, v in final_va_diag.items() if k in DIAG_KEYS})
            row["nll_per_snp_gap"] = final_va_diag["ordinal_nll_per_snp"] - final_tr_diag["ordinal_nll_per_snp"]
            row["balanced_acc_gap"] = final_tr_diag["balanced_acc"] - final_va_diag["balanced_acc"]
            row["is_best_checkpoint"] = True
            break

    pd.DataFrame(log).to_csv(log_csv, index=False)
    return log, best_val_recon, best_epoch


# ──────────────────────────────────────────────────────────────────
# embedding + evaluation
# ──────────────────────────────────────────────────────────────────
def extract_mu(model, G):
    model.eval()
    x = torch.tensor(G, dtype=torch.float32)
    with torch.no_grad():
        mu, _ = model.encode(x)
    return mu.numpy()


def eval_mse_metrics(model, x):
    """Validation dosage MSE/MAE on the original 0-2 scale (no inverse transform needed)."""
    model.eval()
    with torch.no_grad():
        rec, _, _ = model(x)
    pred, truth = rec.numpy(), x.numpy()
    return {
        "va_ordinal_nll": np.nan,
        "va_dosage_mse": float(np.mean((pred - truth) ** 2)),
        "va_dosage_mae": float(np.mean(np.abs(pred - truth))),
        "va_exact_acc": np.nan, "va_balanced_acc": np.nan,
        "va_recall_g0": np.nan, "va_recall_g1": np.nan, "va_recall_g2": np.nan,
        "va_maf_stratified_recall": "{}",
    }


def eval_ord_metrics(model, x, y, maf, maf_bins):
    """Expected-dosage MSE/MAE (E[G]=P(G=1)+2P(G=2)), exact/balanced genotype
    accuracy, per-class recall, MAF-stratified recall, and mean ordinal NLL —
    all on held-out validation subjects (point 6)."""
    model.eval()
    with torch.no_grad():
        rec, _, _ = model(x)
    p_cls, t0, t1 = ordinal_probs(rec)
    check_ordinal_probs(p_cls, t0, t1)
    nll_per_subj = ordinal_nll_per_subject(rec, y)
    p = p_cls.numpy()
    truth = y.numpy()
    exp_dosage = p[..., 1] + 2.0 * p[..., 2]
    truth_f = truth.astype(np.float32)

    pred_cls = p.argmax(axis=-1)
    exact_acc = float(np.mean(pred_cls == truth))
    accs = [float(np.mean(pred_cls[truth == g] == g)) for g in (0, 1, 2) if (truth == g).any()]
    bal_acc = float(np.mean(accs)) if accs else float("nan")
    recall_by_class = {}
    for g in (0, 1, 2):
        mask = truth == g
        recall_by_class[g] = float(np.mean(pred_cls[mask] == g)) if mask.any() else float("nan")

    # MAF-stratified recall: bin SNPs by train-set MAF, recall = exact-match
    # rate over (subject, SNP) pairs whose SNP falls in that bin.
    correct = (pred_cls == truth)
    labels = maf_bin_labels(maf_bins)
    maf_recall = {}
    for lo, hi, lab in zip(maf_bins[:-1], maf_bins[1:], labels):
        snp_mask = (maf >= lo) & (maf < hi if hi < maf_bins[-1] else maf <= hi)
        maf_recall[lab] = float(correct[:, snp_mask].mean()) if snp_mask.any() else float("nan")

    return {
        "va_ordinal_nll": float(nll_per_subj.mean().item()),
        "va_dosage_mse": float(np.mean((exp_dosage - truth_f) ** 2)),
        "va_dosage_mae": float(np.mean(np.abs(exp_dosage - truth_f))),
        "va_exact_acc": exact_acc, "va_balanced_acc": bal_acc,
        "va_recall_g0": recall_by_class[0], "va_recall_g1": recall_by_class[1], "va_recall_g2": recall_by_class[2],
        "va_maf_stratified_recall": json.dumps(maf_recall),
    }


def evaluate_split(model, x, y, loss_type, free_bits, maf=None, maf_bins=None):
    """model.eval()/no_grad evaluator applied IDENTICALLY to train and
    validation tensors (point 1) -- so the two are actually comparable
    (unlike the stochastic per-minibatch tr_loss from training). Reuses
    eval_ord_metrics/eval_mse_metrics (no duplicated metric logic), strips
    their 'va_' prefix (generic here -- caller re-prefixes tr_/va_), and adds
    raw KL + NLL-per-SNP on top. Never called anywhere that affects
    checkpoint selection, the LR scheduler, patience, or gradients -- purely
    a read-only report."""
    if loss_type == "ORD":
        ev = eval_ord_metrics(model, x, y, maf, maf_bins)
    else:
        ev = eval_mse_metrics(model, x)
    ev = {(k[3:] if k.startswith("va_") else k): v for k, v in ev.items()}
    model.eval()
    with torch.no_grad():
        _, mu, lv = model(x)
        _, mean_kl_dim, raw_kl_total = free_bits_rate(mu, lv, free_bits)
    ev["raw_total_kl"] = float(raw_kl_total.item())
    ev["raw_kl_per_dim"] = json.dumps([float(v) for v in mean_kl_dim.detach().cpu().numpy()])
    p = x.shape[1]
    ev["ordinal_nll_per_snp"] = float(ev["ordinal_nll"] / p) if loss_type == "ORD" else float("nan")
    ev.pop("maf_stratified_recall", None)
    return ev


DIAG_KEYS = ["ordinal_nll", "ordinal_nll_per_snp", "dosage_mse", "dosage_mae", "exact_acc",
             "balanced_acc", "recall_g0", "recall_g1", "recall_g2", "raw_total_kl", "raw_kl_per_dim"]


def _nan_diag_row():
    row = {}
    for prefix in ("tr_", "va_"):
        row.update({f"{prefix}{k}": np.nan for k in DIAG_KEYS})
    row["nll_per_snp_gap"] = np.nan
    row["balanced_acc_gap"] = np.nan
    return row


def latent_diagnostics(model, x_va, active_var_threshold, collapse_active_dims_threshold, free_bits, nominal_d,
                        beta_max=None, validation_recon=None):
    """Per-dimension mu variance (collapse check) + raw/free-bits KL, on
    VALIDATION subjects only (point 6 — 'on held-out validation subjects').
    Diagnostic only -- collapse/active-dim flags must never feed checkpoint
    selection or model comparison (that stays on validation ordinal NLL alone,
    see train_block()).

    complete_collapse/partial_collapse are LEGACY Var(mu)-based flags -- kept
    for continuity, NOT redefined here, and NOT a definitive collapse
    measure: the exported downstream representation is the deterministic mu,
    and dims_above_free_bits==0 (the stochastic posterior's KL/free-bits
    crossing) does NOT imply mu is uninformative. Use
    all_dims_below_free_bits / any_dim_above_free_bits for the KL-side
    question; use complete_collapse/partial_collapse only as the pre-existing
    Var(mu) diagnostic they always were."""
    model.eval()
    with torch.no_grad():
        mu, lv = model.encode(x_va)
    mu_np = mu.numpy()
    assert np.isfinite(mu_np).all(), "non-finite mu encountered in latent diagnostics"
    var_per_dim = mu_np.var(axis=0)
    active_dims = int((var_per_dim > active_var_threshold).sum())
    d = mu_np.shape[1]  # unpadded d_used -- encode() never produces MAX_D-padded columns
    rate, mean_kl_dim, raw_kl_total = free_bits_rate(mu, lv, free_bits=free_bits)

    eps = 1e-8
    posterior_var_per_dim = lv.exp().mean(dim=0).detach().numpy()          # mean(exp(logvar), subjects)
    mu_var_np = mu_np.var(axis=0)
    latent_snr_per_dim = mu_var_np / np.maximum(posterior_var_per_dim, eps)  # Var(mu) / mean(posterior var)
    mean_kl_np = mean_kl_dim.detach().numpy()
    dims_above_free_bits = int((mean_kl_np > free_bits).sum())
    kl_to_recon_ratio = (float(beta_max) * float(rate.item()) / max(abs(float(validation_recon)), eps)
                          if beta_max is not None and validation_recon is not None else float("nan"))

    return {
        "d_used": d,
        "active_latent_dims": active_dims,
        "active_fraction": active_dims / d,
        # complete_collapse: zero active dimensions (the block learned nothing).
        # partial_collapse: fewer active dims than the block's CLASS budget
        # (nominal_d, not the clamped d_used) -- flags blocks using less of
        # their assigned capacity than intended, short of total collapse.
        "complete_collapse": bool(active_dims <= collapse_active_dims_threshold),
        "partial_collapse": bool(active_dims < nominal_d),
        "mu_var_per_dim": json.dumps([float(v) for v in var_per_dim]),
        "raw_kl_per_dim": json.dumps([float(v) for v in mean_kl_np]),
        "mu_var_min": float(var_per_dim.min()),
        "mu_var_mean": float(var_per_dim.mean()),
        "mu_var_max": float(var_per_dim.max()),
        "raw_total_kl": float(raw_kl_total.item()),
        "free_bits_rate": float(rate.item()),
        # -- new (item 4): stochastic-posterior diagnostics, on the SAME
        # validation encode() pass, kept distinct from the legacy Var(mu)
        # collapse flags above (never redefines them).
        "posterior_var_per_dim": json.dumps([float(v) for v in posterior_var_per_dim]),
        "latent_snr_per_dim": json.dumps([float(v) for v in latent_snr_per_dim]),
        "dims_above_free_bits": dims_above_free_bits,
        "fraction_dims_above_free_bits": dims_above_free_bits / d,
        "all_dims_below_free_bits": bool(dims_above_free_bits == 0),
        "any_dim_above_free_bits": bool(dims_above_free_bits > 0),
        "raw_kl_minus_free_capacity": float(raw_kl_total.item()) - free_bits * d,
        "kl_to_recon_ratio": kl_to_recon_ratio,
    }


def compute_ld_corr(G_va, mu_va):
    """DEPRECATED (item 5) -- the old rowwise np.corrcoef(mu) statistic this
    computed was replaced by subject_distance_preservation()
    (scripts/baselines/data_utils.py): a Spearman correlation between
    pairwise genotype/latent distances, with the standardizing mean/SD fit
    on TRAIN subjects (this function fit and applied its scaling on
    validation only -- a leakage-adjacent, circular choice). Kept as a
    no-op returning NaN rather than removed outright, so any external
    reader of the 'ld_corr' column doesn't silently get a stale statistic
    or a KeyError; do not resurrect the old computation here."""
    return float("nan")


def g2_calibration(model, x_va, y_va, maf_train, maf_bins):
    """Validation G2 (minor-homozygote) calibration by TRAIN-set MAF bin
    (item 2) -- at the restored best checkpoint, ORD only. Distinguishes a
    CALIBRATED low-confidence prediction (mean_predicted_p_g2 tracks
    observed_g2_frequency, both low -- honest, not a bug) from SYSTEMATIC
    underprediction (mean_predicted_p_g2 << observed_g2_frequency -- the
    model is wrong, not just uncertain), something recall_g2 alone can't
    distinguish. Bins are the SAME maf_bins/labels used for MAF-stratified
    recall (maf_bin_labels), so the two are directly comparable side by
    side; n_observations_by_maf_bin is the (subject x SNP) denominator
    each bin's numbers are computed over."""
    model.eval()
    with torch.no_grad():
        rec, _, _ = model(x_va)
    p_cls, _, _ = ordinal_probs(rec)
    p_cls_np = p_cls.numpy()
    y_np = y_va.numpy()
    p_g2 = p_cls_np[..., 2]
    nll_mat = -np.log(np.clip(np.take_along_axis(p_cls_np, y_np[..., None], axis=-1)[..., 0], 1e-6, None))
    is_g2 = (y_np == 2).astype(np.float64)

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
        nll_bin[lab] = float(nll_mat[:, snp_mask].mean())
        n_obs[lab] = int(is_g2[:, snp_mask].size)
    return {
        "observed_g2_frequency_by_maf_bin": json.dumps(obs),
        "mean_predicted_p_g2_by_maf_bin": json.dumps(pred),
        "g2_brier_score_by_maf_bin": json.dumps(brier),
        "ordinal_nll_per_snp_by_maf_bin": json.dumps(nll_bin),
        "n_observations_by_maf_bin": json.dumps(n_obs),
    }


# ──────────────────────────────────────────────────────────────────
# one model_seed's worth of the pipeline -- the split is FIXED (from PCA,
# via split_seed); only model_seed varies here (init/minibatch/latent
# sampling). See require_pca_split() in data_utils.py: this never creates
# or regenerates a split, and fails loudly if PCA's isn't reusable.
# ──────────────────────────────────────────────────────────────────
def run_one_model_seed(cfg, model_seed, split_seed, out, pca_root):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("embeddings", "logs", "models"):
        (out / sub).mkdir(exist_ok=True)

    # Sensitivity-sweep provenance (optional): threaded through the scalar
    # cfg dict rather than a new parameter, per the sweep's design -- ABSENT
    # for every ordinary (non-sweep) run, so this never changes existing
    # column sets/behavior. `_c()` below adds a key only when condition_id
    # is actually set.
    condition_id = cfg.get("condition_id")

    def _c(d):
        if condition_id is not None:
            d["condition_id"] = condition_id
        return d

    set_seed(model_seed)   # VAE-only: weight init, minibatch order, latent sampling
    device = get_device()

    print(f"\n══════ Loading data (loss_type={cfg['loss_type']}, split_seed={split_seed}, model_seed={model_seed}) ══════")
    bdf = load_block_defs(cfg["block_def"])
    blocks, subjects, block_ids = load_all_blocks(
        bdf, cfg["raw_dir"], use_manifest_path=cfg["use_manifest_path"])

    phase1_dir = cfg.get("phase1_dir")
    tr_ix, va_ix = require_pca_split(subjects, out, phase1_dir)  # fails loudly if unavailable/mismatched -- never regenerates

    # Explicit, visible cross-check (in addition to require_pca_split's
    # internal guarantee): THIS model_seed's split must be byte-identical
    # to what PCA itself has on disk, for every model_seed, every time.
    pca_tr = np.load(Path(phase1_dir) / "train_idx.npy")
    pca_va = np.load(Path(phase1_dir) / "val_idx.npy")
    assert np.array_equal(tr_ix, pca_tr) and np.array_equal(va_ix, pca_va), (
        f"model_seed={model_seed}: loaded split diverges from PCA's split at {phase1_dir} "
        f"-- this must never happen (require_pca_split should have already caught it)."
    )

    recorded_split_seed = read_split_seed(phase1_dir, default=split_seed)
    if recorded_split_seed != split_seed:
        print(f"  [warn] config split_seed={split_seed} != PCA's recorded split_seed={recorded_split_seed} "
              f"(split FILES still match byte-identically, per the assertion above -- only the provenance label differs).")

    train_cfg = dict(epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                      beta_max=cfg["beta_max"], beta_warmup=cfg["beta_warmup"],
                      patience=cfg["patience"], grad_clip=cfg["grad_clip"],
                      checkpoint_start_epoch=cfg["checkpoint_start_epoch"],
                      free_bits=cfg["free_bits"], diagnostic_interval=cfg["diagnostic_interval"])
    objective_mode = cfg["objective_mode"]
    run_clustering_flag = resolve_run_clustering(cfg)

    print("\n══════ Per-block VAE (fit on train, encode mu for all) ══════")
    N, B = len(subjects), len(block_ids)
    padded = np.zeros((N, B, MAX_D), dtype=np.float32)
    block_rows, train_rows = [], []
    d_used_list, nominal_d_list = [], []
    if cfg["save_per_block"]:
        (out / "embeddings" / "per_block").mkdir(exist_ok=True)

    for j, bid in enumerate(block_ids):
        t_blk = time.time()
        G = blocks[bid]["geno"]
        p = G.shape[1]
        size_class, nominal_d, d_used = latent_dim_for_block(p, len(tr_ix))

        if cfg["loss_type"] == "ORD":
            validate_integer_genotypes(G, bid)
            y_all = torch.tensor(np.rint(G), dtype=torch.long)
        else:
            y_all = None
        x_all = torch.tensor(G, dtype=torch.float32)

        model = BlockVAE(p, d_used, cfg["dropout"], loss_type=cfg["loss_type"], objective_mode=objective_mode)
        tr_y = y_all[tr_ix] if y_all is not None else None
        va_y = y_all[va_ix] if y_all is not None else None
        maf = compute_maf(G, tr_ix) if cfg["loss_type"] == "ORD" else None
        log, best_val_recon, best_epoch = train_block(
            model, x_all[tr_ix], x_all[va_ix], device, train_cfg,
            out / "logs" / f"{bid}.csv", model_seed, tr_y=tr_y, va_y=va_y,
            split_seed=split_seed, maf=maf, maf_bins=cfg["maf_bins"])

        if cfg["loss_type"] == "ORD":
            ev = eval_ord_metrics(model, x_all[va_ix], va_y, maf, cfg["maf_bins"])
            calib = g2_calibration(model, x_all[va_ix], va_y, maf, cfg["maf_bins"])
        else:
            ev = eval_mse_metrics(model, x_all[va_ix])
            calib = {k: json.dumps({}) for k in (
                "observed_g2_frequency_by_maf_bin", "mean_predicted_p_g2_by_maf_bin",
                "g2_brier_score_by_maf_bin", "ordinal_nll_per_snp_by_maf_bin", "n_observations_by_maf_bin")}
        diag = latent_diagnostics(model, x_all[va_ix], cfg["active_var_threshold"],
                                   cfg["collapse_active_dims_threshold"], cfg["free_bits"], nominal_d,
                                   beta_max=cfg["beta_max"], validation_recon=best_val_recon)
        ld_corr = compute_ld_corr(G[va_ix], None)   # deprecated -- always NaN, see compute_ld_corr docstring

        mu = extract_mu(model, G)                    # posterior mean for ALL subjects, computed once
        dist_pres = subject_distance_preservation(G[tr_ix], G[va_ix], mu[tr_ix], mu[va_ix])
        # item 2: class support at the restored best checkpoint, train + validation,
        # from the raw dosage (works for both loss_types, not just ORD's y_all).
        tr_cs = class_support(np.rint(G[tr_ix]).astype(np.int64))
        va_cs = class_support(np.rint(G[va_ix]).astype(np.int64))
        mu_padded = pad_to_max_d(mu, d_used)
        padded[:, j, :] = mu_padded
        d_used_list.append(d_used)
        nominal_d_list.append(nominal_d)
        if cfg["save_per_block"]:
            np.save(out / "embeddings" / "per_block" / f"{bid}.npy", mu)
        torch.save(_c({"state_dict": model.state_dict(), "loss_type": cfg["loss_type"],
                    "objective_mode": objective_mode,
                    "latent_dim": d_used, "n_snps": p, "block_id": bid,
                    "h1": model.h1, "h2": model.h2, "best_epoch": best_epoch,
                    "dropout": cfg["dropout"], "beta_max": cfg["beta_max"],
                    "beta_warmup": cfg["beta_warmup"], "free_bits": cfg["free_bits"],
                    "checkpoint_start_epoch": cfg["checkpoint_start_epoch"],
                    "split_seed": split_seed, "model_seed": model_seed}), out / "models" / f"{bid}.pt")

        block_rows.append(_c({
            "pos": j, "block_id": bid, "gene": blocks[bid]["gene"],
            "group": blocks[bid]["group"], "region_id": blocks[bid]["region_id"],
            "n_snps": p, "size_class": size_class, "nominal_d": nominal_d, "d_used": d_used,
            "h1": model.h1, "h2": model.h2, "loss_type": cfg["loss_type"], "objective_mode": objective_mode,
            "split_seed": split_seed, "model_seed": model_seed,
        }))
        train_rows.append(_c({
            "block_id": bid, "p": p, "size_class": size_class, "nominal_d": nominal_d,
            "d_used": d_used, "h1": model.h1, "h2": model.h2, "loss_type": cfg["loss_type"],
            "objective_mode": objective_mode,
            "split_seed": split_seed, "model_seed": model_seed,
            "epochs_run": len(log), "best_epoch": best_epoch,
            "best_va_recon_metric": round(best_val_recon, 4),
            **ev, **diag, "ld_corr": ld_corr, "subject_distance_preservation": dist_pres,
            **{f"tr_{k}": v for k, v in tr_cs.items()}, **{f"va_{k}": v for k, v in va_cs.items()},
            **calib,
            "seconds": round(time.time() - t_blk, 1),
        }))
        print(f"  {bid:30s} p={p:5d} class={size_class:6s} d={d_used} (h1={model.h1},h2={model.h2})  "
              f"epochs={len(log):3d}  best_epoch={best_epoch}  va_recon={best_val_recon:.3f}  "
              f"active_dims={diag['active_latent_dims']}/{d_used}  "
              f"complete_collapse={diag['complete_collapse']}  partial_collapse={diag['partial_collapse']}  "
              f"any_dim_above_free_bits={diag['any_dim_above_free_bits']}")

    block_order = pd.DataFrame(block_rows)
    block_order.to_csv(out / "block_order.csv", index=False)
    pd.DataFrame(train_rows).to_csv(out / "training_summary.csv", index=False)

    # block_order.csv is the canonical mapping between all_blocks.npy axis 1
    # (and concat_embedding.npy's per-block column groups) and block ID —
    # manifest row order, preserved unchanged through filtering, is never re-sorted.
    assert block_order["block_id"].tolist() == block_ids
    assert block_order["pos"].tolist() == list(range(len(block_ids)))
    # all_blocks_latent_dims.npy (saved below) must align 1:1, in order, with
    # block_order.csv's d_used column -- both are built from the same loop
    # over block_ids, but assert it explicitly so a future refactor that
    # breaks that invariant fails loudly instead of silently mislabeling dims.
    assert d_used_list == block_order["d_used"].tolist()

    # cross-check against the PCA baseline's block order/nominal_d --
    # phase1_dir is now the ONE PCA output_root (no per-seed subdirectory:
    # PCA fits once), so this is a direct, unconditional check.
    pca_block_order = Path(phase1_dir) / "block_order.csv"
    assert_matching_block_order_and_d(pca_block_order, block_ids, nominal_d_list, label="PCA")

    concat = padded.reshape(N, B * MAX_D)
    np.save(out / "embeddings" / "concat_embedding.npy", concat)
    np.save(out / "embeddings" / "all_blocks.npy", padded)
    np.save(out / "embeddings" / "all_blocks_latent_dims.npy",
            np.array(d_used_list, dtype=np.int32))                     # actual d_used per block
    print(f"\n  concat embedding {concat.shape} | padded {padded.shape}")

    if run_clustering_flag:
        print("\n══════ Clustering (GMM+BIC primary, KMeans+silhouette parity) ══════")
        best_k_gmm, best_k_km, gmm_df, km_df = cluster(
            concat, cfg["k_range"], model_seed, out / "clustering")
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

    summary = pd.DataFrame([_c({
        "condition": "per_block_vae_concat", "loss_type": cfg["loss_type"], "objective_mode": objective_mode,
        "beta_max": cfg["beta_max"], "free_bits": cfg["free_bits"],
        "beta_warmup": cfg["beta_warmup"],
        "n_subjects": N, "n_blocks": B, "max_d": MAX_D,
        **clustering_metrics, "clustering_run": run_clustering_flag,
        "split_seed": split_seed, "model_seed": model_seed, "seconds": round(time.time() - t0, 1),
    })])
    summary.to_csv(out / "baseline_summary.csv", index=False)
    with open(out / "baseline_meta.json", "w") as f:
        json.dump(_c({
            "condition": "per_block_vae_concat", "loss_type": cfg["loss_type"], "objective_mode": objective_mode,
            "max_d": MAX_D, "split_seed": split_seed, "model_seed": model_seed,
            "n_subjects": N, "n_blocks": B, "k_range": cfg["k_range"], "clustering_run": run_clustering_flag,
            "diagnostic_interval": cfg["diagnostic_interval"],
            "vae_fit_on": "train_only", "embedding": "posterior mean mu (deterministic)",
            "genotype_input": "raw dosage {0,1,2}, unstandardized",
            "split_semantics": "fixed split (split_seed, reused byte-identically from PCA's phase1_dir "
                                "via require_pca_split -- never regenerated); model_seed seeds VAE init/"
                                "minibatch order/latent sampling only. Median-across-model_seed summaries "
                                "quantify VAE training stability, not split variability.",
            "reconstruction_loss": ("ordinal NLL, sum over SNPs / mean over subjects, no /p"
                                    if cfg["loss_type"] == "ORD" else
                                    "MSE, sum over SNPs / mean over subjects, unconstrained linear decoder"),
            "beta_max": cfg["beta_max"], "beta_warmup": cfg["beta_warmup"], "free_bits": cfg["free_bits"],
            "checkpoint_start_epoch": cfg["checkpoint_start_epoch"],
            "checkpoint_selection": "validation ordinal NLL only" if cfg["loss_type"] == "ORD" else "validation MSE only",
            "latent_rule": "p<=49 -> d=2, 50-199 -> d=4, p>=200 -> d=8 (scripts/baselines/latent_rule.py)",
            "clustering": "GMM+BIC,KMeans+silhouette",
            "note": "all_blocks_latent_dims.npy records the actual per-block d_used (<=nominal_d<=MAX_D). "
                    "ld_corr in training_summary.csv is DEPRECATED (always NaN) -- replaced by "
                    "subject_distance_preservation (train-fitted scaling; see data_utils.py). "
                    "objective_mode=='AE' means z=mu always, no KL/free-bits term in the optimized loss -- "
                    "posterior/KL diagnostics for an AE run are reported but NOT optimized and not directly "
                    "interpretable the way they are for a VAE run.",
        }), f, indent=2)

    print(f"\n{'═'*55}")
    print(summary.to_string(index=False))
    print(f"[done] split_seed={split_seed} model_seed={model_seed}  {time.time()-t0:.1f}s → {out}")
    return out


# ══════════════════════════════════════════════════════════════════
# one-factor-at-a-time sensitivity sweep (config-driven, opt-in via
# vae.sensitivity.enabled). Does NOT touch run_one_model_seed()'s training
# semantics, BlockVAE, the ORD loss, checkpointing, diagnostics, PCA code,
# post-hoc code, clustering, or the split -- it only calls run_one_model_seed()
# once per (condition, model_seed) with objective_mode/beta_max/free_bits/
# output_root overridden on a shallow config copy.
# ══════════════════════════════════════════════════════════════════
def _fmt_param(x):
    """Filesystem-safe decimal token: '.'->'p', no unnecessary trailing
    zeros (0.5->'0p5', 1.0->'1', 0.0->'0', 0.25->'0p25')."""
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    return s.replace(".", "p").replace("-", "neg")


def _validate_numeric_nonneg(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a numeric value, not a boolean or other type (got {value!r})")
    fv = float(value)
    if not np.isfinite(fv):
        raise ValueError(f"{name} must be finite (got {value!r})")
    if fv < 0:
        raise ValueError(f"{name} must be nonnegative (got {value!r})")
    return fv


def _validate_bool(name, value):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a YAML boolean (true/false), not {value!r}")
    return value


def resolve_sensitivity_conditions(cfg):
    """Resolve+validate `vae.sensitivity` (arrives flattened into cfg by
    load_baseline_config, NOT a top-level YAML section) into an ordered,
    deduplicated one-factor-at-a-time condition list. Returns None when
    sensitivity is absent or `enabled: false` (ordinary configs need none of
    these keys -- the caller then takes the unchanged scalar path). Never
    touches the filesystem or genotype data -- config-only.

    Returns {"enabled", "dry_run", "mode", "existing_run_policy", "conditions"}
    where each condition dict carries every sensitivity_design.csv column
    except the cfg-level ones (split_seed, model_seeds, ...) added by the
    caller, which knows the full cfg.
    """
    sens = cfg.get("sensitivity")
    if not sens:
        return None
    if not isinstance(sens, dict):
        raise TypeError("vae.sensitivity must be a mapping")

    enabled = _validate_bool("sensitivity.enabled", sens.get("enabled", False))
    if not enabled:
        return None

    dry_run = _validate_bool("sensitivity.dry_run", sens.get("dry_run", False))
    include_ae = _validate_bool("sensitivity.include_ae", sens.get("include_ae", False))

    mode = sens.get("mode", "one_at_a_time")
    if mode != "one_at_a_time":
        raise ValueError(f"sensitivity.mode={mode!r} is not supported (only 'one_at_a_time' is implemented)")

    existing_run_policy = sens.get("existing_run_policy", "error")
    if existing_run_policy not in ("error", "skip_complete"):
        raise ValueError(
            f"sensitivity.existing_run_policy must be 'error' or 'skip_complete' (got {existing_run_policy!r})")

    reference = sens.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("sensitivity.reference must be a mapping with beta_max/free_bits/objective_mode")
    ref_beta = _validate_numeric_nonneg("sensitivity.reference.beta_max", reference.get("beta_max"))
    ref_fb = _validate_numeric_nonneg("sensitivity.reference.free_bits", reference.get("free_bits"))
    ref_mode = str(reference.get("objective_mode", "")).strip().upper()
    if ref_mode != "VAE":
        raise ValueError(f"sensitivity.reference.objective_mode must be VAE (got {reference.get('objective_mode')!r})")

    beta_values = sens.get("beta_values")
    if not isinstance(beta_values, list) or not beta_values:
        raise ValueError("sensitivity.beta_values must be a nonempty list")
    beta_values = [_validate_numeric_nonneg("sensitivity.beta_values[]", v) for v in beta_values]

    free_bits_values = sens.get("free_bits_values")
    if not isinstance(free_bits_values, list) or not free_bits_values:
        raise ValueError("sensitivity.free_bits_values must be a nonempty list")
    free_bits_values = [_validate_numeric_nonneg("sensitivity.free_bits_values[]", v) for v in free_bits_values]

    conditions = []
    seen_keys = set()
    seen_ids = set()

    def _add(condition_id, objective_mode, beta_max, free_bits, free_bits_applicable, is_reference, varied_parameter):
        key = (objective_mode, round(beta_max, 6), round(free_bits, 6))
        if key in seen_keys:
            return
        seen_keys.add(key)
        if condition_id in seen_ids:
            raise ValueError(f"sensitivity: duplicate condition_id {condition_id!r} -- condition IDs must be unique")
        seen_ids.add(condition_id)
        conditions.append({
            "condition_order": len(conditions) + 1,
            "condition_id": condition_id,
            "objective_mode": objective_mode,
            "beta_max": beta_max,
            "free_bits": free_bits,
            "free_bits_applicable": free_bits_applicable,
            "is_reference": is_reference,
            "varied_parameter": varied_parameter,
            "reference_beta": ref_beta,
            "reference_free_bits": ref_fb,
        })

    # 1. reference
    _add(f"vae_beta{_fmt_param(ref_beta)}_lambda{_fmt_param(ref_fb)}", "VAE", ref_beta, ref_fb,
         True, True, "reference")
    # 2. each beta value, reference lambda, VAE
    for b in beta_values:
        _add(f"vae_beta{_fmt_param(b)}_lambda{_fmt_param(ref_fb)}", "VAE", b, ref_fb, True, False, "beta")
    # 3. each lambda value, reference beta, VAE
    for fbv in free_bits_values:
        _add(f"vae_beta{_fmt_param(ref_beta)}_lambda{_fmt_param(fbv)}", "VAE", ref_beta, fbv, True, False, "free_bits")
    # 4. one true AE condition
    if include_ae:
        # beta_max=0.0 for clear provenance; numeric free_bits retained
        # internally (existing diagnostic functions require it) but marked
        # free_bits_applicable=false. AE loss behavior itself (z=mu,
        # reconstruction-only) is unchanged -- see BlockVAE.objective_mode.
        _add("ae", "AE", 0.0, ref_fb, False, False, "objective_mode")

    return {
        "enabled": True, "dry_run": dry_run, "mode": mode,
        "existing_run_policy": existing_run_policy, "conditions": conditions,
    }


def _seed_run_is_complete(seed_dir):
    """The exact completeness criterion for existing_run_policy=skip_complete."""
    required = [
        seed_dir / "baseline_summary.csv", seed_dir / "training_summary.csv",
        seed_dir / "block_order.csv", seed_dir / "embeddings" / "all_blocks.npy",
        seed_dir / "embeddings" / "all_blocks_latent_dims.npy",
    ]
    return all(p.exists() for p in required)


def _seed_dir_has_any_output(seed_dir):
    return seed_dir.exists() and any(seed_dir.rglob("*"))


def run_sensitivity_sweep(cfg, sens):
    """Real (or dry-run) execution of the resolved sensitivity design.
    Writes sensitivity_design.csv before any data loading; dry_run returns
    immediately after that -- never calls load_block_defs/load_all_blocks/
    require_pca_split/run_one_model_seed and creates no condition/seed/
    model/log/embedding directories."""
    conditions = sens["conditions"]
    dry_run = sens["dry_run"]
    existing_run_policy = sens["existing_run_policy"]

    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)   # top-level root only

    model_seeds = resolve_model_seeds(cfg)     # config-only, no filesystem/data I/O
    design_rows = [dict(c, split_seed=int(cfg["split_seed"]), model_seeds=model_seeds,
                         n_model_seeds=len(model_seeds), epochs=cfg["epochs"],
                         beta_warmup=cfg["beta_warmup"], diagnostic_interval=cfg["diagnostic_interval"],
                         block_def=cfg["block_def"], phase1_dir=cfg.get("phase1_dir"),
                         output_dir=str(root / c["condition_id"]))
                   for c in conditions]
    design_df = pd.DataFrame(design_rows)
    design_path = root / "sensitivity_design.csv"
    design_df.to_csv(design_path, index=False)

    print("\n══════ Sensitivity sweep: one_at_a_time design ══════")
    print(design_df.to_string(index=False))
    n_seeds = len(model_seeds)
    seed_word = "model seed" if n_seeds == 1 else "model seeds"
    print(f"\n{len(conditions)} conditions × {n_seeds} {seed_word} = {len(conditions) * n_seeds} runs")
    print(f"[written] {design_path}")

    if dry_run:
        print("\n[dry_run] sensitivity.dry_run=true -- design written, exiting before any data loading or training.")
        return

    split_seed = int(cfg["split_seed"])
    pca_root = Path(cfg["phase1_dir"]) if cfg.get("phase1_dir") else None

    by_seed_rows = []
    for cond in conditions:
        cond_cfg = dict(cfg)   # shallow copy -- override ONLY the sweep parameters
        cond_cfg["objective_mode"] = cond["objective_mode"]
        cond_cfg["beta_max"] = cond["beta_max"]
        cond_cfg["free_bits"] = cond["free_bits"]
        cond_cfg["output_root"] = str(root / cond["condition_id"])
        cond_cfg["condition_id"] = cond["condition_id"]
        cond_root = Path(cond_cfg["output_root"])

        seed_dirs = []
        for model_seed in model_seeds:
            seed_dir = model_seed_output_dir(cond_root, model_seed)
            complete = _seed_run_is_complete(seed_dir)
            has_any = _seed_dir_has_any_output(seed_dir)
            if has_any:
                if existing_run_policy == "skip_complete" and complete:
                    run_status = "skipped_complete"
                    print(f"[skip] {cond['condition_id']}/seed_{model_seed}: complete -> skip_complete")
                else:
                    reason = "already complete" if complete else "PARTIAL/incomplete outputs present"
                    raise RuntimeError(
                        f"sensitivity sweep: {seed_dir} {reason} (existing_run_policy={existing_run_policy!r}). "
                        f"Refusing to silently overwrite models/logs/embeddings/summaries -- remove it manually, "
                        f"point at a fresh output_root, or set existing_run_policy: skip_complete to reuse a "
                        f"fully complete run (never resumes a partial one)."
                    )
            else:
                run_one_model_seed(cond_cfg, model_seed, split_seed, seed_dir, pca_root)
                run_status = "completed"
            seed_dirs.append((model_seed, seed_dir, run_status))

        # existing per-condition aggregation mechanism (median_across_seeds),
        # unchanged, scoped to this condition's own directory.
        summary_paths = [d / "baseline_summary.csv" for (_, d, _) in seed_dirs]
        median_across_seeds(summary_paths, key_cols=["condition", "loss_type"],
                             out_path=cond_root / "median_summary.csv", seeds=model_seeds)
        training_paths = [d / "training_summary.csv" for (_, d, _) in seed_dirs]
        median_across_seeds(training_paths, key_cols=["block_id"],
                             out_path=cond_root / "median_training_summary.csv", seeds=model_seeds)
        print(f"[condition done] {cond['condition_id']}: {len(seed_dirs)} seed(s) → {cond_root}")

        for model_seed, seed_dir, run_status in seed_dirs:
            row = pd.read_csv(seed_dir / "baseline_summary.csv").iloc[0].to_dict()
            row.update({
                "condition_order": cond["condition_order"], "condition_id": cond["condition_id"],
                "objective_mode": cond["objective_mode"], "beta_max": cond["beta_max"],
                "free_bits": cond["free_bits"], "free_bits_applicable": cond["free_bits_applicable"],
                "is_reference": cond["is_reference"], "varied_parameter": cond["varied_parameter"],
                "model_seed": model_seed, "split_seed": split_seed, "epochs": cfg["epochs"],
                "run_status": run_status,
            })
            by_seed_rows.append(row)

    by_seed_df = pd.DataFrame(by_seed_rows).sort_values(["condition_order", "model_seed"]).reset_index(drop=True)
    by_seed_path = root / "sensitivity_summary_by_seed.csv"
    by_seed_df.to_csv(by_seed_path, index=False)
    print(f"[written] {by_seed_path}")

    training_frames = []
    for cond in conditions:
        cond_root = root / cond["condition_id"]
        for model_seed in model_seeds:
            seed_dir = model_seed_output_dir(cond_root, model_seed)
            ts = pd.read_csv(seed_dir / "training_summary.csv")
            # training_summary.csv already carries its own condition_id column
            # (threaded via run_one_model_seed's _c()) -- overwrite rather
            # than insert-duplicate, then bring the four sweep-context
            # columns to the front for readability.
            ts["condition_order"] = cond["condition_order"]
            ts["condition_id"] = cond["condition_id"]
            ts["is_reference"] = cond["is_reference"]
            ts["varied_parameter"] = cond["varied_parameter"]
            front = ["condition_order", "condition_id", "is_reference", "varied_parameter"]
            ts = ts[front + [c for c in ts.columns if c not in front]]
            training_frames.append(ts)
    training_by_block_seed = pd.concat(training_frames, ignore_index=True)
    training_path = root / "sensitivity_training_by_block_seed.csv"
    training_by_block_seed.to_csv(training_path, index=False)
    print(f"[written] {training_path}")

    cond_summary_rows = []
    for cond in conditions:
        sub = by_seed_df[by_seed_df["condition_id"] == cond["condition_id"]]
        cond_summary_rows.append({
            "condition_order": cond["condition_order"], "condition_id": cond["condition_id"],
            "objective_mode": cond["objective_mode"], "beta_max": cond["beta_max"],
            "free_bits": cond["free_bits"], "free_bits_applicable": cond["free_bits_applicable"],
            "is_reference": cond["is_reference"], "varied_parameter": cond["varied_parameter"],
            "n_model_seeds": len(sub),
            "n_completed": int((sub["run_status"] == "completed").sum()),
            "n_skipped_complete": int((sub["run_status"] == "skipped_complete").sum()),
            "median_seconds": float(sub["seconds"].median()),
            "min_seconds": float(sub["seconds"].min()),
            "max_seconds": float(sub["seconds"].max()),
        })
    cond_summary_df = pd.DataFrame(cond_summary_rows).sort_values("condition_order").reset_index(drop=True)
    cond_summary_path = root / "sensitivity_summary.csv"
    cond_summary_df.to_csv(cond_summary_path, index=False)
    print(f"[written] {cond_summary_path}")
    print(f"\n[sensitivity sweep done] {len(conditions)} conditions × {n_seeds} {seed_word} → {root}")


# ──────────────────────────────────────────────────────────────────
# main -- one fixed split (from PCA), one run per model_seed
# ──────────────────────────────────────────────────────────────────
def _validate_loss_type(loss_type):
    s = str(loss_type).strip().upper()
    if s not in ("MSE", "ORD"):
        raise ValueError(f"config vae.loss_type must be MSE or ORD (got {loss_type!r})")
    return s


def _validate_objective_mode(objective_mode):
    s = str(objective_mode).strip().upper()
    if s not in ("VAE", "AE"):
        raise ValueError(f"config vae.objective_mode must be VAE or AE (got {objective_mode!r})")
    return s


def main():
    ap = argparse.ArgumentParser("Per-block VAE baseline (concat); config-driven")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help=f"config YAML (default: {DEFAULT_CONFIG})")
    args = ap.parse_args()
    cfg = load_baseline_config(args.config, "vae")
    cfg["loss_type"] = _validate_loss_type(cfg["loss_type"])
    cfg.setdefault("checkpoint_start_epoch", 51)
    cfg.setdefault("free_bits", 0.5)
    cfg.setdefault("active_var_threshold", 0.01)
    cfg.setdefault("collapse_active_dims_threshold", 0)
    cfg.setdefault("maf_bins", [0.0, 0.05, 0.10, 0.20, 0.50])
    cfg.setdefault("diagnostic_interval", 10)          # item 1: train/val eval cadence, epochs
    cfg.setdefault("objective_mode", "VAE")             # item 9: VAE (default) | AE -- never auto-runs AE
    cfg["objective_mode"] = _validate_objective_mode(cfg["objective_mode"])
    cfg.setdefault("run_clustering", True)               # item 8: default on, shared with PCA
    if not isinstance(cfg["diagnostic_interval"], int) or isinstance(cfg["diagnostic_interval"], bool):
        raise TypeError("vae.diagnostic_interval must be an integer number of epochs (0 disables periodic diagnostics)")
    if cfg["diagnostic_interval"] < 0:
        raise ValueError("vae.diagnostic_interval must be >= 0")
    resolve_run_clustering(cfg)  # validate before an expensive run starts

    # sensitivity sweep (opt-in, config-driven): validated before ANY data
    # loading. Absent/enabled:false -> None -> unchanged scalar path below;
    # ordinary baseline/AE configs need none of these keys.
    sens = resolve_sensitivity_conditions(cfg)
    if sens is not None:
        run_sensitivity_sweep(cfg, sens)
        return

    split_seed = int(cfg["split_seed"])
    model_seeds = resolve_model_seeds(cfg)
    root = Path(cfg["output_root"])
    pca_root = Path(cfg["phase1_dir"]) if cfg.get("phase1_dir") else None

    seed_dirs = []
    for model_seed in model_seeds:
        out = model_seed_output_dir(root, model_seed)
        seed_dirs.append(run_one_model_seed(cfg, model_seed, split_seed, out, pca_root))

    summaries = [d / "baseline_summary.csv" for d in seed_dirs]
    med = median_across_seeds(summaries, key_cols=["condition", "loss_type"],
                               out_path=root / "median_summary.csv", seeds=model_seeds)
    print(f"\n[median] {len(model_seeds)} model_seed(s) median summary (split held fixed at "
          f"split_seed={split_seed}) → {root / 'median_summary.csv'}")
    print(med.to_string(index=False))

    trainings = [d / "training_summary.csv" for d in seed_dirs]
    med_t = median_across_seeds(trainings, key_cols=["block_id"],
                                 out_path=root / "median_training_summary.csv", seeds=model_seeds)
    print(f"[median] {len(model_seeds)} model_seed(s) median per-block training summary → "
          f"{root / 'median_training_summary.csv'}")


if __name__ == "__main__":
    main()
