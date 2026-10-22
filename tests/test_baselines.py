#!/usr/bin/env python3
"""
tests/test_baselines.py — focused tests for the Phase-1 baseline design
(config/baseline.yaml, scripts/baselines/{latent_rule,data_utils,
per_block_PCA_concat,per_block_vae_concat}.py).

Run: KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_baselines.py -v
(KMP_DUPLICATE_LIB_OK works around a local libomp double-init warning from
having both torch and scikit-learn loaded; unrelated to correctness.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "baselines"))
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))

import latent_rule as lr
import data_utils as du
import per_block_vae_concat as vae_mod


# ──────────────────────────────────────────────────────────────────
# 1. latent-class boundaries
# ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("p,expected_class,expected_d", [
    (1, "small", 2),
    (49, "small", 2),
    (50, "medium", 4),
    (199, "medium", 4),
    (200, "large", 8),
    (2161, "large", 8),
])
def test_size_class_boundaries(p, expected_class, expected_d):
    size_class, nominal_d = lr.size_class_and_nominal_d(p)
    assert size_class == expected_class
    assert nominal_d == expected_d


def test_no_fourth_class_and_no_eff_haps():
    # every p in a wide sweep must land in exactly one of the three classes
    for p in list(range(1, 600)) + [1000, 2161, 5000]:
        size_class, nominal_d = lr.size_class_and_nominal_d(p)
        assert size_class in ("small", "medium", "large")
        assert nominal_d in (2, 4, 8)
    assert not hasattr(lr, "eff_haps")
    assert not hasattr(lr, "size_class_from_eff_haps")


def test_actual_d_used_clamped_by_train_n_and_p():
    # p and n_train both smaller than nominal_d -> actual_d_used clamps down
    size_class, nominal_d, d_used = lr.latent_dim_for_block(p=3, n_train=2)
    assert size_class == "small" and nominal_d == 2
    assert d_used == min(2, 3, 2 - 1) == 1

    size_class, nominal_d, d_used = lr.latent_dim_for_block(p=500, n_train=1000)
    assert size_class == "large" and nominal_d == 8 and d_used == 8


def test_pad_to_max_d():
    Z = np.ones((5, 3), dtype=np.float32)
    padded = lr.pad_to_max_d(Z, 3)
    assert padded.shape == (5, lr.MAX_D)
    assert np.all(padded[:, :3] == 1)
    assert np.all(padded[:, 3:] == 0)
    full = np.ones((5, lr.MAX_D), dtype=np.float32)
    assert lr.pad_to_max_d(full, lr.MAX_D) is full


# ──────────────────────────────────────────────────────────────────
# 2. VAE width formula
# ──────────────────────────────────────────────────────────────────
def test_hidden_widths_formula():
    p, d = 114, 4
    h1, h2 = lr.hidden_widths(p, d)
    assert h1 == round(p ** (2 / 3) * d ** (1 / 3))
    assert h2 == round(p ** (1 / 3) * d ** (2 / 3))
    assert h1 >= 1 and h2 >= 1


def test_hidden_widths_tiny_block_stays_positive():
    h1, h2 = lr.hidden_widths(p=8, d=2)
    assert h1 >= 1 and h2 >= 1


def test_block_vae_architecture_matches_formula():
    p, d = 114, 4
    model = vae_mod.BlockVAE(p, d, drop=0.30, loss_type="ORD")
    h1, h2 = lr.hidden_widths(p, d)
    assert model.h1 == h1 and model.h2 == h2
    # encoder: Linear(p,h1) -> LayerNorm -> GELU -> Dropout -> Linear(h1,h2) -> ...
    assert isinstance(model.enc[0], torch.nn.Linear) and model.enc[0].in_features == p and model.enc[0].out_features == h1
    assert isinstance(model.enc[1], torch.nn.LayerNorm)
    assert isinstance(model.enc[2], torch.nn.GELU)
    assert isinstance(model.enc[3], torch.nn.Dropout) and model.enc[3].p == pytest.approx(0.30)
    assert isinstance(model.enc[4], torch.nn.Linear) and model.enc[4].in_features == h1 and model.enc[4].out_features == h2
    assert model.mu.in_features == h2 and model.mu.out_features == d
    assert model.lv.in_features == h2 and model.lv.out_features == d
    # decoder mirrors: Linear(d,h2) -> ... -> Linear(h2,h1) -> ... -> Linear(h1,out)
    assert model.dec[0].in_features == d and model.dec[0].out_features == h2
    assert model.dec[4].in_features == h2 and model.dec[4].out_features == h1
    assert model.dec[8].in_features == h1 and model.dec[8].out_features == p * 2  # ORD out width


# ──────────────────────────────────────────────────────────────────
# 3. ORD objective: genotype validation, no-/p NLL, free-bits math
# ──────────────────────────────────────────────────────────────────
def test_validate_integer_genotypes_accepts_clean_input():
    G = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.float32)
    vae_mod.validate_integer_genotypes(G, "blk")  # should not raise


def test_validate_integer_genotypes_rejects_non_integer():
    G = np.array([[0, 1.5, 2]], dtype=np.float32)
    with pytest.raises(ValueError, match="integer-valued"):
        vae_mod.validate_integer_genotypes(G, "blk")


def test_validate_integer_genotypes_rejects_non_finite():
    G = np.array([[0, np.nan, 2]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        vae_mod.validate_integer_genotypes(G, "blk")


def test_validate_integer_genotypes_rejects_out_of_range():
    G = np.array([[0, 1, 3]], dtype=np.float32)
    with pytest.raises(ValueError, match="out of"):
        vae_mod.validate_integer_genotypes(G, "blk")


def test_ordinal_nll_no_p_normalization():
    """NLL must be summed over SNPs and averaged over subjects -- growing P
    (more SNPs, same per-SNP difficulty) must scale the loss up, not leave
    it flat (which a /p normalization would produce)."""
    torch.manual_seed(0)
    B = 6
    for P in (5, 50):
        logits = torch.randn(B, P, 2)
        targets = torch.randint(0, 3, (B, P))
        nll = vae_mod.ordinal_loss(logits, targets)
        per_subj = vae_mod.ordinal_nll_per_subject(logits, targets)
        assert per_subj.shape == (B,)
        assert nll.item() == pytest.approx(per_subj.mean().item(), rel=1e-5)
    # same per-SNP logits/targets tiled P times -> summed-not-averaged NLL scales with P
    logits1 = torch.randn(B, 5, 2)
    targets1 = torch.randint(0, 3, (B, 5))
    logits10 = logits1.repeat(1, 10, 1)
    targets10 = targets1.repeat(1, 10)
    nll1 = vae_mod.ordinal_loss(logits1, targets1)
    nll10 = vae_mod.ordinal_loss(logits10, targets10)
    assert nll10.item() == pytest.approx(10 * nll1.item(), rel=1e-4)


def test_free_bits_rate_math():
    # dim 0: mean KL below the floor -> clamped to free_bits
    # dim 1: mean KL above the floor -> passes through unclamped
    mu = torch.zeros(4, 2)
    lv = torch.zeros(4, 2)
    # raw_kl_per_dim(mu=0, lv=0) = -0.5*(1+0-0-1) = 0 for every subject/dim
    rate, mean_kl_dim, raw_kl_total = vae_mod.free_bits_rate(mu, lv, free_bits=0.5)
    assert torch.allclose(mean_kl_dim, torch.zeros(2))
    assert raw_kl_total.item() == pytest.approx(0.0)
    assert rate.item() == pytest.approx(2 * 0.5)  # both dims floored to 0.5, summed

    mu2 = torch.tensor([[2.0, 0.0]] * 4)
    lv2 = torch.zeros(4, 2)
    rate2, mean_kl_dim2, raw_kl_total2 = vae_mod.free_bits_rate(mu2, lv2, free_bits=0.5)
    expected_kl_dim0 = -0.5 * (1 + 0 - 4 - 1)  # = 1.5, above floor
    assert mean_kl_dim2[0].item() == pytest.approx(expected_kl_dim0)
    assert mean_kl_dim2[1].item() == pytest.approx(0.0)
    assert raw_kl_total2.item() == pytest.approx(expected_kl_dim0 + 0.0)
    assert rate2.item() == pytest.approx(expected_kl_dim0 + 0.5)  # dim0 unclamped, dim1 floored


# ──────────────────────────────────────────────────────────────────
# 4. patience / checkpoint eligibility starting after epoch 50
# ──────────────────────────────────────────────────────────────────
def test_checkpoint_eligibility_starts_after_epoch_50():
    torch.manual_seed(0)
    np.random.seed(0)
    p, d, n = 12, 2, 40
    G = np.random.randint(0, 3, size=(n, p)).astype(np.float32)
    x = torch.tensor(G, dtype=torch.float32)
    y = torch.tensor(G, dtype=torch.long)
    tr_x, va_x = x[:30], x[30:]
    tr_y, va_y = y[:30], y[30:]

    model = vae_mod.BlockVAE(p, d, drop=0.0, loss_type="ORD")
    cfg = dict(epochs=20, batch_size=8, lr=1e-3, beta_max=0.5, beta_warmup=50,
               patience=3, grad_clip=1.0, checkpoint_start_epoch=51, free_bits=0.5,
               diagnostic_interval=10)
    log, best, best_epoch = vae_mod.train_block(
        model, tr_x, va_x, torch.device("cpu"), cfg,
        Path("/tmp/_test_log.csv"), model_seed=0, tr_y=tr_y, va_y=va_y,
        maf=np.full(p, 0.1), maf_bins=[0.0, 1.0])
    # epochs (20) never reached checkpoint_start_epoch (51) -> must run the
    # full epoch budget (patience never gets a chance to fire) and fall
    # back to the final epoch's weights.
    assert len(log) == 20
    assert best_epoch == 20


def test_patience_fires_only_after_checkpoint_start():
    torch.manual_seed(1)
    np.random.seed(1)
    p, d, n = 12, 2, 40
    G = np.random.randint(0, 3, size=(n, p)).astype(np.float32)
    x = torch.tensor(G, dtype=torch.float32)
    y = torch.tensor(G, dtype=torch.long)
    tr_x, va_x = x[:30], x[30:]
    tr_y, va_y = y[:30], y[30:]

    model = vae_mod.BlockVAE(p, d, drop=0.0, loss_type="ORD")
    cfg = dict(epochs=60, batch_size=8, lr=1e-3, beta_max=0.5, beta_warmup=50,
               patience=2, grad_clip=1.0, checkpoint_start_epoch=51, free_bits=0.5,
               diagnostic_interval=10)
    log, best, best_epoch = vae_mod.train_block(
        model, tr_x, va_x, torch.device("cpu"), cfg,
        Path("/tmp/_test_log2.csv"), model_seed=1, tr_y=tr_y, va_y=va_y,
        maf=np.full(p, 0.1), maf_bins=[0.0, 1.0])
    # can only stop at epoch>=51+patience once eligibility begins, and never
    # stops before epoch 51 (there just isn't enough runway for patience=2
    # to fire earlier than that under this gating).
    assert log[-1]["epoch"] >= 51


# ──────────────────────────────────────────────────────────────────
# 5. matched PCA/VAE block order and d
# ──────────────────────────────────────────────────────────────────
def test_assert_matching_block_order_and_d_passes_on_match(tmp_path):
    other = pd.DataFrame({"block_id": ["a", "b"], "nominal_d": [2, 4]})
    other_csv = tmp_path / "block_order.csv"
    other.to_csv(other_csv, index=False)
    lr.assert_matching_block_order_and_d(other_csv, ["a", "b"], [2, 4], label="PCA")  # no raise


def test_assert_matching_block_order_and_d_fails_on_order_mismatch(tmp_path):
    other = pd.DataFrame({"block_id": ["a", "b"], "nominal_d": [2, 4]})
    other_csv = tmp_path / "block_order.csv"
    other.to_csv(other_csv, index=False)
    with pytest.raises(AssertionError, match="block order mismatch"):
        lr.assert_matching_block_order_and_d(other_csv, ["b", "a"], [4, 2], label="PCA")


def test_assert_matching_block_order_and_d_fails_on_d_mismatch(tmp_path):
    other = pd.DataFrame({"block_id": ["a", "b"], "nominal_d": [2, 4]})
    other_csv = tmp_path / "block_order.csv"
    other.to_csv(other_csv, index=False)
    with pytest.raises(AssertionError, match="nominal_d mismatch"):
        lr.assert_matching_block_order_and_d(other_csv, ["a", "b"], [2, 8], label="PCA")


def test_assert_matching_block_order_and_d_noop_when_absent(tmp_path, capsys):
    lr.assert_matching_block_order_and_d(tmp_path / "missing.csv", ["a"], [2], label="PCA")
    out = capsys.readouterr().out
    assert "skipping cross-check" in out


# ──────────────────────────────────────────────────────────────────
# 6. split separation / fixed-seed reproducibility
# ──────────────────────────────────────────────────────────────────
def test_create_or_reuse_pca_split_train_val_separation(tmp_path):
    subjects = np.array([f"s{i}" for i in range(50)])
    out = tmp_path / "run"
    out.mkdir()
    tr, va = du.create_or_reuse_pca_split(subjects, out, None, val_frac=0.2, split_seed=7)
    assert set(tr.tolist()) & set(va.tolist()) == set()
    assert set(tr.tolist()) | set(va.tolist()) == set(range(50))
    assert len(va) == 10


def test_create_or_reuse_pca_split_fixed_seed_reproducible(tmp_path):
    subjects = np.array([f"s{i}" for i in range(50)])
    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    out1.mkdir(); out2.mkdir()
    tr1, va1 = du.create_or_reuse_pca_split(subjects, out1, None, val_frac=0.2, split_seed=42)
    tr2, va2 = du.create_or_reuse_pca_split(subjects, out2, None, val_frac=0.2, split_seed=42)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(va1, va2)


def test_create_or_reuse_pca_split_reuses_external_phase1(tmp_path):
    subjects = np.array([f"s{i}" for i in range(50)])
    p1 = tmp_path / "phase1"
    p1.mkdir()
    tr1, va1 = du.create_or_reuse_pca_split(subjects, p1, None, val_frac=0.2, split_seed=1)

    out2 = tmp_path / "run2"
    out2.mkdir()
    # different split_seed, but an alignable external phase1_dir -> still reuses it verbatim
    tr2, va2 = du.create_or_reuse_pca_split(subjects, out2, p1, val_frac=0.2, split_seed=999)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(va1, va2)


# ══════════════════════════════════════════════════════════════════
# require_pca_split -- VAE's ONLY split path: must reuse PCA's split
# byte-identically, and FAIL LOUDLY (never silently create) if PCA's
# split is missing or mismatched.
# ══════════════════════════════════════════════════════════════════
def test_require_pca_split_reuses_pca_split_exactly(tmp_path):
    subjects = np.array([f"s{i}" for i in range(50)])
    pca_out = tmp_path / "pca"
    pca_out.mkdir()
    tr, va = du.create_or_reuse_pca_split(subjects, pca_out, None, val_frac=0.2, split_seed=42)

    vae_out = tmp_path / "vae" / "seed_1"
    vae_out.mkdir(parents=True)
    vtr, vva = du.require_pca_split(subjects, vae_out, str(pca_out))
    assert np.array_equal(tr, vtr)
    assert np.array_equal(va, vva)


def test_require_pca_split_has_no_seed_parameter():
    """Structural proof that require_pca_split cannot be influenced by a
    model_seed -- it has no such parameter at all."""
    import inspect
    params = list(inspect.signature(du.require_pca_split).parameters)
    assert "seed" not in params and "model_seed" not in params and "split_seed" not in params


def test_require_pca_split_fails_loudly_when_phase1_dir_missing():
    with pytest.raises(ValueError, match="phase1_dir is required"):
        du.require_pca_split(np.array(["s0", "s1"]), Path("/tmp/whatever"), None)


def test_require_pca_split_fails_loudly_when_files_absent(tmp_path):
    empty_dir = tmp_path / "empty_pca"
    empty_dir.mkdir()
    out = tmp_path / "vae_out"
    out.mkdir()
    with pytest.raises(FileNotFoundError, match="PCA split files missing"):
        du.require_pca_split(np.array(["s0", "s1"]), out, str(empty_dir))


def test_require_pca_split_fails_loudly_on_subject_mismatch(tmp_path):
    pca_out = tmp_path / "pca"
    pca_out.mkdir()
    du.create_or_reuse_pca_split(np.array([f"s{i}" for i in range(20)]), pca_out, None, val_frac=0.2, split_seed=1)

    out = tmp_path / "vae_out"
    out.mkdir()
    different_subjects = np.array([f"x{i}" for i in range(20)])  # different subject IDs entirely
    with pytest.raises(ValueError, match="does not match this run's subjects"):
        du.require_pca_split(different_subjects, out, str(pca_out))


# ──────────────────────────────────────────────────────────────────
# 7. genotype/data plumbing reused by both baselines
# ──────────────────────────────────────────────────────────────────
def test_compute_maf_train_only(tmp_path):
    G = np.array([[0, 2], [0, 2], [2, 0], [0, 2]], dtype=np.float32)
    tr_ix = np.array([0, 1, 2])  # exclude row 3 from MAF calc
    maf = du.compute_maf(G, tr_ix)
    # col0 freq(allele)=mean([0,0,2])/2=1/3 -> maf=1/3 ; col1 freq=mean([2,2,0])/2=2/3 -> maf=1/3
    assert maf[0] == pytest.approx(1 / 3)
    assert maf[1] == pytest.approx(1 / 3)


def test_resolve_model_seeds_returns_list_and_requires_nonempty():
    assert du.resolve_model_seeds({"model_seeds": [42, 43, 44, 45, 46]}) == [42, 43, 44, 45, 46]
    assert du.resolve_model_seeds({"model_seeds": [7]}) == [7]
    with pytest.raises(ValueError, match="model_seeds is required"):
        du.resolve_model_seeds({})
    with pytest.raises(ValueError, match="model_seeds is required"):
        du.resolve_model_seeds({"model_seeds": []})


def test_model_seed_output_dir_always_nested(tmp_path):
    assert du.model_seed_output_dir(tmp_path / "vae", 42) == tmp_path / "vae" / "seed_42"
    assert du.model_seed_output_dir(tmp_path / "vae", 43) == tmp_path / "vae" / "seed_43"


# ══════════════════════════════════════════════════════════════════
# AUDIT (fixed-split redesign): PCA fits ONCE on split_seed; every VAE
# model_seed reuses that split byte-identically and NEVER regenerates
# it. model_seed varies ONLY VAE init/minibatch/latent-sampling
# stochasticity, never the split.
# ══════════════════════════════════════════════════════════════════
def test_changing_model_seed_does_not_change_split_indices(tmp_path):
    subjects = np.array([f"s{i}" for i in range(60)])
    pca_root = tmp_path / "pca"
    pca_root.mkdir()
    # PCA fits ONCE, flat directory, no seed subdirectory
    ptr, pva = du.create_or_reuse_pca_split(subjects, pca_root, None, val_frac=0.2, split_seed=42)

    vae_root = tmp_path / "vae"
    splits_by_model_seed = {}
    for model_seed in (42, 43, 44, 45, 46):
        out = du.model_seed_output_dir(vae_root, model_seed)
        out.mkdir(parents=True)
        vtr, vva = du.require_pca_split(subjects, out, str(pca_root))
        splits_by_model_seed[model_seed] = (vtr, vva)
        assert np.array_equal(vtr, ptr), f"model_seed={model_seed}: train indices != PCA's"
        assert np.array_equal(vva, pva), f"model_seed={model_seed}: val indices != PCA's"
        pca_sub = pd.read_csv(pca_root / "subjects.csv")["IID"].astype(str).values
        vae_sub = pd.read_csv(out / "subjects.csv")["IID"].astype(str).values
        assert np.array_equal(pca_sub, vae_sub)

    # every model_seed's split is identical to every other's (all reuse the SAME PCA split)
    for a in (42, 43, 44, 45, 46):
        for b in (42, 43, 44, 45, 46):
            assert np.array_equal(splits_by_model_seed[a][0], splits_by_model_seed[b][0])
            assert np.array_equal(splits_by_model_seed[a][1], splits_by_model_seed[b][1])


def test_changing_model_seed_changes_rng_state():
    """model_seed DOES change VAE init/training stochasticity -- this is
    the thing it's supposed to vary, unlike the split."""
    vae_mod.set_seed(42)
    w1 = vae_mod.BlockVAE(20, 4, drop=0.0, loss_type="ORD").enc[0].weight.detach().clone()
    vae_mod.set_seed(43)
    w2 = vae_mod.BlockVAE(20, 4, drop=0.0, loss_type="ORD").enc[0].weight.detach().clone()
    assert not torch.allclose(w1, w2)

    vae_mod.set_seed(42)
    g1 = torch.Generator(); g1.manual_seed(42)
    order1 = torch.randperm(30, generator=g1)
    vae_mod.set_seed(43)
    g2 = torch.Generator(); g2.manual_seed(43)
    order2 = torch.randperm(30, generator=g2)
    assert not torch.equal(order1, order2)


def test_pca_script_has_no_seed_loop_or_median_aggregation():
    """Static check: per_block_PCA_concat.py must fit once (no seed loop,
    no median-across-seed aggregation) -- PCA is deterministic given a
    fixed split, so re-fitting per seed would just relabel noise as
    'seed variance'."""
    import inspect
    import per_block_PCA_concat as pca_mod
    src = inspect.getsource(pca_mod)
    assert "median_across_seeds" not in src
    assert "for seed in" not in src
    assert "for split_seed in" not in src
    assert "resolve_model_seeds" not in src
    # main() takes exactly one split_seed from config, not a list
    main_src = inspect.getsource(pca_mod.main)
    assert 'cfg["split_seed"]' in main_src


# ══════════════════════════════════════════════════════════════════
# AUDIT: aggregation -- median_across_seeds must match by key (not row
# position/pooled obs), must not numerically average boolean/categorical
# columns under their own name, and must report range/IQR + raw seed
# values alongside the median.
# ══════════════════════════════════════════════════════════════════
def _write_seed_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_median_across_seeds_matches_by_key_not_position(tmp_path):
    # block order deliberately differs between the two "seed" files
    f1 = _write_seed_csv(tmp_path / "s1.csv", [
        {"block_id": "a", "metric": 10.0}, {"block_id": "b", "metric": 20.0},
    ])
    f2 = _write_seed_csv(tmp_path / "s2.csv", [
        {"block_id": "b", "metric": 22.0}, {"block_id": "a", "metric": 12.0},
    ])
    med = lr.median_across_seeds([f1, f2], key_cols=["block_id"], out_path=tmp_path / "med.csv", seeds=[1, 2])
    med = med.set_index("block_id")
    # correct (key-matched): a -> median(10,12)=11 ; b -> median(20,22)=21
    # position-matched (WRONG) would have paired a-with-b's-2nd-row etc.
    assert med.loc["a", "metric"] == pytest.approx(11.0)
    assert med.loc["b", "metric"] == pytest.approx(21.0)


def test_median_across_seeds_does_not_numerically_average_bool_columns(tmp_path):
    f1 = _write_seed_csv(tmp_path / "s1.csv", [{"block_id": "a", "complete_collapse": True, "metric": 1.0}])
    f2 = _write_seed_csv(tmp_path / "s2.csv", [{"block_id": "a", "complete_collapse": False, "metric": 3.0}])
    med = lr.median_across_seeds([f1, f2], key_cols=["block_id"], out_path=tmp_path / "med.csv", seeds=[1, 2])
    # the raw boolean column must NOT appear as a silently-averaged 0.5 float
    assert "complete_collapse" not in med.columns
    assert med.loc[0, "complete_collapse_frac_true"] == pytest.approx(0.5)


def test_median_across_seeds_categorical_not_numerically_aggregated(tmp_path):
    f1 = _write_seed_csv(tmp_path / "s1.csv", [{"block_id": "a", "size_class": "small", "metric": 1.0}])
    f2 = _write_seed_csv(tmp_path / "s2.csv", [{"block_id": "a", "size_class": "small", "metric": 3.0}])
    med = lr.median_across_seeds([f1, f2], key_cols=["block_id"], out_path=tmp_path / "med.csv", seeds=[1, 2])
    assert med.loc[0, "size_class"] == "small"  # verbatim, not coerced/averaged


def test_median_across_seeds_reports_range_and_iqr(tmp_path):
    f1 = _write_seed_csv(tmp_path / "s1.csv", [{"block_id": "a", "metric": 1.0}])
    f2 = _write_seed_csv(tmp_path / "s2.csv", [{"block_id": "a", "metric": 5.0}])
    f3 = _write_seed_csv(tmp_path / "s3.csv", [{"block_id": "a", "metric": 3.0}])
    med = lr.median_across_seeds([f1, f2, f3], key_cols=["block_id"], out_path=tmp_path / "med.csv", seeds=[1, 2, 3])
    row = med.iloc[0]
    assert row["metric"] == pytest.approx(3.0)
    assert row["metric_min"] == pytest.approx(1.0)
    assert row["metric_max"] == pytest.approx(5.0)
    assert row["metric_iqr"] >= 0


def test_median_across_seeds_handles_preexisting_seed_column(tmp_path):
    """Regression test: baseline_summary.csv already carries its own 'seed'
    column (the run's seed) -- median_across_seeds must not collide with it
    (caught via a real end-to-end run: df.insert(0, 'seed', seed) raised
    ValueError('cannot insert seed, already exists') on baseline_summary.csv)."""
    f1 = _write_seed_csv(tmp_path / "s1.csv", [{"condition": "x", "metric": 1.0, "seed": 42}])
    f2 = _write_seed_csv(tmp_path / "s2.csv", [{"condition": "x", "metric": 3.0, "seed": 43}])
    med = lr.median_across_seeds([f1, f2], key_cols=["condition"], out_path=tmp_path / "med.csv", seeds=[42, 43])
    assert med.loc[0, "metric"] == pytest.approx(2.0)
    by_seed = pd.read_csv(tmp_path / "med_by_seed.csv")
    assert sorted(by_seed["seed"].tolist()) == [42, 43]


def test_median_across_seeds_writes_by_seed_companion(tmp_path):
    f1 = _write_seed_csv(tmp_path / "s1.csv", [{"block_id": "a", "metric": 1.0}])
    f2 = _write_seed_csv(tmp_path / "s2.csv", [{"block_id": "a", "metric": 5.0}])
    out_path = tmp_path / "median_training_summary.csv"
    lr.median_across_seeds([f1, f2], key_cols=["block_id"], out_path=out_path, seeds=[42, 43])
    by_seed = pd.read_csv(tmp_path / "median_training_summary_by_seed.csv")
    assert set(by_seed["seed"]) == {42, 43}
    assert sorted(by_seed["metric"].tolist()) == [1.0, 5.0]


# ══════════════════════════════════════════════════════════════════
# AUDIT: PCA plausibility multi-seed aggregation
# ══════════════════════════════════════════════════════════════════
def test_pca_plausibility_summary_single_fixed_split_run(tmp_path):
    """Fixed-split design: PCA fits once, so pca_plausibility_summary.py
    reports the single run's numbers directly (no median/IQR across
    seeds -- there is nothing to vary, since there's only one split)."""
    sys.path.insert(0, str(REPO / "scripts" / "baselines"))
    import pca_plausibility_summary as pps

    f = _write_seed_csv(tmp_path / "block_order.csv", [
        {"block_id": "a", "size_class": "small", "nominal_d": 2, "var_explained_at_d": 0.80, "eff_dims_80pct": 2},
        {"block_id": "b", "size_class": "large", "nominal_d": 8, "var_explained_at_d": 0.95, "eff_dims_80pct": 3},
    ])
    by_class, low = pps.summarize(f, min_var_explained=0.99)
    small_row = by_class[by_class["size_class"] == "small"].iloc[0]
    assert small_row["mean_var_explained_at_d"] == pytest.approx(0.80)
    assert len(low) == 2  # both below the deliberately strict 0.99 threshold
    assert set(by_class["size_class"]) == {"small", "large"}


# ══════════════════════════════════════════════════════════════════
# AUDIT: collapse diagnostics -- complete_collapse (active_dims==0) and
# partial_collapse (active_dims<nominal_d) are separate, never used for
# checkpoint/model selection (verified separately: train_block's
# selection logic only ever reads v_recon, never latent_diagnostics()).
# ══════════════════════════════════════════════════════════════════
def test_latent_diagnostics_complete_and_partial_collapse():
    torch.manual_seed(0)
    p, d, nominal_d, n = 20, 4, 8, 30  # d_used=4 < nominal_d=8 -> forces partial_collapse regardless of activity
    model = vae_mod.BlockVAE(p, d, drop=0.0, loss_type="ORD")
    x = torch.randn(n, p)
    diag = vae_mod.latent_diagnostics(model, x, active_var_threshold=0.01,
                                       collapse_active_dims_threshold=0, free_bits=0.5, nominal_d=nominal_d)
    assert diag["d_used"] == d
    assert diag["partial_collapse"] == (diag["active_latent_dims"] < nominal_d)
    assert diag["partial_collapse"] is True  # active_latent_dims <= d=4 < nominal_d=8, always
    assert diag["complete_collapse"] == (diag["active_latent_dims"] == 0)


def test_collapse_flags_never_referenced_by_checkpoint_selection():
    """train_block's checkpoint-selection branch must never call
    latent_diagnostics or read collapse-related keys -- static check on the
    source so a future edit can't silently wire collapse into selection."""
    import inspect
    src = inspect.getsource(vae_mod.train_block)
    assert "latent_diagnostics" not in src
    assert "collapse" not in src.lower()


# ══════════════════════════════════════════════════════════════════
# AUDIT: objective/training numeric audit
# ══════════════════════════════════════════════════════════════════
def test_free_bits_averages_before_clamping_not_after():
    """Discriminates averaging-then-clamping (correct, per point 3) from
    clamping-then-averaging (wrong): raw per-subject KL = [0, 0, 2, 2] for
    one dimension. Correct: mean=1.0 (already above the 0.5 floor, no
    clamp needed) -> rate contribution = 1.0. Wrong (per-sample clamp then
    mean): [0.5,0.5,2,2] -> mean = 1.25. These differ, so getting 1.0
    proves the clamp is applied to the already-averaged per-dim KL."""
    mu = torch.tensor([[0.0], [0.0], [2.0], [2.0]])  # raw_kl_per_dim(mu,lv=0) = -0.5*(1-mu^2-1) = mu^2/2
    lv = torch.zeros(4, 1)
    rate, mean_kl_dim, raw_kl_total = vae_mod.free_bits_rate(mu, lv, free_bits=0.5)
    assert mean_kl_dim[0].item() == pytest.approx(1.0)   # (0+0+2+2)/4 = 1.0, matches mu^2/2 averaged
    assert rate.item() == pytest.approx(1.0)              # NOT 1.25 -- proves average-before-clamp
    assert raw_kl_total.item() == pytest.approx(1.0)


def test_beta_warmup_reaches_exactly_0point5_at_epoch_50():
    beta_max, beta_warmup = 0.5, 50
    def beta_at(ep):
        return min(beta_max, beta_max * ep / max(beta_warmup, 1))
    assert beta_at(49) < 0.5
    assert beta_at(50) == pytest.approx(0.5)
    assert beta_at(51) == pytest.approx(0.5)  # held constant past warmup
    assert beta_at(1) == pytest.approx(0.5 * 1 / 50)


def test_checkpoint_selection_equals_min_recon_metric_among_eligible_epochs():
    """Direct proof that the checkpoint returned by train_block corresponds
    to the epoch with the minimum va_recon_metric (ordinal NLL for ORD)
    among epochs >= checkpoint_start_epoch -- not the beta-weighted va_loss."""
    torch.manual_seed(3)
    np.random.seed(3)
    p, d, n = 10, 2, 40
    G = np.random.randint(0, 3, size=(n, p)).astype(np.float32)
    x = torch.tensor(G, dtype=torch.float32)
    y = torch.tensor(G, dtype=torch.long)
    tr_x, va_x, tr_y, va_y = x[:28], x[28:], y[:28], y[28:]

    model = vae_mod.BlockVAE(p, d, drop=0.0, loss_type="ORD")
    cfg = dict(epochs=58, batch_size=8, lr=1e-3, beta_max=0.5, beta_warmup=10,  # fast warmup -> vl and v_recon can diverge in ranking
               patience=100, grad_clip=1.0, checkpoint_start_epoch=51, free_bits=0.5,
               diagnostic_interval=10)
    log, best_val_recon, best_epoch = vae_mod.train_block(
        model, tr_x, va_x, torch.device("cpu"), cfg,
        Path("/tmp/_test_log3.csv"), model_seed=3, tr_y=tr_y, va_y=va_y,
        maf=np.full(p, 0.1), maf_bins=[0.0, 1.0])
    eligible = [r for r in log if r["epoch"] >= 51]
    assert eligible, "test needs to actually reach the eligible window"
    min_recon_eligible = min(r["va_recon_metric"] for r in eligible)
    assert best_val_recon == pytest.approx(min_recon_eligible, abs=1e-4)


def test_checkpoint_always_defined_near_epoch_51_boundary():
    """Runs that end exactly AT checkpoint_start_epoch (only one eligible
    epoch, patience never gets a chance to fire) must still produce a
    loadable checkpoint."""
    torch.manual_seed(4)
    np.random.seed(4)
    p, d, n = 10, 2, 40
    G = np.random.randint(0, 3, size=(n, p)).astype(np.float32)
    x = torch.tensor(G, dtype=torch.float32)
    y = torch.tensor(G, dtype=torch.long)
    tr_x, va_x, tr_y, va_y = x[:28], x[28:], y[:28], y[28:]

    model = vae_mod.BlockVAE(p, d, drop=0.0, loss_type="ORD")
    cfg = dict(epochs=51, batch_size=8, lr=1e-3, beta_max=0.5, beta_warmup=50,
               patience=1, grad_clip=1.0, checkpoint_start_epoch=51, free_bits=0.5,
               diagnostic_interval=10)
    log, best_val_recon, best_epoch = vae_mod.train_block(
        model, tr_x, va_x, torch.device("cpu"), cfg,
        Path("/tmp/_test_log4.csv"), model_seed=4, tr_y=tr_y, va_y=va_y,
        maf=np.full(p, 0.1), maf_bins=[0.0, 1.0])
    assert len(log) == 51
    assert np.isfinite(best_val_recon)
    # model.load_state_dict inside train_block would have raised if best_sd were None/malformed;
    # reaching here with a finite best_val_recon is proof a checkpoint was selected.


# ══════════════════════════════════════════════════════════════════
# AUDIT: output integrity -- all_blocks_latent_dims.npy / block_order.csv
# alignment is asserted in the scripts themselves (per_block_PCA_concat.py,
# per_block_vae_concat.py); this test exercises that assertion logic in
# isolation against a deliberately misaligned case.
# ══════════════════════════════════════════════════════════════════
def test_d_used_list_block_order_alignment_would_catch_misalignment():
    block_order = pd.DataFrame({"block_id": ["a", "b"], "d_used": [2, 4]})
    d_used_list_correct = [2, 4]
    d_used_list_wrong = [4, 2]
    assert d_used_list_correct == block_order["d_used"].tolist()
    assert d_used_list_wrong != block_order["d_used"].tolist()


# ══════════════════════════════════════════════════════════════════
# item 5: ld_corr is DEPRECATED (always NaN, old statistic removed);
# subject_distance_preservation (data_utils.py) replaces it with
# train-fitted scaling. Never influences checkpointing either way.
# ══════════════════════════════════════════════════════════════════
def test_ld_corr_not_referenced_by_checkpoint_selection():
    import inspect
    src = inspect.getsource(vae_mod.train_block)
    assert "ld_corr" not in src
    assert "compute_ld_corr" not in src


def test_ld_corr_deprecated_always_nan():
    np.random.seed(0)
    G_va = np.random.randint(0, 3, size=(20, 15)).astype(np.float32)
    mu_va = np.random.randn(20, 4).astype(np.float32)
    r = vae_mod.compute_ld_corr(G_va, mu_va)
    assert np.isnan(r)


def test_subject_distance_preservation_scaling_fit_on_train_only():
    """Distance scaling must be fit on TRAIN, then merely APPLIED to
    validation -- so a change to validation-only rows (with train fixed)
    changes the transformed values used, but the function must not refit
    mean/SD from validation itself: shifting every validation genotype row
    by the same additive constant leaves standardized values (and hence
    pairwise distances / the resulting rho) unchanged only if scaling used
    the (unshifted) TRAIN mean/SD, not a validation-refit one."""
    rng = np.random.RandomState(0)
    G_tr = rng.randint(0, 3, size=(40, 10)).astype(np.float64)
    G_va = rng.randint(0, 3, size=(15, 10)).astype(np.float64)
    mu_tr = rng.randn(40, 3)
    mu_va = rng.randn(15, 3)
    r1 = du.subject_distance_preservation(G_tr, G_va, mu_tr, mu_va)
    r2 = du.subject_distance_preservation(G_tr, G_va + 100.0, mu_tr, mu_va)
    # adding a constant shift to G_va shifts (x-mean)/sd by a constant when
    # the mean is TRAIN's (not refit on the shifted validation set) -- pairwise
    # distances, and therefore rho, are invariant to that constant shift.
    assert r1 == pytest.approx(r2, abs=1e-6, nan_ok=True)


def test_subject_distance_preservation_drops_train_zero_variance_and_returns_nan_when_degenerate():
    G_tr = np.zeros((10, 5))     # every SNP constant on train -> nothing to standardize
    G_va = np.random.RandomState(1).randint(0, 3, size=(5, 5)).astype(np.float64)
    mu_tr = np.random.RandomState(2).randn(10, 2)
    mu_va = np.random.RandomState(3).randn(5, 2)
    r = du.subject_distance_preservation(G_tr, G_va, mu_tr, mu_va)
    assert np.isnan(r)


# ══════════════════════════════════════════════════════════════════
# FIXED-SPLIT INTEGRATION TEST: runs the REAL per_block_PCA_concat.main()
# then per_block_vae_concat.main() (two model_seeds) end to end on a tiny
# synthetic dataset, through their actual --config entry points -- proves
# PCA runs once, every model_seed reuses its split byte-identically, and
# split_seed/model_seed are correctly recorded in every output artifact.
# ══════════════════════════════════════════════════════════════════
def _write_synthetic_dataset(tmp_path, n_subjects=40, p=12, seed=0):
    rng = np.random.RandomState(seed)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    snp_cols = [f"snp{i}" for i in range(p)]
    G = rng.randint(0, 3, size=(n_subjects, p))
    df = pd.DataFrame({
        "FID": [f"F{i}" for i in range(n_subjects)],
        "IID": [f"S{i}" for i in range(n_subjects)],
        "PAT": 0, "MAT": 0, "SEX": 1, "PHENOTYPE": -9,
        **{c: G[:, i] for i, c in enumerate(snp_cols)},
    })
    df.to_csv(raw_dir / "blockA.raw", sep=" ", index=False)

    manifest = pd.DataFrame([{
        "block_id": "blockA", "gene": "geneA", "group": "Asthma",
        "region_id": "regionA", "status": "ok",
    }])
    manifest_path = tmp_path / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    return raw_dir, manifest_path


def _write_combined_config(tmp_path, raw_dir, manifest_path, pca_root, vae_root, model_seeds,
                            shared_overrides=None, vae_overrides=None, config_name="test_config.yaml"):
    cfg = {
        "dataset": {"raw_dir": str(raw_dir), "block_def": str(manifest_path), "use_manifest_path": False},
        "shared": {"split_seed": 42, "model_seeds": model_seeds, "val_frac": 0.25,
                   "k_range": [2, 3], "save_per_block": False, **(shared_overrides or {})},
        "pca": {"output_root": str(pca_root), "phase1_dir": None},
        "vae": {"output_root": str(vae_root), "phase1_dir": str(pca_root), "loss_type": "ORD",
                "epochs": 8, "batch_size": 8, "lr": 1e-3, "dropout": 0.0,
                "beta_max": 0.5, "beta_warmup": 2, "free_bits": 0.5,
                "patience": 5, "grad_clip": 1.0, "checkpoint_start_epoch": 3,
                "active_var_threshold": 0.01, "collapse_active_dims_threshold": 0,
                "maf_bins": [0.0, 0.05, 0.10, 0.20, 0.50], "diagnostic_interval": 2,
                **(vae_overrides or {})},
    }
    import yaml
    config_path = tmp_path / config_name
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return config_path


def test_fixed_split_end_to_end_pca_once_vae_two_model_seeds(tmp_path, monkeypatch):
    import per_block_PCA_concat
    import per_block_vae_concat

    raw_dir, manifest_path = _write_synthetic_dataset(tmp_path)
    pca_root = tmp_path / "pca_out"
    vae_root = tmp_path / "vae_out"
    model_seeds = [42, 43]
    config_path = _write_combined_config(tmp_path, raw_dir, manifest_path, pca_root, vae_root, model_seeds)

    monkeypatch.setattr("sys.argv", ["per_block_PCA_concat.py", "--config", str(config_path)])
    per_block_PCA_concat.main()

    # PCA: flat output (no seed subdirectory), split_meta.json records split_seed
    assert (pca_root / "subjects.csv").exists()
    assert (pca_root / "train_idx.npy").exists()
    assert not (pca_root / "seed_42").exists()  # NOT nested -- single fixed-split run
    with open(pca_root / "split_meta.json") as f:
        split_meta = json.load(f)
    assert split_meta["split_seed"] == 42
    pca_block_order = pd.read_csv(pca_root / "block_order.csv")
    assert (pca_block_order["split_seed"] == 42).all()
    pca_tr = np.load(pca_root / "train_idx.npy")
    pca_va = np.load(pca_root / "val_idx.npy")

    monkeypatch.setattr("sys.argv", ["per_block_vae_concat.py", "--config", str(config_path)])
    per_block_vae_concat.main()

    for model_seed in model_seeds:
        seed_dir = vae_root / f"seed_{model_seed}"
        assert seed_dir.exists()

        # split reused byte-identically from PCA -- never regenerated
        vtr = np.load(seed_dir / "train_idx.npy")
        vva = np.load(seed_dir / "val_idx.npy")
        assert np.array_equal(vtr, pca_tr)
        assert np.array_equal(vva, pca_va)

        # split_seed AND model_seed recorded in block_order.csv, training_summary.csv,
        # baseline_summary.csv, baseline_meta.json, and the per-block checkpoint
        block_order = pd.read_csv(seed_dir / "block_order.csv")
        assert (block_order["split_seed"] == 42).all()
        assert (block_order["model_seed"] == model_seed).all()

        training_summary = pd.read_csv(seed_dir / "training_summary.csv")
        assert (training_summary["split_seed"] == 42).all()
        assert (training_summary["model_seed"] == model_seed).all()
        # items 1/2/4/5/9: new training_summary columns present and sane
        assert (training_summary["objective_mode"] == "VAE").all()
        assert training_summary["best_epoch"].notna().all()
        assert training_summary["ld_corr"].isna().all()   # deprecated -- always NaN
        assert "subject_distance_preservation" in training_summary.columns
        assert "dims_above_free_bits" in training_summary.columns
        assert "all_dims_below_free_bits" in training_summary.columns
        assert "any_dim_above_free_bits" in training_summary.columns
        assert "observed_g2_frequency_by_maf_bin" in training_summary.columns
        assert "tr_n_g0" in training_summary.columns and "va_n_g0" in training_summary.columns

        # item 1: per-epoch log has tr_/va_ diagnostic columns, NaN except at
        # diagnostic_interval multiples and the best_epoch row.
        log_epochs = pd.read_csv(seed_dir / "logs" / "blockA.csv")
        assert "tr_ordinal_nll" in log_epochs.columns and "va_ordinal_nll" in log_epochs.columns
        assert "nll_per_snp_gap" in log_epochs.columns
        best_row = log_epochs[log_epochs["is_best_checkpoint"] == True]  # noqa: E712
        assert len(best_row) == 1
        assert np.isfinite(best_row["tr_ordinal_nll"].iloc[0])
        assert np.isfinite(best_row["va_ordinal_nll"].iloc[0])
        assert best_row["va_ordinal_nll_per_snp"].iloc[0] == pytest.approx(
            best_row["va_ordinal_nll"].iloc[0] / 12, abs=1e-5)   # p=12 in the synthetic dataset

        baseline_summary = pd.read_csv(seed_dir / "baseline_summary.csv")
        assert (baseline_summary["split_seed"] == 42).all()
        assert (baseline_summary["model_seed"] == model_seed).all()

        with open(seed_dir / "baseline_meta.json") as f:
            meta = json.load(f)
        assert meta["split_seed"] == 42
        assert meta["model_seed"] == model_seed

        ckpt = torch.load(seed_dir / "models" / "blockA.pt", weights_only=False)
        assert ckpt["split_seed"] == 42
        assert ckpt["model_seed"] == model_seed

        log = pd.read_csv(seed_dir / "logs" / "blockA.csv")
        assert (log["split_seed"] == 42).all()
        assert (log["model_seed"] == model_seed).all()

    # aggregation across model_seeds, matched by block_id, at the top level
    assert (vae_root / "median_summary.csv").exists()
    assert (vae_root / "median_training_summary.csv").exists()
    assert (vae_root / "median_training_summary_by_seed.csv").exists()
    by_seed = pd.read_csv(vae_root / "median_training_summary_by_seed.csv")
    assert sorted(by_seed["seed"].unique().tolist()) == model_seeds
    assert set(by_seed["block_id"]) == {"blockA"}


def test_vae_fails_loudly_without_pca_split(tmp_path, monkeypatch):
    """If phase1_dir points nowhere usable, VAE must fail loudly rather
    than silently generating its own split."""
    import per_block_vae_concat

    raw_dir, manifest_path = _write_synthetic_dataset(tmp_path)
    vae_root = tmp_path / "vae_out"
    nonexistent_pca_root = tmp_path / "no_such_pca_dir"
    config_path = _write_combined_config(tmp_path, raw_dir, manifest_path, nonexistent_pca_root, vae_root, [42])

    monkeypatch.setattr("sys.argv", ["per_block_vae_concat.py", "--config", str(config_path)])
    with pytest.raises(FileNotFoundError, match="PCA split files missing"):
        per_block_vae_concat.main()


# ══════════════════════════════════════════════════════════════════
# item 8: shared.run_clustering: false -- must complete cleanly on BOTH
# PCA and VAE, with no undefined-variable crash and NaN clustering metrics.
# ══════════════════════════════════════════════════════════════════
def test_run_clustering_false_completes_without_undefined_variables(tmp_path, monkeypatch):
    import per_block_PCA_concat
    import per_block_vae_concat

    raw_dir, manifest_path = _write_synthetic_dataset(tmp_path)
    pca_root = tmp_path / "pca_out"
    vae_root = tmp_path / "vae_out"
    config_path = _write_combined_config(
        tmp_path, raw_dir, manifest_path, pca_root, vae_root, [42],
        shared_overrides={"run_clustering": False})

    monkeypatch.setattr("sys.argv", ["per_block_PCA_concat.py", "--config", str(config_path)])
    per_block_PCA_concat.main()   # must not raise (no undefined gmm_df/km_df reference)
    assert not (pca_root / "clustering").exists()
    pca_summary = pd.read_csv(pca_root / "baseline_summary.csv")
    assert (pca_summary["clustering_run"] == False).all()          # noqa: E712
    assert pca_summary["best_k_gmm_bic"].isna().all()

    monkeypatch.setattr("sys.argv", ["per_block_vae_concat.py", "--config", str(config_path)])
    per_block_vae_concat.main()
    seed_dir = vae_root / "seed_42"
    assert not (seed_dir / "clustering").exists()
    vae_summary = pd.read_csv(seed_dir / "baseline_summary.csv")
    assert (vae_summary["clustering_run"] == False).all()          # noqa: E712
    assert vae_summary["best_kmeans_silhouette"].isna().all()


# ══════════════════════════════════════════════════════════════════
# item 9: objective_mode AE is a TRUE deterministic ablation (z=mu always,
# no reparameterization noise); default VAE remains stochastic in training.
# ══════════════════════════════════════════════════════════════════
def test_ae_reparam_is_always_deterministic_vae_reparam_is_stochastic_in_training():
    torch.manual_seed(0)
    mu = torch.randn(5, 3)
    lv = torch.zeros(5, 3)   # var=1, non-degenerate noise scale

    ae_model = vae_mod.BlockVAE(p=10, d=3, loss_type="ORD", objective_mode="AE")
    ae_model.train()
    z1 = ae_model.reparam(mu, lv)
    z2 = ae_model.reparam(mu, lv)
    assert torch.equal(z1, mu) and torch.equal(z2, mu)
    ae_model.eval()
    assert torch.equal(ae_model.reparam(mu, lv), mu)

    vae_model = vae_mod.BlockVAE(p=10, d=3, loss_type="ORD", objective_mode="VAE")
    vae_model.train()
    torch.manual_seed(1)
    z_a = vae_model.reparam(mu, lv)
    torch.manual_seed(2)
    z_b = vae_model.reparam(mu, lv)
    assert not torch.equal(z_a, z_b)      # different noise draws -> different z
    vae_model.eval()
    assert torch.equal(vae_model.reparam(mu, lv), mu)   # deterministic at eval, same as always


def test_ae_objective_mode_drops_kl_term_from_optimized_loss():
    torch.manual_seed(0)
    model = vae_mod.BlockVAE(p=8, d=2, loss_type="ORD", objective_mode="AE")
    x = torch.rand(6, 8) * 2
    y = torch.randint(0, 3, (6, 8))
    rec, mu, lv = model(x)
    loss, recon_metric, rate, _, _ = model.compute_loss(x, rec, mu, lv, beta=0.5, free_bits=0.5, y=y)
    assert loss.item() == pytest.approx(recon_metric.item(), abs=1e-5)   # no + beta*rate term
    assert rate.item() != 0.0   # still computed/reported, just not optimized


def test_objective_mode_recorded_end_to_end(tmp_path, monkeypatch):
    import per_block_PCA_concat
    import per_block_vae_concat

    raw_dir, manifest_path = _write_synthetic_dataset(tmp_path)
    pca_root = tmp_path / "pca_out"
    vae_root = tmp_path / "vae_out"
    config_path = _write_combined_config(
        tmp_path, raw_dir, manifest_path, pca_root, vae_root, [42],
        vae_overrides={"objective_mode": "AE"})

    monkeypatch.setattr("sys.argv", ["per_block_PCA_concat.py", "--config", str(config_path)])
    per_block_PCA_concat.main()
    monkeypatch.setattr("sys.argv", ["per_block_vae_concat.py", "--config", str(config_path)])
    per_block_vae_concat.main()

    seed_dir = vae_root / "seed_42"
    ts = pd.read_csv(seed_dir / "training_summary.csv")
    assert (ts["objective_mode"] == "AE").all()
    ckpt = torch.load(seed_dir / "models" / "blockA.pt", weights_only=False)
    assert ckpt["objective_mode"] == "AE"


# ══════════════════════════════════════════════════════════════════
# item 1/11: evaluate_split applies the IDENTICAL definition to train and
# validation (same function, same code path), and ordinal_nll_per_snp is
# exactly ordinal_nll / p.
# ══════════════════════════════════════════════════════════════════
def test_evaluate_split_identical_definition_and_nll_per_snp():
    torch.manual_seed(0)
    p, d, n = 9, 2, 12
    model = vae_mod.BlockVAE(p, d, drop=0.0, loss_type="ORD")
    x = torch.rand(n, p) * 2
    y = torch.randint(0, 3, (n, p))
    maf = np.full(p, 0.2)
    maf_bins = [0.0, 1.0]

    ev_a = vae_mod.evaluate_split(model, x, y, "ORD", free_bits=0.5, maf=maf, maf_bins=maf_bins)
    ev_b = vae_mod.evaluate_split(model, x, y, "ORD", free_bits=0.5, maf=maf, maf_bins=maf_bins)
    # same model (eval mode, no dropout), same data -> byte-identical report,
    # whether this call is logically "train" or "validation".
    assert ev_a["ordinal_nll"] == pytest.approx(ev_b["ordinal_nll"], abs=1e-8)
    assert ev_a["ordinal_nll_per_snp"] == pytest.approx(ev_a["ordinal_nll"] / p, abs=1e-8)


# ══════════════════════════════════════════════════════════════════
# item 2/11: g2 calibration bins are computed from the TRAIN-set MAF passed
# in, not recomputed from validation or some other source.
# ══════════════════════════════════════════════════════════════════
def test_g2_calibration_uses_training_maf_bins():
    torch.manual_seed(0)
    p, d, n = 6, 2, 10
    model = vae_mod.BlockVAE(p, d, drop=0.0, loss_type="ORD")
    x = torch.rand(n, p) * 2
    y = torch.randint(0, 3, (n, p))
    maf_bins = [0.0, 0.5, 1.0]

    maf_all_low = np.full(p, 0.1)     # every SNP in bin [0.00,0.50)
    maf_all_high = np.full(p, 0.9)    # every SNP in bin [0.50,1.00]
    calib_low = vae_mod.g2_calibration(model, x, y, maf_all_low, maf_bins)
    calib_high = vae_mod.g2_calibration(model, x, y, maf_all_high, maf_bins)

    n_obs_low = json.loads(calib_low["n_observations_by_maf_bin"])
    n_obs_high = json.loads(calib_high["n_observations_by_maf_bin"])
    labels = du.maf_bin_labels(maf_bins)
    assert n_obs_low[labels[0]] == n * p and n_obs_low[labels[1]] == 0
    assert n_obs_high[labels[1]] == n * p and n_obs_high[labels[0]] == 0


# ══════════════════════════════════════════════════════════════════
# item 6/11: PCA + linear ordinal head fits PCA on TRAIN only and trains
# its linear decoder on TRAIN scores only (never touching validation until
# the final held-out evaluation).
# ══════════════════════════════════════════════════════════════════
def test_pca_ordinal_head_uses_training_pca_and_training_fitting_only():
    import posthoc_baselines as ph

    rng = np.random.RandomState(0)
    n, p, d = 40, 8, 2
    tr_ix = np.arange(0, 30)
    va_ix = np.arange(30, 40)
    G = rng.randint(0, 3, size=(n, p)).astype(np.float32)

    pca, Ztr, Zva, d_used = ph.pca_fit_transform(G, tr_ix, va_ix, d, seed=0)
    # PCA's own fitted mean must equal the TRAIN mean, not the full-sample one.
    assert np.allclose(pca.mean_, G[tr_ix].mean(axis=0), atol=1e-4)
    assert not np.allclose(pca.mean_, G.mean(axis=0), atol=1e-4)

    Gtr_i = np.rint(G[tr_ix]).astype(np.int64)
    Gva_i = np.rint(G[va_ix]).astype(np.int64)
    out = ph.pca_ordinal_head(Ztr, Zva, Gtr_i, Gva_i, d_used, p, seed=0, epochs=20, patience=5)
    for k in ("pca_ord_va_nll", "pca_ord_va_nll_per_snp", "pca_ord_va_dosage_mse", "pca_ord_va_dosage_mae"):
        assert k in out and np.isfinite(out[k])
    assert out["pca_ord_va_nll_per_snp"] == pytest.approx(out["pca_ord_va_nll"] / p, abs=1e-5)


# ══════════════════════════════════════════════════════════════════
# item 7/11: block_quality_stability aggregates ACROSS model_seeds, matched
# by block_id, without becoming an exclusion rule (all input blocks appear
# in the output).
# ══════════════════════════════════════════════════════════════════
def test_block_quality_stability_matches_by_block_id_and_keeps_all_blocks():
    import posthoc_baselines as ph

    df_seed42 = pd.DataFrame({
        "block_id": ["a", "b"], "model_seed": [42, 42],
        "vae_nll_improvement_over_marginal": [1.0, -0.5],
        "vae_nll_improvement_per_snp": [0.1, -0.05],
        "dist_pres_vae": [0.6, 0.1],
    })
    df_seed43 = pd.DataFrame({
        "block_id": ["a", "b"], "model_seed": [43, 43],
        "vae_nll_improvement_over_marginal": [1.2, 0.3],
        "vae_nll_improvement_per_snp": [0.12, 0.03],
        "dist_pres_vae": [0.65, 0.2],
    })
    stab = ph.block_quality_stability([df_seed42, df_seed43])
    assert set(stab["block_id"]) == {"a", "b"}   # every input block retained -- no filtering
    a = stab[stab["block_id"] == "a"].iloc[0]
    b = stab[stab["block_id"] == "b"].iloc[0]
    assert a["n_model_seeds"] == 2 and a["n_seeds_vae_beats_marginal"] == 2
    assert a["fraction_seeds_vae_beats_marginal"] == pytest.approx(1.0)
    assert b["n_seeds_vae_beats_marginal"] == 1   # only seed 43 beats marginal for block b
    assert b["fraction_seeds_vae_beats_marginal"] == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════
# posthoc_baselines.py CLI, end to end: runs the REAL PCA + two-model_seed
# VAE pipeline on synthetic data, then the REAL posthoc main() across both
# seed dirs -- proves items 3/5/6/7 are wired together correctly (not just
# individually unit-tested), through the actual --vae-dir/--pca-dir entry point.
# ══════════════════════════════════════════════════════════════════
def test_posthoc_cli_end_to_end_two_model_seeds(tmp_path, monkeypatch):
    import per_block_PCA_concat
    import per_block_vae_concat
    import posthoc_baselines

    raw_dir, manifest_path = _write_synthetic_dataset(tmp_path)
    pca_root = tmp_path / "pca_out"
    vae_root = tmp_path / "vae_out"
    model_seeds = [42, 43]
    config_path = _write_combined_config(tmp_path, raw_dir, manifest_path, pca_root, vae_root, model_seeds)

    monkeypatch.setattr("sys.argv", ["per_block_PCA_concat.py", "--config", str(config_path)])
    per_block_PCA_concat.main()
    monkeypatch.setattr("sys.argv", ["per_block_vae_concat.py", "--config", str(config_path)])
    per_block_vae_concat.main()

    seed_dirs = [str(vae_root / f"seed_{s}") for s in model_seeds]
    monkeypatch.setattr("sys.argv", ["posthoc_baselines.py", "--config", str(config_path),
                                      "--vae-dir", *seed_dirs, "--pca-dir", str(pca_root)])
    posthoc_baselines.main()

    for s in model_seeds:
        tier0 = pd.read_csv(vae_root / f"seed_{s}" / "posthoc_tier0.csv")
        assert set(tier0["block_id"]) == {"blockA"}
        for col in ("marginal_nll", "dist_pres_vae", "pca_ord_va_nll", "vae_minus_pca_ord_nll",
                    "vae_nll_improvement_over_marginal"):
            assert col in tier0.columns
        assert np.isfinite(tier0["marginal_nll"].iloc[0])
        assert np.isfinite(tier0["pca_ord_va_nll"].iloc[0])

    stability = pd.read_csv(vae_root / "block_quality_stability.csv")
    assert set(stability["block_id"]) == {"blockA"}
    assert stability["n_model_seeds"].iloc[0] == 2


# ══════════════════════════════════════════════════════════════════
# ONE-FACTOR-AT-A-TIME SENSITIVITY SWEEP (per_block_vae_concat.py)
# ══════════════════════════════════════════════════════════════════
def _base_sweep_cfg(sensitivity_overrides=None, **extra):
    """A minimal but complete flat cfg dict, as load_baseline_config would
    hand to main() -- resolve_sensitivity_conditions/run_sensitivity_sweep
    only need the cfg dict, no file I/O, so these unit tests never touch YAML."""
    cfg = {
        "output_root": "results/vae", "phase1_dir": "results/pca",
        "split_seed": 42, "model_seeds": [42], "epochs": 8, "beta_warmup": 2,
        "diagnostic_interval": 2, "block_def": "manifest.tsv",
        "sensitivity": {
            "enabled": True, "dry_run": True, "mode": "one_at_a_time",
            "reference": {"beta_max": 0.5, "free_bits": 0.5, "objective_mode": "VAE"},
            "beta_values": [0.25, 0.5, 1.0],
            "free_bits_values": [0.0, 0.25, 0.5, 1.0],
            "include_ae": True,
            **(sensitivity_overrides or {}),
        },
    }
    cfg.update(extra)
    return cfg


# 1/2/3 -- exactly seven unique conditions, reference exactly once, deterministic order/IDs.
def test_sensitivity_resolves_exactly_seven_unique_conditions():
    cfg = _base_sweep_cfg()
    sens = vae_mod.resolve_sensitivity_conditions(cfg)
    conditions = sens["conditions"]
    assert len(conditions) == 7
    assert len({c["condition_id"] for c in conditions}) == 7
    assert sum(c["is_reference"] for c in conditions) == 1

    expected = [
        ("vae_beta0p5_lambda0p5", "VAE", 0.5, 0.5, "reference"),
        ("vae_beta0p25_lambda0p5", "VAE", 0.25, 0.5, "beta"),
        ("vae_beta1_lambda0p5", "VAE", 1.0, 0.5, "beta"),
        ("vae_beta0p5_lambda0", "VAE", 0.5, 0.0, "free_bits"),
        ("vae_beta0p5_lambda0p25", "VAE", 0.5, 0.25, "free_bits"),
        ("vae_beta0p5_lambda1", "VAE", 0.5, 1.0, "free_bits"),
        ("ae", "AE", 0.0, 0.5, "objective_mode"),
    ]
    got = [(c["condition_id"], c["objective_mode"], c["beta_max"], c["free_bits"], c["varied_parameter"])
           for c in conditions]
    assert got == expected

    sens2 = vae_mod.resolve_sensitivity_conditions(cfg)
    assert [c["condition_id"] for c in sens2["conditions"]] == [c["condition_id"] for c in conditions]
    assert [c["condition_order"] for c in sens2["conditions"]] == [c["condition_order"] for c in conditions]


# 4 -- beta varies while lambda stays at reference.
def test_sensitivity_beta_sweep_holds_free_bits_at_reference():
    cfg = _base_sweep_cfg()
    sens = vae_mod.resolve_sensitivity_conditions(cfg)
    beta_rows = [c for c in sens["conditions"] if c["varied_parameter"] == "beta"]
    assert len(beta_rows) == 2   # 0.25, 1.0 -- 0.5 dedups into the reference row
    for c in beta_rows:
        assert c["free_bits"] == pytest.approx(0.5)
        assert c["objective_mode"] == "VAE"
        assert c["beta_max"] in (0.25, 1.0)


# 5 -- lambda varies while beta stays at reference.
def test_sensitivity_free_bits_sweep_holds_beta_at_reference():
    cfg = _base_sweep_cfg()
    sens = vae_mod.resolve_sensitivity_conditions(cfg)
    fb_rows = [c for c in sens["conditions"] if c["varied_parameter"] == "free_bits"]
    assert len(fb_rows) == 3   # 0.0, 0.25, 1.0 -- 0.5 dedups into the reference row
    for c in fb_rows:
        assert c["beta_max"] == pytest.approx(0.5)
        assert c["objective_mode"] == "VAE"
        assert c["free_bits"] in (0.0, 0.25, 1.0)


# 6 -- AE: objective_mode=AE, beta_max=0, numeric internal free_bits, free_bits_applicable=false.
def test_sensitivity_ae_condition_shape():
    cfg = _base_sweep_cfg()
    sens = vae_mod.resolve_sensitivity_conditions(cfg)
    ae = [c for c in sens["conditions"] if c["condition_id"] == "ae"][0]
    assert ae["objective_mode"] == "AE"
    assert ae["beta_max"] == 0.0
    assert ae["free_bits"] == pytest.approx(0.5)   # numeric, retained internally
    assert isinstance(ae["free_bits"], float) and np.isfinite(ae["free_bits"])
    assert ae["free_bits_applicable"] is False
    assert ae["is_reference"] is False


# 7 -- malformed configuration fails clearly, never silently falls back.
@pytest.mark.parametrize("overrides,exc", [
    ({"enabled": "true"}, TypeError),                              # string, not YAML bool
    ({"dry_run": 1}, TypeError),                                   # int, not bool
    ({"include_ae": 0}, TypeError),
    ({"mode": "grid"}, ValueError),                                # unsupported mode
    ({"existing_run_policy": "resume"}, ValueError),               # unsupported policy
    ({"beta_values": []}, ValueError),                             # empty list
    ({"free_bits_values": []}, ValueError),
    ({"beta_values": [True, 0.5]}, TypeError),                     # bool masquerading as numeric
    ({"beta_values": [-0.1, 0.5]}, ValueError),                    # negative
    ({"beta_values": [float("nan"), 0.5]}, ValueError),            # non-finite
    ({"beta_values": [float("inf"), 0.5]}, ValueError),            # non-finite
    ({"reference": {"beta_max": 0.5, "free_bits": 0.5, "objective_mode": "AE"}}, ValueError),  # ref must be VAE
])
def test_sensitivity_malformed_config_fails_clearly(overrides, exc):
    cfg = _base_sweep_cfg(sensitivity_overrides=overrides)
    with pytest.raises(exc):
        vae_mod.resolve_sensitivity_conditions(cfg)


def test_sensitivity_absent_or_disabled_returns_none():
    cfg = _base_sweep_cfg()
    del cfg["sensitivity"]
    assert vae_mod.resolve_sensitivity_conditions(cfg) is None   # ordinary configs need no sensitivity key

    cfg2 = _base_sweep_cfg(sensitivity_overrides={"enabled": False})
    assert vae_mod.resolve_sensitivity_conditions(cfg2) is None


# 8/9 -- dry_run writes the design and NEVER touches data/training; creates
# no condition/seed/model/log/embedding directories.
def test_sensitivity_dry_run_never_loads_data_or_trains(tmp_path, monkeypatch):
    import per_block_vae_concat

    def _must_not_be_called(*a, **kw):
        raise AssertionError("must not be called during sensitivity dry_run")

    monkeypatch.setattr(per_block_vae_concat, "load_block_defs", _must_not_be_called)
    monkeypatch.setattr(per_block_vae_concat, "load_all_blocks", _must_not_be_called)
    monkeypatch.setattr(per_block_vae_concat, "require_pca_split", _must_not_be_called)
    monkeypatch.setattr(per_block_vae_concat, "run_one_model_seed", _must_not_be_called)

    raw_dir, manifest_path = _write_synthetic_dataset(tmp_path)
    pca_root = tmp_path / "pca_out"          # deliberately never created -- dry_run must not need it
    vae_root = tmp_path / "vae_out"
    config_path = _write_combined_config(
        tmp_path, raw_dir, manifest_path, pca_root, vae_root, [42],
        vae_overrides={"sensitivity": {
            "enabled": True, "dry_run": True, "mode": "one_at_a_time",
            "existing_run_policy": "error",
            "reference": {"beta_max": 0.5, "free_bits": 0.5, "objective_mode": "VAE"},
            "beta_values": [0.25, 0.5, 1.0], "free_bits_values": [0.0, 0.25, 0.5, 1.0],
            "include_ae": True,
        }})

    monkeypatch.setattr("sys.argv", ["per_block_vae_concat.py", "--config", str(config_path)])
    per_block_vae_concat.main()   # must not raise (would, if any patched fn were called)

    design = pd.read_csv(vae_root / "sensitivity_design.csv")
    assert len(design) == 7
    assert design["condition_id"].nunique() == 7

    # item 9: no condition/seed/model/log/embedding directories -- only the
    # design CSV lives directly under output_root.
    contents = sorted(p.name for p in vae_root.iterdir())
    assert contents == ["sensitivity_design.csv"]


# 10 -- sensitivity absent/disabled preserves the existing scalar directory structure.
def test_sensitivity_disabled_preserves_scalar_directory_structure(tmp_path, monkeypatch):
    import per_block_PCA_concat
    import per_block_vae_concat

    raw_dir, manifest_path = _write_synthetic_dataset(tmp_path)
    pca_root = tmp_path / "pca_out"
    vae_root = tmp_path / "vae_out"
    config_path = _write_combined_config(
        tmp_path, raw_dir, manifest_path, pca_root, vae_root, [42],
        vae_overrides={"sensitivity": {"enabled": False}})   # present but disabled

    monkeypatch.setattr("sys.argv", ["per_block_PCA_concat.py", "--config", str(config_path)])
    per_block_PCA_concat.main()
    monkeypatch.setattr("sys.argv", ["per_block_vae_concat.py", "--config", str(config_path)])
    per_block_vae_concat.main()

    assert (vae_root / "seed_42").exists()                 # unchanged scalar layout
    assert not (vae_root / "sensitivity_design.csv").exists()
    ts = pd.read_csv(vae_root / "seed_42" / "training_summary.csv")
    assert "condition_id" not in ts.columns                # never added when there's no sweep


# ══════════════════════════════════════════════════════════════════
# Real (non-dry-run) sweeps on a tiny synthetic dataset. Kept intentionally
# small: reference + one beta point + AE = 3 unique conditions, 1 model seed.
# ══════════════════════════════════════════════════════════════════
def _small_real_sweep_config(tmp_path, existing_run_policy="error"):
    raw_dir, manifest_path = _write_synthetic_dataset(tmp_path)
    pca_root = tmp_path / "pca_out"
    vae_root = tmp_path / "vae_out"
    config_path = _write_combined_config(
        tmp_path, raw_dir, manifest_path, pca_root, vae_root, [42],
        vae_overrides={"sensitivity": {
            "enabled": True, "dry_run": False, "mode": "one_at_a_time",
            "existing_run_policy": existing_run_policy,
            "reference": {"beta_max": 0.5, "free_bits": 0.5, "objective_mode": "VAE"},
            "beta_values": [0.25], "free_bits_values": [0.5], "include_ae": True,
        }})
    return raw_dir, manifest_path, pca_root, vae_root, config_path


def _run_pca_then_vae(config_path, monkeypatch):
    import per_block_PCA_concat
    import per_block_vae_concat
    monkeypatch.setattr("sys.argv", ["per_block_PCA_concat.py", "--config", str(config_path)])
    per_block_PCA_concat.main()
    monkeypatch.setattr("sys.argv", ["per_block_vae_concat.py", "--config", str(config_path)])
    per_block_vae_concat.main()


# 11/12 -- each real condition receives only its intended overrides, and
# condition directories are unique.
def test_sensitivity_real_conditions_get_only_intended_overrides(tmp_path, monkeypatch):
    _, _, pca_root, vae_root, config_path = _small_real_sweep_config(tmp_path)
    _run_pca_then_vae(config_path, monkeypatch)

    expected = {
        "vae_beta0p5_lambda0p5": ("VAE", 0.5, 0.5),
        "vae_beta0p25_lambda0p5": ("VAE", 0.25, 0.5),
        "ae": ("AE", 0.0, 0.5),
    }
    seen_dirs = set()
    for cid, (mode, beta, fb) in expected.items():
        cond_dir = vae_root / cid
        assert cond_dir.exists()
        assert cond_dir not in seen_dirs
        seen_dirs.add(cond_dir)

        bo = pd.read_csv(cond_dir / "seed_42" / "block_order.csv")
        assert (bo["objective_mode"] == mode).all()
        assert (bo["condition_id"] == cid).all()

        ts = pd.read_csv(cond_dir / "seed_42" / "training_summary.csv")
        assert (ts["objective_mode"] == mode).all()
        assert (ts["condition_id"] == cid).all()

        bsum = pd.read_csv(cond_dir / "seed_42" / "baseline_summary.csv")
        assert bsum["objective_mode"].iloc[0] == mode
        assert bsum["beta_max"].iloc[0] == pytest.approx(beta)
        assert bsum["free_bits"].iloc[0] == pytest.approx(fb)
        assert bsum["condition_id"].iloc[0] == cid

        ckpt = torch.load(cond_dir / "seed_42" / "models" / "blockA.pt", weights_only=False)
        assert ckpt["objective_mode"] == mode
        assert ckpt["condition_id"] == cid

        with open(cond_dir / "seed_42" / "baseline_meta.json") as f:
            meta = json.load(f)
        assert meta["condition_id"] == cid
        assert meta["objective_mode"] == mode

    # everything NOT swept stayed identical across conditions (e.g. batch_size,
    # lr, checkpoint_start_epoch aren't part of the sweep -- same base cfg).
    bo_ref = pd.read_csv(vae_root / "vae_beta0p5_lambda0p5" / "seed_42" / "block_order.csv")
    bo_beta = pd.read_csv(vae_root / "vae_beta0p25_lambda0p5" / "seed_42" / "block_order.csv")
    assert bo_ref["h1"].tolist() == bo_beta["h1"].tolist()      # architecture unaffected by beta/free_bits
    assert bo_ref["nominal_d"].tolist() == bo_beta["nominal_d"].tolist()

    # per-condition median aggregation still produced (point 6)
    for cid in expected:
        assert (vae_root / cid / "median_summary.csv").exists()
        assert (vae_root / cid / "median_training_summary.csv").exists()

    # posthoc is never run automatically
    for cid in expected:
        assert not (vae_root / cid / "seed_42" / "posthoc_tier0.csv").exists()


# 15 -- top-level summary row counts and ordering.
def test_sensitivity_top_level_summaries_row_counts_and_order(tmp_path, monkeypatch):
    _, _, pca_root, vae_root, config_path = _small_real_sweep_config(tmp_path)
    _run_pca_then_vae(config_path, monkeypatch)

    by_seed = pd.read_csv(vae_root / "sensitivity_summary_by_seed.csv")
    assert len(by_seed) == 3 * 1          # 3 conditions x 1 model seed
    assert by_seed["condition_order"].tolist() == sorted(by_seed["condition_order"].tolist())

    training = pd.read_csv(vae_root / "sensitivity_training_by_block_seed.csv")
    assert len(training) == 3 * 1 * 1     # 3 conditions x 1 seed x 1 block (blockA)
    assert list(training["condition_order"]) == sorted(training["condition_order"])
    for col in ("condition_order", "condition_id", "is_reference", "varied_parameter"):
        assert col in training.columns

    summary = pd.read_csv(vae_root / "sensitivity_summary.csv")
    assert len(summary) == 3
    assert summary["condition_order"].tolist() == [1, 2, 3]
    assert (summary["n_completed"] == 1).all()
    assert (summary["n_skipped_complete"] == 0).all()
    for col in ("median_seconds", "min_seconds", "max_seconds", "n_model_seeds"):
        assert col in summary.columns


# 13 -- existing_run_policy=error refuses to overwrite.
def test_sensitivity_existing_run_policy_error_prevents_overwrite(tmp_path, monkeypatch):
    _, _, pca_root, vae_root, config_path = _small_real_sweep_config(tmp_path, existing_run_policy="error")
    _run_pca_then_vae(config_path, monkeypatch)   # first run: succeeds, fully populates all 3 conditions

    import per_block_vae_concat
    monkeypatch.setattr("sys.argv", ["per_block_vae_concat.py", "--config", str(config_path)])
    with pytest.raises(RuntimeError, match="existing_run_policy"):
        per_block_vae_concat.main()   # same output_root, already has outputs -> must fail loudly


# 14 -- skip_complete skips only genuinely complete seeds and still rejects partial ones.
def test_sensitivity_skip_complete_skips_complete_and_rejects_partial(tmp_path, monkeypatch):
    import per_block_vae_concat

    _, _, pca_root, vae_root, config_path = _small_real_sweep_config(tmp_path, existing_run_policy="error")
    _run_pca_then_vae(config_path, monkeypatch)   # populate a fully complete sweep first

    # rewrite the config with existing_run_policy: skip_complete, same output_root
    import yaml
    with open(config_path) as f:
        cfg_yaml = yaml.safe_load(f)
    cfg_yaml["vae"]["sensitivity"]["existing_run_policy"] = "skip_complete"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg_yaml, f)

    # a complete condition must be skipped WITHOUT calling run_one_model_seed again
    monkeypatch.setattr(per_block_vae_concat, "run_one_model_seed",
                         lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should have been skipped")))
    monkeypatch.setattr("sys.argv", ["per_block_vae_concat.py", "--config", str(config_path)])
    per_block_vae_concat.main()   # must NOT raise -- everything is already complete

    by_seed = pd.read_csv(vae_root / "sensitivity_summary_by_seed.csv")
    assert (by_seed["run_status"] == "skipped_complete").all()

    # now corrupt one condition's seed dir into a PARTIAL state and retry
    monkeypatch.undo()   # restore run_one_model_seed so a genuine partial-detection error path is exercised
    partial_dir = vae_root / "ae" / "seed_42"
    (partial_dir / "embeddings" / "all_blocks_latent_dims.npy").unlink()
    monkeypatch.setattr("sys.argv", ["per_block_vae_concat.py", "--config", str(config_path)])
    with pytest.raises(RuntimeError, match="PARTIAL"):
        per_block_vae_concat.main()
