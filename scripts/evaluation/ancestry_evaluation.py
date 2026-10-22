#!/usr/bin/env python3
"""
ancestry_evaluation.py — model-agnostic ancestry-signal evaluation.

Config-driven (see config/ancestry_evaluation.yaml); the only CLI surface
is an optional config-path argument. Runs identically for baseline_pca,
baseline_vae, and main_model -- only the `models:` section of the config
differs; a model whose embedding_file/subject_id_file are missing is
skipped, not an error.

Invariant (not runtime-checked, true by construction): this script only
ever reads already-trained embeddings from disk. No ancestry information
is available to, or used by, representation training.

Every join between embeddings, ancestry PCs, and the split is done by
subject ID, never by row position -- with one documented exception:
`cluster_file` (as written by scripts/baselines/*.py) carries no ID
column, so cluster rows are assumed positionally aligned to that model's
subject_id_file (true for per_block_PCA_concat.py / per_block_vae_concat.py
cluster() output); a row-count mismatch is a hard error.

Components (each independently toggleable in config `components:`):
  1. dimension-level  -- StandardScaler + PCA fit on TRAIN embedding only,
                          transform train/test, Pearson r vs ancestry PCs.
  2. embedding-level  -- Ridge + GradientBoostingRegressor on the ORIGINAL
                          embedding (not embedding PCs) predicting ancestry
                          PCs, held-out R2/MAE/RMSE. Same train/test subjects
                          (the shared split file) for every model.
  3. cluster-level    -- (A) multinomial logistic regression: ancestry PCs
                          -> cluster, held-out accuracy/balanced-accuracy/
                          macro-F1 vs a majority-class baseline.
                          (B) eta-squared (one-way ANOVA) per ancestry PC
                          across clusters, plus cluster-wise means.

Outputs, under <output_root>/<model_name>/:
  dimension_correlations.csv, dimension_summary.csv,
  ancestry_prediction_metrics.csv, ancestry_predictions.csv,
  cluster_prediction_metrics.csv, cluster_ancestry_effects.csv,
  summary.json, plots/*.png

The shared split file (<split.file>, default
results/GACRS/ancestry_evaluation/evaluation_split.csv) is created once,
from the intersection of ancestry-PC subject IDs and the subject IDs of
every model present on disk at first-run time, and is never regenerated
on later runs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import f_oneway, pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_CONFIG = "config/ancestry_evaluation.yaml"


# ──────────────────────────────────────────────────────────────────
# safeguards
# ──────────────────────────────────────────────────────────────────
def check_duplicates(ids, label):
    idx = pd.Index(ids)
    dup = idx[idx.duplicated()]
    if len(dup):
        raise ValueError(f"{label}: {len(dup)} duplicate subject ID(s), e.g. {list(dup[:5])}")


def check_finite(arr, label):
    arr = np.asarray(arr, dtype=float)
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{label}: {n_bad} non-finite value(s) found")


# ──────────────────────────────────────────────────────────────────
# loaders
# ──────────────────────────────────────────────────────────────────
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_ancestry(cfg):
    acfg = cfg["ancestry"]
    path = acfg["pc_file"]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"ancestry pc_file not found: {path} "
            "(set ancestry.pc_file in the config to your ancestry-PC table)"
        )
    df = pd.read_csv(path, sep=r"\s+")
    id_col, pc_cols = acfg["id_column"], acfg["pc_columns"]
    missing = [c for c in [id_col, *pc_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing column(s) {missing} (have {list(df.columns)})")
    df[id_col] = df[id_col].astype(str)
    check_duplicates(df[id_col], f"{path} ({id_col})")
    check_finite(df[pc_cols].to_numpy(), f"{path} ancestry PCs")
    return df.set_index(id_col)[pc_cols]


def load_model_embedding(mcfg):
    ids = pd.read_csv(mcfg["subject_id_file"]).iloc[:, 0].astype(str).to_numpy()
    embedding = np.load(mcfg["embedding_file"])
    if embedding.ndim != 2:
        raise ValueError(f"{mcfg['embedding_file']}: expected 2D embedding, got shape {embedding.shape}")
    if len(ids) != embedding.shape[0]:
        raise ValueError(
            f"{mcfg['subject_id_file']} has {len(ids)} IDs but "
            f"{mcfg['embedding_file']} has {embedding.shape[0]} rows"
        )
    return ids, embedding


def load_cluster(mcfg, ids):
    path = Path(mcfg["cluster_file"])
    df = pd.read_csv(path)
    col = mcfg["cluster_column"]
    if col not in df.columns:
        raise KeyError(f"{path}: column {col!r} not found (have {list(df.columns)})")
    if len(df) != len(ids):
        raise ValueError(
            f"{path} has {len(df)} rows but subject_id_file has {len(ids)} rows -- "
            "cluster_file carries no ID column and is assumed positionally aligned "
            "to subject_id_file (repo convention); row counts must match."
        )
    return pd.Series(df[col].to_numpy(), index=ids, name="cluster")


# ──────────────────────────────────────────────────────────────────
# shared split
# ──────────────────────────────────────────────────────────────────
def build_or_load_split(cfg, ancestry_df, present_models, seed):
    split_path = Path(cfg["split"]["file"])
    if split_path.exists():
        print(f"[split] reusing existing split file: {split_path}")
        return pd.read_csv(split_path, dtype={"subject_id": str})

    common = set(ancestry_df.index)
    print(f"[split] ancestry subjects: {len(common)}")
    for name, mcfg in present_models.items():
        ids = pd.read_csv(mcfg["subject_id_file"]).iloc[:, 0].astype(str)
        before = len(common)
        common &= set(ids)
        print(f"[split] intersect with {name}: {before} -> {len(common)}")

    common = sorted(common)
    if len(common) < 10:
        raise RuntimeError(f"Subject intersection too small to build a split (n={len(common)})")

    test_frac = cfg["split"].get("test_frac", 0.20)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(common))
    n_test = int(len(common) * test_frac)
    test_idx = set(perm[:n_test].tolist())
    rows = [
        {"subject_id": sid, "split": "test" if i in test_idx else "train"}
        for i, sid in enumerate(common)
    ]
    df = pd.DataFrame(rows)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_path, index=False)
    n_tr = int((df["split"] == "train").sum())
    n_te = int((df["split"] == "test").sum())
    print(f"[split] created split (seed={seed}): train={n_tr} test={n_te} -> {split_path}")
    return df


def split_ids_for_model(split_df, available_ids, ancestry_index):
    available = set(available_ids) & set(ancestry_index)
    train_all = split_df.loc[split_df["split"] == "train", "subject_id"]
    test_all = split_df.loc[split_df["split"] == "test", "subject_id"]
    train_ids = [s for s in train_all if s in available]
    test_ids = [s for s in test_all if s in available]
    dropped = (len(train_all) - len(train_ids)) + (len(test_all) - len(test_ids))
    if dropped:
        print(f"[warn] {dropped} split subject(s) missing from this model/ancestry -- dropped")
    assert set(train_ids).isdisjoint(test_ids), "train/test overlap detected"
    return train_ids, test_ids


# ──────────────────────────────────────────────────────────────────
# 1. dimension-level test
# ──────────────────────────────────────────────────────────────────
def dimension_level(name, embedding, ids, ancestry_df, train_ids, test_ids, out_dir, cfg, seed):
    id_to_row = {sid: i for i, sid in enumerate(ids)}
    X_train = embedding[[id_to_row[s] for s in train_ids]]
    X_test = embedding[[id_to_row[s] for s in test_ids]]

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    n_comp = min(cfg.get("embedding_pca", {}).get("n_components", 10), Xtr.shape[1], len(train_ids) - 1)
    pca = PCA(n_components=n_comp, random_state=seed).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)

    all_ids = list(train_ids) + list(test_ids)
    Z_all = np.vstack([Ztr, Zte])
    anc = ancestry_df.loc[all_ids].to_numpy()
    pc_names = list(ancestry_df.columns)
    embed_pc_names = [f"embed_pc{j + 1}" for j in range(n_comp)]

    corr = np.empty((n_comp, len(pc_names)))
    for j in range(n_comp):
        for k in range(len(pc_names)):
            corr[j, k] = pearsonr(Z_all[:, j], anc[:, k])[0]

    corr_df = pd.DataFrame(corr, index=embed_pc_names, columns=pc_names)
    corr_df.to_csv(out_dir / "dimension_correlations.csv")

    summary_rows = []
    for k, pcname in enumerate(pc_names):
        j_best = int(np.argmax(np.abs(corr[:, k])))
        summary_rows.append(
            {
                "ancestry_pc": pcname,
                "best_embedding_pc": embed_pc_names[j_best],
                "r": float(corr[j_best, k]),
                "max_abs_r": float(abs(corr[j_best, k])),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "dimension_summary.csv", index=False)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(4, 0.5 * len(pc_names) + 2), max(4, 0.3 * n_comp + 2)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(pc_names)))
    ax.set_xticklabels(pc_names, rotation=45, ha="right")
    ax.set_yticks(range(n_comp))
    ax.set_yticklabels(embed_pc_names)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"{name}: embedding PC vs ancestry PC correlation")
    fig.tight_layout()
    fig.savefig(plot_dir / "dimension_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    return {
        "n_embedding_pcs": n_comp,
        "mean_max_abs_r": float(summary_df["max_abs_r"].mean()),
        "max_abs_r": float(summary_df["max_abs_r"].max()),
    }


# ──────────────────────────────────────────────────────────────────
# 2. embedding-level test
# ──────────────────────────────────────────────────────────────────
def embedding_level(name, embedding, ids, ancestry_df, train_ids, test_ids, out_dir, cfg, seed):
    id_to_row = {sid: i for i, sid in enumerate(ids)}
    X_train = embedding[[id_to_row[s] for s in train_ids]]
    X_test = embedding[[id_to_row[s] for s in test_ids]]

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    pc_names = list(ancestry_df.columns)
    y_train_all = ancestry_df.loc[train_ids].to_numpy()
    y_test_all = ancestry_df.loc[test_ids].to_numpy()

    probes = {
        "ridge": lambda: Ridge(random_state=seed),
        "gbr": lambda: GradientBoostingRegressor(random_state=seed),
    }

    metric_rows, pred_rows = [], []
    for k, pcname in enumerate(pc_names):
        y_train, y_test = y_train_all[:, k], y_test_all[:, k]
        for probe_name, make in probes.items():
            model = make()
            model.fit(Xtr, y_train)
            pred = model.predict(Xte)
            metric_rows.append(
                {
                    "ancestry_pc": pcname,
                    "model": probe_name,
                    "r2": float(r2_score(y_test, pred)),
                    "mae": float(mean_absolute_error(y_test, pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "n_train": len(train_ids),
                    "n_test": len(test_ids),
                }
            )
            for sid, yt, yp in zip(test_ids, y_test, pred):
                pred_rows.append(
                    {
                        "subject_id": sid,
                        "ancestry_pc": pcname,
                        "model": probe_name,
                        "y_true": float(yt),
                        "y_pred": float(yp),
                    }
                )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "ancestry_prediction_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out_dir / "ancestry_predictions.csv", index=False)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(pc_names) + 2), 4))
    width = 0.35
    xs = np.arange(len(pc_names))
    for i, probe_name in enumerate(probes):
        vals = [
            metrics_df.loc[(metrics_df["ancestry_pc"] == p) & (metrics_df["model"] == probe_name), "r2"].iloc[0]
            for p in pc_names
        ]
        ax.bar(xs + (i - 0.5) * width, vals, width, label=probe_name)
    ax.set_xticks(xs)
    ax.set_xticklabels(pc_names, rotation=45, ha="right")
    ax.set_ylabel("held-out R2")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{name}: ancestry PC prediction from embedding")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "ancestry_prediction_r2.png", dpi=150)
    plt.close(fig)

    return {
        "mean_r2_ridge": float(metrics_df.loc[metrics_df["model"] == "ridge", "r2"].mean()),
        "mean_r2_gbr": float(metrics_df.loc[metrics_df["model"] == "gbr", "r2"].mean()),
        "max_r2": float(metrics_df["r2"].max()),
    }


# ──────────────────────────────────────────────────────────────────
# 3. cluster-level test
# ──────────────────────────────────────────────────────────────────
def cluster_level(name, cluster_series, ancestry_df, train_ids, test_ids, out_dir, cfg, seed):
    valid = set(cluster_series.dropna().index)
    tr = [s for s in train_ids if s in valid]
    te = [s for s in test_ids if s in valid]
    dropped = (len(train_ids) - len(tr)) + (len(test_ids) - len(te))
    if dropped:
        print(f"[warn] {name}: dropped {dropped} subject(s) missing a cluster label")
    assert set(tr).isdisjoint(te), "train/test overlap detected"

    pc_names = list(ancestry_df.columns)
    X_train, X_test = ancestry_df.loc[tr].to_numpy(), ancestry_df.loc[te].to_numpy()
    y_train, y_test = cluster_series.loc[tr].to_numpy(), cluster_series.loc[te].to_numpy()

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, random_state=seed)  # multinomial: automatic for lbfgs + >2 classes
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    majority = pd.Series(y_train).mode().iloc[0]
    pred_baseline = np.full_like(y_test, majority)

    def _metrics(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }

    rows = [
        {
            "method": "logistic_regression",
            "n_train": len(tr),
            "n_test": len(te),
            "n_classes": len(set(y_train)),
            **_metrics(y_test, pred),
        },
        {
            "method": "majority_baseline",
            "n_train": len(tr),
            "n_test": len(te),
            "n_classes": len(set(y_train)),
            **_metrics(y_test, pred_baseline),
        },
    ]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "cluster_prediction_metrics.csv", index=False)

    # Part B: eta-squared + cluster-wise means, over all subjects with a cluster label
    all_ids = tr + te
    anc_all = ancestry_df.loc[all_ids]
    clusters_all = cluster_series.loc[all_ids]
    cluster_labels = sorted(clusters_all.unique())

    effect_rows = []
    for pcname in pc_names:
        vals = anc_all[pcname]
        groups = [vals[clusters_all == c].to_numpy() for c in cluster_labels]
        groups = [g for g in groups if len(g) > 0]
        f_stat, p_value = f_oneway(*groups) if len(groups) > 1 else (float("nan"), float("nan"))
        grand_mean = vals.mean()
        ss_total = float(((vals - grand_mean) ** 2).sum())
        ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups))
        eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")
        for c in cluster_labels:
            cvals = vals[clusters_all == c]
            effect_rows.append(
                {
                    "ancestry_pc": pcname,
                    "cluster": c,
                    "cluster_n": int(len(cvals)),
                    "cluster_mean": float(cvals.mean()) if len(cvals) else float("nan"),
                    "eta_squared": float(eta_sq),
                    "f_stat": float(f_stat),
                    "p_value": float(p_value),
                }
            )
    effects_df = pd.DataFrame(effect_rows)
    effects_df.to_csv(out_dir / "cluster_ancestry_effects.csv", index=False)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    if len(pc_names) >= 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        for c in cluster_labels:
            m = clusters_all == c
            ax.scatter(
                anc_all.loc[m, pc_names[0]], anc_all.loc[m, pc_names[1]],
                s=8, alpha=0.7, label=f"cluster {c}",
            )
        ax.set_xlabel(pc_names[0])
        ax.set_ylabel(pc_names[1])
        ax.set_title(f"{name}: {pc_names[0]}/{pc_names[1]} by cluster")
        ax.legend(fontsize=7, markerscale=2)
        fig.tight_layout()
        fig.savefig(plot_dir / "ancestry_pc1_pc2_by_cluster.png", dpi=150)
        plt.close(fig)

    eta_by_pc = effects_df.groupby("ancestry_pc")["eta_squared"].first()
    return {
        "logistic_regression": {k: v for k, v in rows[0].items() if isinstance(v, (int, float))},
        "majority_baseline": {k: v for k, v in rows[1].items() if isinstance(v, (int, float))},
        "eta_squared_mean": float(eta_by_pc.mean()),
        "eta_squared_max": float(eta_by_pc.max()),
    }


# ──────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Model-agnostic ancestry evaluation (config-driven).")
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help=f"config YAML (default: {DEFAULT_CONFIG})")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    np.random.seed(seed)

    out_root = Path(cfg["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[config] {args.config}")
    ancestry_df = load_ancestry(cfg)
    print(f"[ancestry] {len(ancestry_df)} subjects, PCs={list(ancestry_df.columns)}")

    present_models = {}
    for mname, mcfg in cfg["models"].items():
        emb_ok = Path(mcfg["embedding_file"]).exists()
        id_ok = Path(mcfg["subject_id_file"]).exists()
        if emb_ok and id_ok:
            present_models[mname] = mcfg
        else:
            print(f"[skip] {mname}: embedding_file exists={emb_ok}, subject_id_file exists={id_ok}")

    if not present_models:
        raise RuntimeError("No configured model has both embedding_file and subject_id_file on disk.")

    split_df = build_or_load_split(cfg, ancestry_df, present_models, seed)
    components = cfg.get("components", {})

    for mname, mcfg in present_models.items():
        print(f"\n══════ {mname} ══════")
        ids, embedding = load_model_embedding(mcfg)
        check_duplicates(ids, f"{mname} subject_id_file")
        check_finite(embedding, f"{mname} embedding")

        train_ids, test_ids = split_ids_for_model(split_df, ids, ancestry_df.index)
        print(f"[{mname}] n_subjects={len(ids)} train={len(train_ids)} test={len(test_ids)}")

        out_dir = out_root / mname
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": mname,
            "seed": seed,
            "n_subjects_total": int(len(ids)),
            "n_train": len(train_ids),
            "n_test": len(test_ids),
        }

        if components.get("dimension_level", True):
            summary["dimension_level"] = dimension_level(
                mname, embedding, ids, ancestry_df, train_ids, test_ids, out_dir, cfg, seed
            )

        if components.get("embedding_level", True):
            summary["embedding_level"] = embedding_level(
                mname, embedding, ids, ancestry_df, train_ids, test_ids, out_dir, cfg, seed
            )

        if components.get("cluster_level", True) and mcfg.get("cluster_file"):
            cluster_path = Path(mcfg["cluster_file"])
            if cluster_path.exists():
                cluster_series = load_cluster(mcfg, ids)
                summary["cluster_level"] = cluster_level(
                    mname, cluster_series, ancestry_df, train_ids, test_ids, out_dir, cfg, seed
                )
            else:
                print(f"[skip] {mname}: cluster_file not found ({cluster_path})")
        else:
            print(f"[skip] {mname}: cluster-level test disabled or no cluster_file configured")

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] {mname} -> {out_dir}")


if __name__ == "__main__":
    main()
