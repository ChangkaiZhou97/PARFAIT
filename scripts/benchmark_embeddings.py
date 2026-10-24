#!/usr/bin/env python3
"""Evaluate and plot DLPFC per-sample ARI/NMI for multiple embeddings.

This script is intentionally separate from the PARFAIT trainer. It asks a
simple benchmarking question:

    for each DLPFC slice and each embedding, how well do Leiden clusters
    recover the manual layer labels?

It also can append existing PARFAIT best-grid metrics from
``best_metrics.csv`` files so raw embeddings and model outputs can be viewed
in one table/plot. Those rows are marked as aggregate PARFAIT metrics rather
than raw source embeddings.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


DLPFC_SIDS = [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
]

DEFAULT_EMBEDDINGS = [
    "hvg_pca",
    "spatial_xy",
    "scgpt",
    "loki_text",
    "loki_image",
    "phikon_v2",
    "uni2_h",
    "nicheformer",
    "stpath",
]

DISPLAY_NAMES = {
    "hvg_pca": "HVG-PCA50",
    "spatial_xy": "spatial",
    "BayesSpace": "BayesSpace",
    "SpaGCN": "SpaGCN",
    "STAGATE": "STAGATE",
    "scgpt": "scGPT-spatial",
    "loki_text": "loki-text",
    "loki_image": "loki-image",
    "loki_text_image_concat": "loki-image+text",
    "nicheformer": "nicheformer",
    "phikon_v2": "phikon-v2",
    "uni2_h": "uni-v2",
    "stpath": "stpath",
    "PARFAIT": "PARFAIT",
    "PARFAIT_ARI": "PARFAIT-ARI",
    "PARFAIT_NMI": "PARFAIT-NMI",
}

BAR_COLORS = {
    "hvg_pca": "#6f6f6f",
    "spatial_xy": "#9e9e9e",
    "BayesSpace": "#4daf4a",
    "SpaGCN": "#984ea3",
    "STAGATE": "#a65628",
    "scgpt": "#1b9e77",
    "loki_text": "#d95f02",
    "loki_image": "#7570b3",
    "loki_text_image_concat": "#a6761d",
    "nicheformer": "#e7298a",
    "phikon_v2": "#66a61e",
    "uni2_h": "#e6ab02",
    "stpath": "#1f78b4",
    "PARFAIT": "#cc0000",
    "PARFAIT_ARI": "#cc0000",
    "PARFAIT_NMI": "#cc0000",
}

DEFAULT_FM_SPECS = {
    "scgpt": {
        "folder_template": "DLPFC_{sid}_scgpt_with_spatial",
        "obsm_key": "X_scgpt_spatial",
        "family": "transcriptome_fm",
        "embedding_type": "foundation_model_transcriptome",
    },
    "loki_text": {
        "folder_template": "DLPFC_{sid}_loki_text",
        "obsm_key": "X_loki_text",
        "family": "gene_text_fm",
        "embedding_type": "foundation_model_gene_text",
    },
    "loki_image": {
        "folder_template": "DLPFC_{sid}_loki_image_10x",
        "obsm_key": "X_loki_image",
        "family": "histology_fm",
        "embedding_type": "foundation_model_histology_image",
    },
    "phikon_v2": {
        "folder_template": "DLPFC_{sid}_phikon_v2",
        "obsm_key": "X_path_phikon_v2",
        "family": "pathology_fm",
        "embedding_type": "foundation_model_pathology_image",
    },
    "uni2_h": {
        "folder_template": "DLPFC_{sid}_uni2-h",
        "obsm_key": "X_path_uni2_h",
        "family": "pathology_fm",
        "embedding_type": "foundation_model_pathology_image",
    },
    "nicheformer": {
        "folder_template": "nicheformer_{sid}_test",
        "obsm_key": "X_nicheformer",
        "family": "transcriptome_fm",
        "embedding_type": "foundation_model_transcriptome",
    },
    "stpath": {
        "folder_template": "DLPFC_{sid}_stpath",
        "obsm_key": "X_stpath",
        "family": "joint_fm",
        "embedding_type": "foundation_model_joint_stpath",
    },
}

TRADITIONAL_BASELINE_SPECS = {
    "bayesspace": {
        "embedding": "BayesSpace",
        "dirs": ["bayesspace"],
        "metrics_file": "dlpfc_bayesspace_metrics.csv",
    },
    "spagcn": {
        "embedding": "SpaGCN",
        "dirs": ["spagcn"],
        "metrics_file": "dlpfc_spagcn_metrics.csv",
    },
    "stagate": {
        "embedding": "STAGATE",
        "dirs": ["stagate_mclust_gpu", "stagate_mclust", "stagate"],
        "metrics_file": "dlpfc_stagate_metrics.csv",
    },
}

INVALID_LABELS = {"", "nan", "none", "na", "null", "unavailable", "unknown"}


@dataclass
class EvalRow:
    sample: str
    embedding: str
    family: str
    embedding_type: str
    status: str
    ARI: float = math.nan
    NMI: float = math.nan
    n_clusters: int = 0
    n_labels: int = 0
    n_spots: int = 0
    n_valid_labels: int = 0
    label_key: str = ""
    cluster_method: str = ""
    cluster_error: str = ""
    leiden_resolution: float = math.nan
    leiden_n_neighbors: int = 0
    cluster_pca_dim: int = 0
    seed: int = 0
    n_runs: int = 0
    path: str = ""
    key: str = ""
    shape: str = ""
    order_mode: str = ""
    selection_metric: str = ""
    t: float = math.nan
    s: float = math.nan
    error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default=".")
    p.add_argument("--data_root", type=str, default="data/DLPFC_local")
    p.add_argument("--embedding_root", type=str, default="output")
    p.add_argument("--samples", type=str, default="all")
    p.add_argument("--embedding_names", type=str, default=",".join(DEFAULT_EMBEDDINGS))
    p.add_argument("--embedding_specs_json", type=str, default="")
    p.add_argument("--label_key", type=str, default="auto")
    p.add_argument("--spatial_key", type=str, default="spatial")
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument(
        "--cluster_method",
        choices=["leiden", "kmeans"],
        default="leiden",
        help="Leiden matches the PARFAIT metric. Use kmeans for a lightweight local smoke test.",
    )
    p.add_argument("--leiden_n_neighbors", type=int, default=15)
    p.add_argument("--leiden_resolution", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--cluster_pca_dim",
        type=int,
        default=50,
        help="Project high-dimensional embeddings before clustering. Set 0 to disable.",
    )
    p.add_argument(
        "--include_parfait_best",
        dest="include_parfait_best",
        action="store_true",
        default=True,
        help="Append PARFAIT best-grid metrics from --parfait_root. Enabled by default.",
    )
    p.add_argument(
        "--no_include_parfait_best",
        dest="include_parfait_best",
        action="store_false",
        help="Do not append PARFAIT best-grid metrics.",
    )
    p.add_argument(
        "--include_traditional_baselines",
        dest="include_traditional_baselines",
        action="store_true",
        default=True,
        help="Append precomputed BayesSpace/SpaGCN/STAGATE metrics if present.",
    )
    p.add_argument(
        "--no_include_traditional_baselines",
        dest="include_traditional_baselines",
        action="store_false",
        help="Do not append precomputed traditional baseline metrics.",
    )
    p.add_argument(
        "--traditional_baseline_root",
        type=str,
        default="figures/figure2/output/dlpfc_traditional_baselines",
        help="Root containing bayesspace/spagcn/stagate*/dlpfc_*_metrics.csv.",
    )
    p.add_argument(
        "--traditional_baseline_methods",
        type=str,
        default="bayesspace,spagcn,stagate",
        help="Comma-separated traditional baselines to append.",
    )
    p.add_argument(
        "--parfait_root",
        type=str,
        default="output/DLPFC_parfait_finegrid",
        help="Root containing <sid>/seed_*/best_metrics.csv.",
    )
    p.add_argument(
        "--parfait_name",
        type=str,
        default="PARFAIT",
        help="Embedding label used for the appended PARFAIT rows.",
    )
    p.add_argument(
        "--parfait_seed_mode",
        choices=["mean", "all", "seed_00"],
        default="mean",
        help=(
            "How to use seed_*/best_metrics.csv under --parfait_root. "
            "mean creates one row per sample averaged over seeds; all keeps one row per seed; "
            "seed_00 uses only seed_00."
        ),
    )
    p.add_argument(
        "--parfait_selection",
        choices=["metric_best", "ari", "nmi", "both"],
        default="metric_best",
        help=(
            "Which PARFAIT metric columns to append. metric_best creates one PARFAIT row with "
            "ARI=best_ARI and NMI=best_NMI; ari/nmi use the same selected grid point; "
            "both appends separate *_ARI and *_NMI rows."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="figures/figure2",
    )
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def resolve_samples(spec: str, data_root: Path) -> List[str]:
    if spec.lower() == "all":
        found = [sid for sid in DLPFC_SIDS if (data_root / sid / f"{sid}_with_gt.h5ad").exists()]
        return found or DLPFC_SIDS
    return split_csv(spec)


def resolve_label_key(adata: ad.AnnData, requested: str) -> str:
    if requested and requested != "auto":
        if requested not in adata.obs:
            raise KeyError(f"label_key={requested!r} not found. Available obs columns: {list(adata.obs.columns)}")
        return requested
    for key in ["ground_truth", "z", "layer", "annotation", "label", "domain"]:
        if key in adata.obs:
            return key
    raise KeyError(f"No label column found. Available obs columns: {list(adata.obs.columns)}")


def valid_label_mask(labels: Sequence[object]) -> np.ndarray:
    return np.array([str(x).strip().lower() not in INVALID_LABELS for x in labels], dtype=bool)


def sanitize_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {X.shape}")
    bad = ~np.isfinite(X)
    if bad.any():
        col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
        rows, cols = np.where(bad)
        X[rows, cols] = col_mean[cols]
    zero_var = np.nanstd(X, axis=0) == 0
    if zero_var.all():
        raise ValueError("All embedding dimensions have zero variance")
    if zero_var.any():
        X = X[:, ~zero_var]
    return X.astype(np.float32, copy=False)


def maybe_project_for_clustering(X: np.ndarray, dim: int, seed: int) -> np.ndarray:
    X = sanitize_matrix(X)
    if dim <= 0 or X.shape[1] <= dim:
        return StandardScaler().fit_transform(X).astype(np.float32)
    Xs = StandardScaler().fit_transform(X)
    n_components = min(dim, Xs.shape[0] - 1, Xs.shape[1])
    if n_components < 2:
        return Xs.astype(np.float32)
    return PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit_transform(Xs).astype(np.float32)


def cluster_labels(
    X: np.ndarray,
    true_labels: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, str, str]:
    Z = maybe_project_for_clustering(X, args.cluster_pca_dim, args.seed)
    if args.cluster_method == "leiden":
        try:
            import scanpy as sc

            tmp = ad.AnnData(X=np.zeros((Z.shape[0], 1), dtype=np.float32))
            tmp.obsm["X_emb"] = Z
            sc.pp.neighbors(
                tmp,
                use_rep="X_emb",
                n_neighbors=args.leiden_n_neighbors,
                random_state=args.seed,
            )
            sc.tl.leiden(
                tmp,
                resolution=args.leiden_resolution,
                random_state=args.seed,
                flavor="igraph",
                n_iterations=2,
                directed=False,
            )
            return tmp.obs["leiden"].astype(str).to_numpy(), "leiden", ""
        except Exception as e:
            # Leiden can fail on lightweight environments without igraph/leidenalg.
            # KMeans keeps the script usable for QC and local smoke tests.
            n_labels = max(2, int(np.unique(true_labels[valid]).size))
            pred = KMeans(n_clusters=n_labels, n_init=20, random_state=args.seed).fit_predict(Z)
            return pred.astype(str), "kmeans_fallback", repr(e)

    n_labels = max(2, int(np.unique(true_labels[valid]).size))
    pred = KMeans(n_clusters=n_labels, n_init=20, random_state=args.seed).fit_predict(Z)
    return pred.astype(str), "kmeans", ""


def score_embedding(
    sid: str,
    name: str,
    family: str,
    embedding_type: str,
    X: np.ndarray,
    labels: np.ndarray,
    label_key: str,
    args: argparse.Namespace,
    path: str = "",
    key: str = "",
    order_mode: str = "",
) -> EvalRow:
    valid = valid_label_mask(labels)
    n_valid = int(valid.sum())
    n_labels = int(np.unique(labels[valid].astype(str)).size) if n_valid else 0
    row = EvalRow(
        sample=sid,
        embedding=name,
        family=family,
        embedding_type=embedding_type,
        status="success",
        n_spots=int(len(labels)),
        n_valid_labels=n_valid,
        n_labels=n_labels,
        label_key=label_key,
        leiden_resolution=float(args.leiden_resolution),
        leiden_n_neighbors=int(args.leiden_n_neighbors),
        cluster_pca_dim=int(args.cluster_pca_dim),
        seed=int(args.seed),
        path=path,
        key=key,
        shape=f"{X.shape[0]}x{X.shape[1]}",
        order_mode=order_mode,
    )
    try:
        if n_valid < 2 or n_labels < 2:
            row.status = "no_labels"
            return row
        pred, method, cluster_error = cluster_labels(X, labels.astype(str), valid, args)
        row.cluster_method = method
        row.cluster_error = cluster_error
        row.n_clusters = int(np.unique(pred[valid]).size)
        row.ARI = float(adjusted_rand_score(labels[valid].astype(str), pred[valid]))
        row.NMI = float(normalized_mutual_info_score(labels[valid].astype(str), pred[valid]))
    except Exception as e:
        row.status = "failed"
        row.error = repr(e)
    return row


def compute_hvg_pca(adata: ad.AnnData, n_hvg: int, n_pcs: int, seed: int) -> np.ndarray:
    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)
        lib = np.asarray(X.sum(axis=1)).ravel()
        lib = np.where(lib > 0, lib, 1.0)
        Xn = sparse.diags((1e4 / lib).astype(np.float32)).dot(X)
        Xn.data = np.log1p(Xn.data)
        mean = np.asarray(Xn.mean(axis=0)).ravel()
        mean_sq = np.asarray(Xn.power(2).mean(axis=0)).ravel()
        var = mean_sq - mean**2
        top = np.argsort(var)[::-1][: min(n_hvg, Xn.shape[1])]
        Xt = Xn[:, top]
        n_components = min(n_pcs, Xt.shape[0] - 1, Xt.shape[1] - 1)
        if n_components < 2:
            return np.asarray(Xt.toarray(), dtype=np.float32)
        return TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(Xt).astype(np.float32)

    Xd = np.asarray(X, dtype=np.float32)
    lib = Xd.sum(axis=1)
    lib = np.where(lib > 0, lib, 1.0)
    Xn = np.log1p(Xd * (1e4 / lib[:, None]))
    var = np.var(Xn, axis=0)
    top = np.argsort(var)[::-1][: min(n_hvg, Xn.shape[1])]
    Xt = Xn[:, top]
    n_components = min(n_pcs, Xt.shape[0] - 1, Xt.shape[1])
    if n_components < 2:
        return Xt.astype(np.float32)
    return PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit_transform(Xt).astype(np.float32)


def spatial_xy(adata: ad.AnnData, spatial_key: str) -> np.ndarray:
    if spatial_key in adata.obsm:
        return np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    if {"array_row", "array_col"}.issubset(set(adata.obs.columns)):
        return adata.obs[["array_row", "array_col"]].to_numpy(dtype=np.float32)
    raise KeyError(f"Spatial key {spatial_key!r} not found and array_row/array_col unavailable")


def load_specs(spec_json: str) -> Dict[str, Dict[str, str]]:
    specs = dict(DEFAULT_FM_SPECS)
    if not spec_json:
        return specs
    path = Path(spec_json)
    if path.exists():
        obj = json.loads(path.read_text())
    else:
        obj = json.loads(spec_json)
    for item in obj:
        specs[item["name"]] = {
            "folder_template": item["folder_template"],
            "obsm_key": item.get("obsm_key", item.get("key", "")),
            "family": item.get("family", "foundation_model"),
            "embedding_type": item.get("embedding_type", "foundation_model"),
        }
    return specs


def resolve_embedding_path(repo: Path, embedding_root: Path, sid: str, folder_template: str) -> Path:
    folder = folder_template.format(sid=sid, dataset="DLPFC")
    p = Path(folder)
    if p.suffix == ".h5ad":
        return p if p.is_absolute() else repo / p
    return embedding_root / folder / "embeddings.h5ad"


def load_h5ad_embedding(
    path: Path,
    key: str,
    target_obs_names: Sequence[str],
) -> Tuple[np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    emb = ad.read_h5ad(path, backed="r")
    try:
        if key not in emb.obsm:
            raise KeyError(f"{key!r} not in obsm keys {list(emb.obsm.keys())}")
        X = np.asarray(emb.obsm[key], dtype=np.float32)
        emb_obs = list(map(str, emb.obs_names))
    finally:
        emb.file.close()

    target = list(map(str, target_obs_names))
    if emb_obs == target:
        return X, "exact_obs_order"
    if len(emb_obs) == len(target) and len(set(emb_obs)) != len(emb_obs):
        return X, "row_count_match_duplicate_obs_fallback"
    pos = {name: i for i, name in enumerate(emb_obs)}
    missing = [name for name in target if name not in pos]
    if missing:
        if len(emb_obs) == len(target):
            return X, f"row_count_match_fallback_missing_{len(missing)}_barcodes"
        raise ValueError(f"{len(missing)}/{len(target)} target barcodes missing from {path}")
    idx = np.array([pos[name] for name in target], dtype=int)
    return X[idx], "obs_name_reordered"


def make_concat(name: str, cache: Dict[str, Tuple[np.ndarray, str, str, str]], parts: Sequence[str]) -> Tuple[np.ndarray, str, str]:
    missing = [p for p in parts if p not in cache]
    if missing:
        raise FileNotFoundError(f"Cannot build {name}; missing parts: {missing}")
    X = np.concatenate([cache[p][0] for p in parts], axis=1).astype(np.float32)
    paths = "+".join(cache[p][1] for p in parts)
    keys = "+".join(cache[p][2] for p in parts)
    return X, paths, keys


def read_one_sample(
    repo: Path,
    data_root: Path,
    embedding_root: Path,
    sid: str,
    embedding_names: Sequence[str],
    specs: Dict[str, Dict[str, str]],
    args: argparse.Namespace,
) -> List[EvalRow]:
    raw_path = data_root / sid / f"{sid}_with_gt.h5ad"
    if not raw_path.exists():
        return [
            EvalRow(
                sample=sid,
                embedding="sample",
                family="input",
                embedding_type="input",
                status="failed",
                path=str(raw_path),
                error=f"Missing raw h5ad: {raw_path}",
            )
        ]

    adata = ad.read_h5ad(raw_path)
    label_key = resolve_label_key(adata, args.label_key)
    labels = adata.obs[label_key].astype(str).to_numpy()
    rows: List[EvalRow] = []
    cache: Dict[str, Tuple[np.ndarray, str, str, str]] = {}

    for name in embedding_names:
        try:
            if name == "hvg_pca":
                X = compute_hvg_pca(adata, args.n_hvg, args.n_pcs, args.seed)
                rows.append(
                    score_embedding(
                        sid,
                        name,
                        "expression_baseline",
                        "baseline_not_foundation_model",
                        X,
                        labels,
                        label_key,
                        args,
                        path=str(raw_path),
                        key=f"lognorm_hvg{args.n_hvg}_pca{args.n_pcs}",
                        order_mode="raw_obs_order",
                    )
                )
                cache[name] = (X, str(raw_path), f"lognorm_hvg{args.n_hvg}_pca{args.n_pcs}", "raw_obs_order")
                continue
            if name == "spatial_xy":
                X = spatial_xy(adata, args.spatial_key)
                rows.append(
                    score_embedding(
                        sid,
                        name,
                        "spatial_baseline",
                        "baseline_not_foundation_model",
                        X,
                        labels,
                        label_key,
                        args,
                        path=str(raw_path),
                        key=args.spatial_key,
                        order_mode="raw_obs_order",
                    )
                )
                cache[name] = (X, str(raw_path), args.spatial_key, "raw_obs_order")
                continue
            if name == "loki_text_image_concat":
                if "loki_text" not in cache:
                    spec = specs["loki_text"]
                    p = resolve_embedding_path(repo, embedding_root, sid, spec["folder_template"])
                    X_part, order_mode = load_h5ad_embedding(p, spec["obsm_key"], adata.obs_names)
                    cache["loki_text"] = (X_part, str(p), spec["obsm_key"], order_mode)
                if "loki_image" not in cache:
                    spec = specs["loki_image"]
                    p = resolve_embedding_path(repo, embedding_root, sid, spec["folder_template"])
                    X_part, order_mode = load_h5ad_embedding(p, spec["obsm_key"], adata.obs_names)
                    cache["loki_image"] = (X_part, str(p), spec["obsm_key"], order_mode)
                X, paths, keys = make_concat(name, cache, ["loki_text", "loki_image"])
                rows.append(
                    score_embedding(
                        sid,
                        name,
                        "multimodal_concat",
                        "foundation_model_concat",
                        X,
                        labels,
                        label_key,
                        args,
                        path=paths,
                        key=keys,
                        order_mode="concat_from_aligned_parts",
                    )
                )
                continue

            if name not in specs:
                raise KeyError(f"Unknown embedding {name!r}; known specs: {sorted(specs)}")
            spec = specs[name]
            p = resolve_embedding_path(repo, embedding_root, sid, spec["folder_template"])
            X, order_mode = load_h5ad_embedding(p, spec["obsm_key"], adata.obs_names)
            cache[name] = (X, str(p), spec["obsm_key"], order_mode)
            rows.append(
                score_embedding(
                    sid,
                    name,
                    spec.get("family", "foundation_model"),
                    spec.get("embedding_type", "foundation_model"),
                    X,
                    labels,
                    label_key,
                    args,
                    path=str(p),
                    key=spec["obsm_key"],
                    order_mode=order_mode,
                )
            )
        except Exception as e:
            spec = specs.get(name, {})
            rows.append(
                EvalRow(
                    sample=sid,
                    embedding=name,
                    family=spec.get("family", "unknown"),
                    embedding_type=spec.get("embedding_type", "unknown"),
                    status="failed",
                    n_spots=int(adata.n_obs),
                    n_valid_labels=int(valid_label_mask(labels).sum()),
                    label_key=label_key,
                    path=str(resolve_embedding_path(repo, embedding_root, sid, spec["folder_template"]))
                    if spec.get("folder_template")
                    else str(raw_path),
                    key=spec.get("obsm_key", ""),
                    error=repr(e),
                )
            )
    return rows


def parse_seed_from_path(path: Path) -> int:
    m = re.search(r"seed[_-]?(\d+)", str(path.parent))
    if m:
        return int(m.group(1))
    return -1


def parfait_embedding_name(base: str, selection: str, split_both: bool) -> str:
    if split_both:
        return f"{base}_{selection.upper()}"
    return base


def mean_numeric(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return math.nan
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def median_numeric(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return math.nan
    return float(pd.to_numeric(df[col], errors="coerce").median())


def rounded_mean_int(df: pd.DataFrame, col: str) -> int:
    val = mean_numeric(df, col)
    return int(round(val)) if math.isfinite(val) else 0


def parfait_metric_values(df: pd.DataFrame, selection: str) -> Dict[str, float]:
    if selection == "metric_best":
        return {
            "ARI": mean_numeric(df, "best_ARI"),
            "NMI": mean_numeric(df, "best_NMI"),
            "n_clusters": rounded_mean_int(df, "n_clusters_at_best_ARI"),
            "t": math.nan,
            "s": math.nan,
        }
    if selection == "ari":
        return {
            "ARI": mean_numeric(df, "best_ARI"),
            "NMI": mean_numeric(df, "NMI_at_best_ARI"),
            "n_clusters": rounded_mean_int(df, "n_clusters_at_best_ARI"),
            "t": median_numeric(df, "t_at_best_ARI"),
            "s": median_numeric(df, "s_at_best_ARI"),
        }
    if selection == "nmi":
        return {
            "ARI": mean_numeric(df, "ARI_at_best_NMI"),
            "NMI": mean_numeric(df, "best_NMI"),
            "n_clusters": rounded_mean_int(df, "n_clusters_at_best_NMI"),
            "t": median_numeric(df, "t_at_best_NMI"),
            "s": median_numeric(df, "s_at_best_NMI"),
        }
    raise ValueError(f"Unsupported PARFAIT selection: {selection}")


def make_parfait_row(
    sid: str,
    embedding: str,
    selection: str,
    df: pd.DataFrame,
    path: str,
    key: str,
    args: argparse.Namespace,
) -> EvalRow:
    vals = parfait_metric_values(df, selection)
    return EvalRow(
        sample=sid,
        embedding=embedding,
        family="parfait",
        embedding_type="parfait_best_grid_metric_not_raw_embedding",
        status="success",
        ARI=float(vals["ARI"]),
        NMI=float(vals["NMI"]),
        n_clusters=int(vals["n_clusters"]),
        label_key="ground_truth",
        cluster_method="precomputed_grid_metrics",
        leiden_resolution=mean_numeric(df, "leiden_resolution"),
        leiden_n_neighbors=rounded_mean_int(df, "leiden_n_neighbors"),
        seed=-1 if args.parfait_seed_mode == "mean" else int(df["seed"].iloc[0]),
        n_runs=int(len(df)),
        path=path,
        key=key,
        order_mode="precomputed",
        selection_metric=selection,
        t=float(vals["t"]),
        s=float(vals["s"]),
    )


def append_parfait_best(repo: Path, parfait_root: Path, sids: Sequence[str], args: argparse.Namespace) -> List[EvalRow]:
    root = parfait_root if parfait_root.is_absolute() else repo / parfait_root
    rows: List[EvalRow] = []
    for sid in sids:
        files = sorted((root / sid).glob("seed_*/best_metrics.csv"))
        if not files:
            files = sorted(root.glob(f"{sid}/seed_*/best_metrics.csv"))
        if args.parfait_seed_mode == "seed_00":
            files = [f for f in files if f.parent.name in {"seed_00", "seed_0"}]
        if not files:
            rows.append(
                EvalRow(
                    sample=sid,
                    embedding=args.parfait_name,
                    family="parfait",
                    embedding_type="parfait_best_grid_metric",
                    status="failed",
                    path=str(root / sid),
                    error="No best_metrics.csv found",
                )
            )
            continue
        seed_frames = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    raise ValueError("empty best_metrics.csv")
                r = df.iloc[[0]].copy()
                if "seed" not in r or pd.isna(r["seed"].iloc[0]):
                    r["seed"] = parse_seed_from_path(f)
                r["source_path"] = str(f)
                seed_frames.append(r)
            except Exception as e:
                rows.append(
                    EvalRow(
                        sample=sid,
                        embedding=args.parfait_name,
                        family="parfait",
                        embedding_type="parfait_best_grid_metric",
                        status="failed",
                        path=str(f),
                        error=repr(e),
                    )
                )
        if not seed_frames:
            continue
        seed_df = pd.concat(seed_frames, ignore_index=True)
        selections = ["ari", "nmi"] if args.parfait_selection == "both" else [args.parfait_selection]
        if args.parfait_seed_mode == "mean":
            for sel in selections:
                rows.append(
                    make_parfait_row(
                        sid=sid,
                        embedding=parfait_embedding_name(args.parfait_name, sel, args.parfait_selection == "both"),
                        selection=sel,
                        df=seed_df,
                        path=str(root / sid),
                        key="best_metrics.csv:mean_over_seeds",
                        args=args,
                    )
                )
        else:
            for _, r in seed_df.iterrows():
                one = pd.DataFrame([r])
                seed_path = str(r.get("source_path", root / sid))
                for sel in selections:
                    rows.append(
                        make_parfait_row(
                            sid=sid,
                            embedding=parfait_embedding_name(args.parfait_name, sel, args.parfait_selection == "both"),
                            selection=sel,
                            df=one,
                            path=seed_path,
                            key="best_metrics.csv",
                            args=args,
                        )
                    )
    return rows


def first_existing_path(root: Path, dirs: Sequence[str], filename: str) -> Optional[Path]:
    for dirname in dirs:
        path = root / dirname / filename
        if path.exists():
            return path
    return None


def append_traditional_baselines(repo: Path, baseline_root: Path, sids: Sequence[str], args: argparse.Namespace) -> List[EvalRow]:
    root = baseline_root if baseline_root.is_absolute() else repo / baseline_root
    rows: List[EvalRow] = []
    requested = [x.lower() for x in split_csv(args.traditional_baseline_methods)]
    sample_set = set(map(str, sids))
    for method in requested:
        spec = TRADITIONAL_BASELINE_SPECS.get(method)
        if spec is None:
            rows.append(
                EvalRow(
                    sample="",
                    embedding=method,
                    family="traditional_spatial_method",
                    embedding_type="traditional_spatial_domain",
                    status="failed",
                    error=f"Unknown traditional baseline {method!r}; known: {sorted(TRADITIONAL_BASELINE_SPECS)}",
                )
            )
            continue
        path = first_existing_path(root, spec["dirs"], spec["metrics_file"])
        if path is None:
            print(f"[traditional] skip {spec['embedding']}: metrics CSV not found under {root}", flush=True)
            continue
        df = pd.read_csv(path)
        if "sample" in df.columns:
            df = df[df["sample"].astype(str).isin(sample_set)].copy()
        for _, r in df.iterrows():
            status_raw = str(r.get("status", "ok")).strip().lower()
            status = "success" if status_raw in {"ok", "success"} else status_raw
            rows.append(
                EvalRow(
                    sample=str(r.get("sample", "")),
                    embedding=str(r.get("embedding", spec["embedding"])),
                    family=str(r.get("family", "traditional_spatial_method")),
                    embedding_type=str(r.get("embedding_type", "traditional_spatial_domain")),
                    status=status,
                    ARI=float(pd.to_numeric(pd.Series([r.get("ARI", math.nan)]), errors="coerce").iloc[0]),
                    NMI=float(pd.to_numeric(pd.Series([r.get("NMI", math.nan)]), errors="coerce").iloc[0]),
                    n_clusters=int(pd.to_numeric(pd.Series([r.get("n_clusters", 0)]), errors="coerce").fillna(0).iloc[0]),
                    n_labels=int(pd.to_numeric(pd.Series([r.get("n_labels", 0)]), errors="coerce").fillna(0).iloc[0]),
                    n_spots=int(pd.to_numeric(pd.Series([r.get("n_spots", 0)]), errors="coerce").fillna(0).iloc[0]),
                    n_valid_labels=int(pd.to_numeric(pd.Series([r.get("n_valid_labels", 0)]), errors="coerce").fillna(0).iloc[0]),
                    label_key=str(r.get("label_key", "")),
                    cluster_method="precomputed_traditional_baseline",
                    path=str(path),
                    key=str(r.get("cluster_key", "")),
                    order_mode="precomputed",
                    error=str(r.get("error", "")),
                )
            )
        print(f"[traditional] loaded {len(df)} rows for {spec['embedding']} from {path}", flush=True)
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].eq("success")].copy()
    if ok.empty:
        return pd.DataFrame()
    agg = (
        ok.groupby(["embedding", "family", "embedding_type"], dropna=False)
        .agg(
            n_rows=("ARI", "size"),
            n_samples=("sample", "nunique"),
            ARI_mean=("ARI", "mean"),
            ARI_std=("ARI", "std"),
            ARI_median=("ARI", "median"),
            NMI_mean=("NMI", "mean"),
            NMI_std=("NMI", "std"),
            NMI_median=("NMI", "median"),
        )
        .reset_index()
        .sort_values(["ARI_mean", "NMI_mean"], ascending=False)
    )
    return agg


def order_embeddings(cols: Iterable[str]) -> List[str]:
    preferred = [
        "PARFAIT",
        "PARFAIT_ARI",
        "PARFAIT_NMI",
        "parfait",
        "parfait_best_ARI",
        "parfait_best_NMI",
        "parfait_best",
        "hvg_pca",
        "spatial_xy",
        "scgpt",
        "nicheformer",
        "loki_text",
        "loki_image",
        "stpath",
        "phikon_v2",
        "uni2_h",
        "BayesSpace",
        "SpaGCN",
        "STAGATE",
    ]
    cols = list(cols)
    return [x for x in preferred if x in cols] + sorted([x for x in cols if x not in preferred])


def plot_heatmap(df: pd.DataFrame, metric: str, out: Path) -> Optional[Path]:
    ok = df[df["status"].eq("success")].copy()
    if ok.empty:
        return None
    pivot = ok.pivot_table(index="sample", columns="embedding", values=metric, aggfunc="mean")
    pivot = pivot.reindex(index=sorted(pivot.index), columns=order_embeddings(pivot.columns))
    if pivot.empty:
        return None
    fig_w = max(9, 0.7 * len(pivot.columns) + 2)
    fig_h = max(5, 0.35 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(f"DLPFC per-sample {metric} by embedding")
    ax.set_xlabel("Embedding")
    ax.set_ylabel("DLPFC sample")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val < 0.65 else "black")
    fig.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    png = out / f"heatmap_{metric.lower()}_sample_by_embedding.png"
    pdf = out / f"heatmap_{metric.lower()}_sample_by_embedding.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    pivot.to_csv(out / f"matrix_{metric.lower()}_sample_by_embedding.csv")
    return png


def plot_bar(summary: pd.DataFrame, metric: str, out: Path) -> Optional[Path]:
    if summary.empty:
        return None
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    df = summary.dropna(subset=[mean_col]).copy()
    if df.empty:
        return None
    order = order_embeddings(df["embedding"])
    df["embedding_order"] = pd.Categorical(df["embedding"], categories=order, ordered=True)
    df = df.sort_values("embedding_order")

    fig_w = max(8, 0.72 * len(df) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    x = np.arange(len(df))
    yerr = df[std_col].fillna(0).to_numpy()
    colors = [BAR_COLORS.get(str(e), "#4C78A8") for e in df["embedding"]]
    ax.bar(
        x,
        df[mean_col].to_numpy(),
        yerr=yerr,
        color=colors,
        alpha=0.9,
        edgecolor="#333333",
        linewidth=0.7,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(str(e), str(e)) for e in df["embedding"]], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Mean {metric} across samples")
    ax.set_xlabel("Embedding")
    ax.set_title(f"DLPFC mean {metric} by embedding")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for xi, val in zip(x, df[mean_col]):
        ax.text(xi, min(0.98, val + 0.025), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    png = out / f"bar_mean_{metric.lower()}_by_embedding.png"
    pdf = out / f"bar_mean_{metric.lower()}_by_embedding.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_violin(df: pd.DataFrame, metric: str, out: Path) -> Optional[Path]:
    ok = df[df["status"].eq("success")].dropna(subset=[metric]).copy()
    if ok.empty:
        return None
    order = order_embeddings(ok["embedding"])
    if not order:
        return None

    fig_w = 6.5
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    rng = np.random.default_rng(0)
    positions = np.arange(len(order), dtype=float)

    for i, emb in enumerate(order):
        vals = ok.loc[ok["embedding"].eq(emb), metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        color = BAR_COLORS.get(str(emb), "#4C78A8")
        parts = ax.violinplot(
            vals,
            positions=[positions[i]],
            widths=0.72,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("#333333")
            body.set_alpha(0.42)
            body.set_linewidth(0.8)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#222222")
            parts["cmedians"].set_linewidth(1.4)

        jitter = rng.normal(0.0, 0.055, size=vals.size)
        ax.scatter(
            np.full(vals.size, positions[i]) + jitter,
            vals,
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.92,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY_NAMES.get(str(e), str(e)) for e in order], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel(metric)
    ax.set_xlabel("Embedding")
    ax.set_title(f"DLPFC {metric} distribution by embedding")
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    png = out / f"violin_{metric.lower()}_by_embedding.png"
    pdf = out / f"violin_{metric.lower()}_by_embedding.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    small = df.head(max_rows).copy()
    for c in small.select_dtypes(include=[float]).columns:
        small[c] = small[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    try:
        return small.to_markdown(index=False)
    except Exception:
        return small.to_csv(index=False)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, out: Path, args: argparse.Namespace, plot_paths: List[Path]) -> None:
    failures = df[~df["status"].eq("success")].copy()
    lines = [
        "# DLPFC Embedding ARI/NMI Report",
        "",
        "This report clusters each DLPFC sample embedding and compares predicted clusters with manual layer labels.",
        "",
        "## Settings",
        "",
        f"- Samples: `{args.samples}`",
        f"- Embeddings requested: `{args.embedding_names}`",
        f"- Label key: `{args.label_key}`",
        f"- Cluster method: `{args.cluster_method}`",
        f"- Leiden neighbors/resolution: `{args.leiden_n_neighbors}` / `{args.leiden_resolution}`",
        f"- High-dimensional clustering PCA dim: `{args.cluster_pca_dim}`",
        f"- PARFAIT best metrics included: `{bool(args.include_parfait_best)}`",
        f"- Traditional baseline metrics included: `{bool(args.include_traditional_baselines)}`",
        f"- Traditional baseline root / methods: `{args.traditional_baseline_root}` / `{args.traditional_baseline_methods}`",
        f"- PARFAIT root: `{args.parfait_root}`",
        f"- PARFAIT seed mode / selection: `{args.parfait_seed_mode}` / `{args.parfait_selection}`",
        "",
        "## Mean Performance",
        "",
        df_to_markdown(summary),
        "",
        "## Output Files",
        "",
        f"- Long table: `{out / 'dlpfc_embedding_ari_nmi_long.csv'}`",
        f"- Summary table: `{out / 'dlpfc_embedding_ari_nmi_summary.csv'}`",
    ]
    for p in plot_paths:
        lines.append(f"- Plot: `{p}`")
    if not failures.empty:
        lines.extend(
            [
                "",
                "## Missing Or Failed Embeddings",
                "",
                df_to_markdown(failures[["sample", "embedding", "status", "path", "key", "error"]], max_rows=40),
            ]
        )
    lines.extend(
        [
            "",
            "## Note",
            "",
            "PARFAIT rows come from existing `best_metrics.csv` files under `--parfait_root`. They are best-grid PARFAIT model results, not raw source embedding clustering results.",
        ]
    )
    (out / "dlpfc_embedding_ari_nmi_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = repo / data_root
    embedding_root = Path(args.embedding_root)
    if not embedding_root.is_absolute():
        embedding_root = repo / embedding_root
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = repo / out
    out.mkdir(parents=True, exist_ok=True)

    samples = resolve_samples(args.samples, data_root)
    embeddings = split_csv(args.embedding_names)
    specs = load_specs(args.embedding_specs_json)

    all_rows: List[EvalRow] = []
    for i, sid in enumerate(samples, start=1):
        print(f"[{i}/{len(samples)}] sample {sid}", flush=True)
        rows = read_one_sample(repo, data_root, embedding_root, sid, embeddings, specs, args)
        all_rows.extend(rows)
        for r in rows:
            if r.status == "success":
                print(f"  [ok] {r.embedding:24s} ARI={r.ARI:.3f} NMI={r.NMI:.3f} ({r.cluster_method})", flush=True)
            else:
                print(f"  [skip] {r.embedding:24s} {r.status}: {r.error}", flush=True)

    if args.include_parfait_best:
        print("[parfait] reading existing best_metrics.csv files", flush=True)
        all_rows.extend(append_parfait_best(repo, Path(args.parfait_root), samples, args))
    if args.include_traditional_baselines:
        print("[traditional] reading existing BayesSpace/SpaGCN/STAGATE metrics", flush=True)
        all_rows.extend(append_traditional_baselines(repo, Path(args.traditional_baseline_root), samples, args))

    df = pd.DataFrame([asdict(r) for r in all_rows])
    df.to_csv(out / "dlpfc_embedding_ari_nmi_long.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "dlpfc_embedding_ari_nmi_summary.csv", index=False)

    plot_paths: List[Path] = []
    if not args.no_plots:
        for metric in ["ARI", "NMI"]:
            p = plot_heatmap(df, metric, out)
            if p is not None:
                plot_paths.append(p)
            p = plot_violin(df, metric, out)
            if p is not None:
                plot_paths.append(p)

    write_report(df, summary, out, args, plot_paths)
    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
