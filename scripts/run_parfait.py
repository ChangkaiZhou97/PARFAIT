#!/usr/bin/env python3
"""
PARFAIT

New in this version
-------------------
1) Fine t/s grids, with heatmap + contour plots for ARI/NMI.
2) Seed-aware output directories and per-run result files, suitable for 20 seeds × 12 slices.
3) Aggregate mode: reads all per-seed results and produces per-slice violin plots of best ARI/NMI,
   plus mean heatmaps over seeds/slices.

Typical training output directory:
    OUTPUT_ROOT/<SID>/seed_<SEED>/

Key outputs per run:
    losses.npy
    elbo_curve.png
    t_s_table.csv
    rayleigh_table.csv
    grid_metrics.csv                all (t, s) -> ARI, NMI, n_clusters
    best_metrics.csv                one-row summary for this SID/seed
    heatmap_ari.png / heatmap_nmi.png
    contour_ari.png / contour_nmi.png
    learned_ts_overlay.png
    run_config.json
    run_summary.txt

Aggregate outputs:
    OUTPUT_ROOT/aggregate/best_metrics_all.csv
    OUTPUT_ROOT/aggregate/grid_metrics_all.csv
    OUTPUT_ROOT/aggregate/violin_best_ari_by_slice.png
    OUTPUT_ROOT/aggregate/violin_best_nmi_by_slice.png
    OUTPUT_ROOT/aggregate/mean_heatmap_ari_all.png
    OUTPUT_ROOT/aggregate/mean_heatmap_nmi_all.png
    OUTPUT_ROOT/aggregate/<SID>_mean_heatmap_ari.png
    OUTPUT_ROOT/aggregate/<SID>_mean_heatmap_nmi.png
    OUTPUT_ROOT/aggregate/learned_ts_all.csv
    OUTPUT_ROOT/aggregate/learned_ts_by_slice_embedding_summary.csv
    OUTPUT_ROOT/aggregate/violin_t_learned_by_embedding_all.png
    OUTPUT_ROOT/aggregate/violin_s_learned_by_embedding_all.png
    OUTPUT_ROOT/aggregate/<SID>_violin_t_learned_by_embedding.png
    OUTPUT_ROOT/aggregate/<SID>_violin_s_learned_by_embedding.png
    OUTPUT_ROOT/aggregate/mean_heatmap_t_learned_by_slice_embedding.png
    OUTPUT_ROOT/aggregate/mean_heatmap_s_learned_by_slice_embedding.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


DLPFC_SIDS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]


def _parse_grid(spec: Optional[str], n: int) -> List[float]:
    """Parse comma-separated grid, otherwise linspace(0, 1, n)."""
    if spec is None or str(spec).strip() == "":
        return np.round(np.linspace(0.0, 1.0, n), 4).tolist()
    vals = [float(x.strip()) for x in spec.split(",") if x.strip() != ""]
    vals = sorted(set(np.round(vals, 4).tolist()))
    if min(vals) < 0 or max(vals) > 1:
        raise ValueError(f"Grid values must be in [0, 1], got {vals}")
    return vals


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default=".")
    p.add_argument("--sid", type=str, default="151673")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_root", type=str, default=None,
                   help="Root for multi-slice/multi-seed outputs. Default: <repo>/output/DLPFC_parfait_finegrid")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Optional exact output directory. Overrides output_root/SID/seed_XXX.")
    p.add_argument("--data_root", type=str, default="data/DLPFC_local",
                   help="Root with DLPFC <sid>/<sid>_with_gt.h5ad files.")
    p.add_argument("--embedding_root", type=str, default="output",
                   help="Root with precomputed embedding folders.")

    # aggregate mode
    p.add_argument("--aggregate_only", action="store_true",
                   help="Do not train. Read all grid/best CSVs under output_root and make summary plots.")
    p.add_argument("--aggregate_sids", type=str, default="",
                   help="Comma-separated SIDs to aggregate. Default: all found under output_root.")

    # PARFAIT hyperparams
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--n_neighbors_spatial", type=int, default=6)
    p.add_argument("--K_diff", type=int, default=4)
    p.add_argument("--alpha_max", type=float, default=0.1)
    p.add_argument("--lambda_spatial", type=float, default=0.0,
                   help="GMRF on z^1; default 0 because z^1 is already smooth.")
    p.add_argument("--lambda_path", type=float, default=0.0)
    p.add_argument("--beta_strength_t", type=float, default=4.0)
    p.add_argument("--beta_strength_s", type=float, default=4.0)
    p.add_argument("--n_steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kl_warmup_steps", type=int, default=1000)
    p.add_argument("--kl_ramp_steps", type=int, default=2000)

    # 2D evaluation grid. Default 21×21 = 0.05 step on both axes.
    p.add_argument("--n_t", type=int, default=21, help="grid points on t axis if --t_grid not supplied")
    p.add_argument("--n_s", type=int, default=21, help="grid points on s axis if --s_grid not supplied")
    p.add_argument("--t_grid", type=str, default=None,
                   help="Comma-separated explicit t grid, e.g. '0,0.05,0.1,...,1'. Overrides --n_t.")
    p.add_argument("--s_grid", type=str, default=None,
                   help="Comma-separated explicit s grid. Overrides --n_s.")

    # Leiden config
    p.add_argument("--leiden_n_neighbors", type=int, default=15)
    p.add_argument("--leiden_resolution", type=float, default=0.5)
    p.add_argument("--leiden_seed", type=int, default=0)

    # output/storage controls
    p.add_argument("--save_latents", action="store_true",
                   help="Save all grid latents into latents.h5ad. Off by default because 20 seeds × fine grid is large.")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def leiden_metrics(Z: np.ndarray, gt: np.ndarray, n_neighbors=15,
                   resolution=0.5, seed=0):
    a = ad.AnnData(X=np.zeros((Z.shape[0], 1), dtype=np.float32))
    a.obsm["X_emb"] = Z.astype(np.float32)
    sc.pp.neighbors(a, use_rep="X_emb", n_neighbors=n_neighbors, random_state=seed)
    sc.tl.leiden(a, resolution=resolution, random_state=seed,
                 flavor="igraph", n_iterations=2, directed=False)
    pred = a.obs["leiden"].astype(str).values
    gt_str = gt.astype(str)
    valid = gt_str != "nan"
    return {
        "ARI": float(adjusted_rand_score(gt_str[valid], pred[valid])),
        "NMI": float(normalized_mutual_info_score(gt_str[valid], pred[valid])),
        "n_clusters": int(np.unique(pred[valid]).size),
    }


def _metric_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot_table(index="s", columns="t", values=metric, aggfunc="mean").sort_index().sort_index(axis=1)


def plot_heatmap(pivot: pd.DataFrame, title: str, fname: Path,
                 cmap: str = "viridis", annotate: bool = False):
    fig, ax = plt.subplots(figsize=(8, 6))
    ts = pivot.columns.values.astype(float)
    ss = pivot.index.values.astype(float)
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[ts.min(), ts.max(), ss.min(), ss.max()],
        cmap=cmap,
    )
    plt.colorbar(im, ax=ax)
    if annotate and pivot.size <= 121:
        for i, s in enumerate(ss):
            for j, t in enumerate(ts):
                ax.text(t, s, f"{pivot.values[i, j]:.2f}",
                        ha="center", va="center", color="white", fontsize=7)
    ax.set_xlabel("t  (abstraction; 0=micro, 1=macro)")
    ax.set_ylabel("s  (spatial smoothing; 0=raw, 1=max)")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_contour(pivot: pd.DataFrame, title: str, fname: Path,
                 cmap: str = "viridis", levels: int = 18):
    fig, ax = plt.subplots(figsize=(8, 6))
    ts = pivot.columns.values.astype(float)
    ss = pivot.index.values.astype(float)
    T, S = np.meshgrid(ts, ss)
    vals = pivot.values.astype(float)
    cf = ax.contourf(T, S, vals, levels=levels, cmap=cmap)
    cs = ax.contour(T, S, vals, levels=min(levels, 10), colors="black", linewidths=0.4, alpha=0.65)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel("t  (abstraction; 0=micro, 1=macro)")
    ax.set_ylabel("s  (spatial smoothing; 0=raw, 1=max)")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_learned_overlay(ari_pivot: pd.DataFrame, t_s_df: pd.DataFrame,
                         best_row: pd.Series, fname: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ts = ari_pivot.columns.values.astype(float)
    ss = ari_pivot.index.values.astype(float)
    im = ax.imshow(
        ari_pivot.values,
        origin="lower",
        aspect="auto",
        extent=[ts.min(), ts.max(), ss.min(), ss.max()],
        cmap="viridis",
        alpha=0.85,
    )
    plt.colorbar(im, ax=ax, label="ARI")
    cmap_pts = plt.get_cmap("tab10")
    for i, row in t_s_df.reset_index(drop=True).iterrows():
        ax.scatter(row["t_learned"], row["s_learned"], s=160, marker="*",
                   color=cmap_pts(i % 10), edgecolor="black", linewidth=1.2, zorder=5,
                   label=f"{row['embedding']} (t={row['t_learned']:.2f}, s={row['s_learned']:.2f})")
    ax.scatter(best_row["t"], best_row["s"], s=300, marker="X",
               color="red", edgecolor="white", linewidth=2, zorder=6,
               label=f"best ARI={best_row['ARI']:.3f}")
    ax.set_xlabel("t  (abstraction)")
    ax.set_ylabel("s  (spatial smoothing)")
    ax.set_title("Learned (t_m, s_m) overlaid on ARI heatmap")
    ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=8)
    plt.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_violin_by_slice(df: pd.DataFrame, metric: str, fname: Path, title: str):
    sids = sorted(df["sid"].astype(str).unique())
    data = [df.loc[df["sid"].astype(str) == sid, metric].dropna().values for sid in sids]
    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(sids)), 5))
    parts = ax.violinplot(data, positions=np.arange(len(sids)), showmeans=True, showmedians=True, showextrema=True)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        rng = np.random.default_rng(1234 + i)
        x = i + rng.normal(0, 0.035, size=len(vals))
        ax.scatter(x, vals, s=16, alpha=0.65)
    ax.set_xticks(np.arange(len(sids)))
    ax.set_xticklabels(sids, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)




def _embedding_order(values: Sequence[str]) -> List[str]:
    preferred = [
        "hvg_pca", "nicheformer", "loki_text", "scgpt", "stpath",
        "loki_image", "phikon_v2", "uni2_h",
    ]
    vals = [str(v) for v in values]
    return [e for e in preferred if e in vals] + [e for e in sorted(vals) if e not in preferred]


def plot_violin_by_embedding(df: pd.DataFrame, metric: str, fname: Path, title: str):
    """Violin plot of learned t/s by embedding.

    The input can be one slice only or all slices pooled. Each violin shows the
    distribution across seeds, or across slice×seed when pooled.
    """
    if df.empty:
        return
    embeddings = _embedding_order(df["embedding"].astype(str).unique())
    data = [df.loc[df["embedding"].astype(str) == emb, metric].dropna().values for emb in embeddings]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(embeddings)), 5))
    ax.violinplot(data, positions=np.arange(len(embeddings)),
                  showmeans=True, showmedians=True, showextrema=True)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        rng = np.random.default_rng(4321 + i)
        x = i + rng.normal(0, 0.035, size=len(vals))
        ax.scatter(x, vals, s=16, alpha=0.65)
    ax.set_xticks(np.arange(len(embeddings)))
    ax.set_xticklabels(embeddings, rotation=45, ha="right")
    if metric in {"t_learned", "s_learned", "t_init", "s_init"}:
        ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_learned_mean_heatmap(df: pd.DataFrame, metric: str, fname: Path, title: str):
    """Heatmap of mean learned t/s values: rows=slices, columns=embeddings."""
    if df.empty:
        return
    sids = sorted(df["sid"].astype(str).unique())
    embeddings = _embedding_order(df["embedding"].astype(str).unique())
    mat = np.full((len(sids), len(embeddings)), np.nan, dtype=float)

    for i, sid in enumerate(sids):
        for j, emb in enumerate(embeddings):
            vals = df.loc[
                (df["sid"].astype(str) == sid) &
                (df["embedding"].astype(str) == emb),
                metric
            ].dropna().values
            if len(vals) > 0:
                mat[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(embeddings)), max(5, 0.35 * len(sids))))
    im = ax.imshow(mat, origin="upper", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label=f"mean {metric}")
    ax.set_xticks(np.arange(len(embeddings)))
    ax.set_xticklabels(embeddings, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(sids)))
    ax.set_yticklabels(sids)
    for i in range(len(sids)):
        for j in range(len(embeddings)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_results(output_root: Path, aggregate_sids: str = ""):
    output_root.mkdir(parents=True, exist_ok=True)
    agg_dir = output_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    wanted = None
    if aggregate_sids.strip():
        wanted = {x.strip() for x in aggregate_sids.split(",") if x.strip()}

    best_files = sorted(output_root.glob("*/seed_*/best_metrics.csv"))
    grid_files = sorted(output_root.glob("*/seed_*/grid_metrics.csv"))
    ts_files = sorted(output_root.glob("*/seed_*/t_s_table.csv"))
    if wanted is not None:
        best_files = [p for p in best_files if p.parents[1].name in wanted]
        grid_files = [p for p in grid_files if p.parents[1].name in wanted]
        ts_files = [p for p in ts_files if p.parents[1].name in wanted]

    if len(best_files) == 0:
        raise FileNotFoundError(f"No best_metrics.csv found under {output_root}/<SID>/seed_*/")

    def _read_with_sid_seed(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["sid"] = str(path.parents[1].name)
        try:
            df["seed"] = int(str(path.parent.name).replace("seed_", ""))
        except Exception:
            df["seed"] = str(path.parent.name)
        return df

    best_df = pd.concat([_read_with_sid_seed(p) for p in best_files], ignore_index=True)
    grid_df = pd.concat([_read_with_sid_seed(p) for p in grid_files], ignore_index=True) if grid_files else pd.DataFrame()
    ts_df = pd.concat([_read_with_sid_seed(p) for p in ts_files], ignore_index=True) if ts_files else pd.DataFrame()

    best_df.to_csv(agg_dir / "best_metrics_all.csv", index=False)
    if not grid_df.empty:
        grid_df.to_csv(agg_dir / "grid_metrics_all.csv", index=False)
    if not ts_df.empty:
        ts_df.to_csv(agg_dir / "t_s_table_all.csv", index=False)
        ts_df.to_csv(agg_dir / "learned_ts_all.csv", index=False)

    # Violin plots of per-seed best ARI/NMI for each slice.
    plot_violin_by_slice(best_df, "best_ARI", agg_dir / "violin_best_ari_by_slice.png",
                         "Best ARI over (t, s) grid across seeds")
    plot_violin_by_slice(best_df, "best_NMI", agg_dir / "violin_best_nmi_by_slice.png",
                         "Best NMI over (t, s) grid across seeds")
    if "NMI_at_best_ARI" in best_df.columns:
        plot_violin_by_slice(best_df, "NMI_at_best_ARI",
                             agg_dir / "violin_nmi_at_best_ari_by_slice.png",
                             "NMI at ARI-best (t, s) across seeds")

    # Violin plots of learned t/s across seeds for each slice, grouped by embedding.
    # These are the main diagnostics for learned scale coordinates.
    if not ts_df.empty:
        for sid, sub in ts_df.groupby("sid"):
            plot_violin_by_embedding(
                sub, "t_learned",
                agg_dir / f"{sid}_violin_t_learned_by_embedding.png",
                f"{sid}: learned t across seeds by embedding",
            )
            plot_violin_by_embedding(
                sub, "s_learned",
                agg_dir / f"{sid}_violin_s_learned_by_embedding.png",
                f"{sid}: learned s across seeds by embedding",
            )

        # Global plots pooled across all slices, still grouped by embedding.
        plot_violin_by_embedding(
            ts_df, "t_learned",
            agg_dir / "violin_t_learned_by_embedding_all.png",
            "Learned t across all slices/seeds by embedding",
        )
        plot_violin_by_embedding(
            ts_df, "s_learned",
            agg_dir / "violin_s_learned_by_embedding_all.png",
            "Learned s across all slices/seeds by embedding",
        )

        # Heatmaps of mean learned t/s across seeds for each slice×embedding.
        plot_learned_mean_heatmap(
            ts_df, "t_learned",
            agg_dir / "mean_heatmap_t_learned_by_slice_embedding.png",
            "Mean learned t by slice and embedding",
        )
        plot_learned_mean_heatmap(
            ts_df, "s_learned",
            agg_dir / "mean_heatmap_s_learned_by_slice_embedding.png",
            "Mean learned s by slice and embedding",
        )

    # Summary tables.
    summary = best_df.groupby("sid", as_index=False).agg(
        n_runs=("seed", "count"),
        best_ARI_mean=("best_ARI", "mean"),
        best_ARI_sd=("best_ARI", "std"),
        best_NMI_mean=("best_NMI", "mean"),
        best_NMI_sd=("best_NMI", "std"),
        t_at_best_ARI_median=("t_at_best_ARI", "median"),
        s_at_best_ARI_median=("s_at_best_ARI", "median"),
    )
    summary.to_csv(agg_dir / "best_metrics_by_slice_summary.csv", index=False)

    if not ts_df.empty:
        ts_summary = ts_df.groupby(["sid", "embedding"], as_index=False).agg(
            n_runs=("seed", "count"),
            t_learned_mean=("t_learned", "mean"),
            t_learned_sd=("t_learned", "std"),
            s_learned_mean=("s_learned", "mean"),
            s_learned_sd=("s_learned", "std"),
        )
        ts_summary.to_csv(agg_dir / "t_s_learned_by_slice_embedding_summary.csv", index=False)

    # Mean heatmaps.
    if not grid_df.empty:
        for metric in ["ARI", "NMI"]:
            mean_df = grid_df.groupby(["s", "t"], as_index=False)[metric].mean()
            pivot = _metric_pivot(mean_df, metric)
            plot_heatmap(pivot, f"Mean {metric} over all slices/seeds", agg_dir / f"mean_heatmap_{metric.lower()}_all.png")
            plot_contour(pivot, f"Mean {metric} over all slices/seeds", agg_dir / f"mean_contour_{metric.lower()}_all.png")

            for sid, sub in grid_df.groupby("sid"):
                mean_sid = sub.groupby(["s", "t"], as_index=False)[metric].mean()
                pivot_sid = _metric_pivot(mean_sid, metric)
                plot_heatmap(pivot_sid, f"{sid}: mean {metric} over seeds", agg_dir / f"{sid}_mean_heatmap_{metric.lower()}.png")
                plot_contour(pivot_sid, f"{sid}: mean {metric} over seeds", agg_dir / f"{sid}_mean_contour_{metric.lower()}.png")

    print(f"[aggregate] wrote summary to {agg_dir}")
    print(summary.to_string(index=False))
    if not ts_df.empty:
        print(f"[aggregate] wrote learned t/s tables and plots to {agg_dir}")


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(repo))

    output_root = Path(args.output_root) if args.output_root else (
        repo / "output" / "DLPFC_parfait_finegrid"
    )

    if args.aggregate_only:
        aggregate_results(output_root, args.aggregate_sids)
        return

    from parfait_model import (
        PARFAITConfig, build_parfait,
        intrinsic_dim_to_t_init, smoothness_to_s_init,
    )
    from prepare_data import load_single_slice

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PARFAIT single-slice  sid={args.sid} seed={args.seed} device={device} ===")

    out_dir = Path(args.output_dir) if args.output_dir else (
        output_root / str(args.sid) / f"seed_{args.seed:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["output_root"] = str(output_root)
    run_config["output_dir_resolved"] = str(out_dir)
    run_config["data_root_resolved"] = str((repo / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root))
    run_config["embedding_root_resolved"] = str((repo / args.embedding_root).resolve() if not Path(args.embedding_root).is_absolute() else Path(args.embedding_root))
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=_json_safe)

    # Deterministic-ish setup. CUDA kernels may still have nondeterminism, but this removes avoidable noise.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    # ---- 1) load ----
    t0 = time.time()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = repo / data_root
    embedding_root = Path(args.embedding_root)
    if not embedding_root.is_absolute():
        embedding_root = repo / embedding_root

    data = load_single_slice(
        sid=args.sid,
        raw_root=data_root,
        output_root=embedding_root,
        n_hvg=args.n_hvg,
    )
    print(f"\nload done in {time.time()-t0:.1f}s   N={data['X_raw'].shape[0]}  "
          f"embeddings={data['emb_names']}")

    # ---- 2) data-driven (t_init, s_init) ----
    from parfait_model import build_laplacian_single_slice
    L_for_R = build_laplacian_single_slice(data["coords"], k=args.n_neighbors_spatial)
    t_init, ID_vals = intrinsic_dim_to_t_init(data["embeddings"])
    s_init, R_vals = smoothness_to_s_init(data["embeddings"], L_for_R)

    init_df = pd.DataFrame({
        "embedding": list(data["embeddings"].keys()),
        "intrinsic_dim_or_effective_rank": [ID_vals[n] for n in data["embeddings"].keys()],
        "rayleigh": [R_vals[n] for n in data["embeddings"].keys()],
        "t_init": [t_init[n] for n in data["embeddings"].keys()],
        "s_init": [s_init[n] for n in data["embeddings"].keys()],
    }).sort_values("rayleigh").reset_index(drop=True)
    init_df.to_csv(out_dir / "rayleigh_table.csv", index=False)
    print("\n[init table] t_init from intrinsic dimension; s_init from Rayleigh/smoothness:")
    print(init_df.to_string(index=False))

    # ---- 3) build model ----
    cfg = PARFAITConfig(
        n_genes=data["X_raw"].shape[1],
        emb_names=data["emb_names"], emb_dims=data["emb_dims"],
        K=args.K,
        K_diff=args.K_diff, alpha_max=args.alpha_max,
        lambda_spatial=args.lambda_spatial,
        lambda_path=args.lambda_path,
        beta_strength_t=args.beta_strength_t,
        beta_strength_s=args.beta_strength_s,
        t_init=t_init, s_init=s_init,
        kl_warmup_steps=args.kl_warmup_steps,
        kl_ramp_steps=args.kl_ramp_steps,
    )
    model, coords_std = build_parfait(cfg, data["coords"], n_neighbors=args.n_neighbors_spatial)
    model.L = model.L.to(device).coalesce()
    model.encoder.to(device)
    model.loc_encoder.to(device)
    model.path_net.to(device)
    model.smoother.to(device)

    X_raw_t = torch.from_numpy(data["X_raw"]).float().to(device)
    log_lib_t = torch.from_numpy(data["log_lib"]).float().to(device)
    coords_std_t = torch.from_numpy(coords_std).float().to(device)
    emb_list_t = [torch.from_numpy(data["embeddings"][n]).float().to(device)
                  for n in data["emb_names"]]

    n_params = sum(p.numel() for m in [model.encoder, model.loc_encoder, model.path_net]
                   for p in m.parameters())
    print(f"\n[model] K={cfg.K}  K_diff={cfg.K_diff}  alpha_max={cfg.alpha_max}  NN params={n_params:,}")

    # ---- 4) fit ----
    t0 = time.time()
    losses = model.fit(
        X_raw_t, emb_list_t, coords_std_t, log_lib=log_lib_t,
        n_steps=args.n_steps, lr=args.lr, log_every=200, seed=args.seed,
    )
    print(f"\ntrain done in {time.time()-t0:.1f}s")
    np.save(out_dir / "losses.npy", np.asarray(losses))

    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(losses)
        ax.set_xlabel("step")
        ax.set_ylabel("-ELBO")
        ax.set_yscale("log")
        ax.axvspan(0, cfg.kl_warmup_steps, alpha=0.1, color="red", label="KL warmup")
        ax.axvspan(cfg.kl_warmup_steps, cfg.kl_warmup_steps + cfg.kl_ramp_steps,
                   alpha=0.1, color="orange", label="KL ramp")
        ax.legend()
        ax.set_title("PARFAIT ELBO")
        plt.tight_layout()
        fig.savefig(out_dir / "elbo_curve.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # ---- 5) learned (t_m, s_m) ----
    t_s_learned = model.get_t_s()
    t_s_df = pd.DataFrame({
        "sid": args.sid,
        "seed": args.seed,
        "embedding": list(t_s_learned.keys()),
        "t_init": [t_init[n] for n in t_s_learned],
        "s_init": [s_init[n] for n in t_s_learned],
        "t_learned": [t_s_learned[n][0] for n in t_s_learned],
        "s_learned": [t_s_learned[n][1] for n in t_s_learned],
        "rayleigh": [R_vals[n] for n in t_s_learned],
        "intrinsic_dim_or_effective_rank": [ID_vals[n] for n in t_s_learned],
    })
    t_s_df.to_csv(out_dir / "t_s_table.csv", index=False)
    print("\n[learned (t_m, s_m)]:")
    print(t_s_df.to_string(index=False))

    # ---- 6) 2D grid evaluation ----
    t_grid = _parse_grid(args.t_grid, args.n_t)
    s_grid = _parse_grid(args.s_grid, args.n_s)
    print(f"\n[grid eval] n={len(t_grid)}×{len(s_grid)}  t_grid={t_grid}  s_grid={s_grid}")
    t0 = time.time()
    z_grid = model.z_at_grid(t_grid, s_grid)
    print(f"  computed {len(z_grid)} grid points in {time.time()-t0:.1f}s")

    print("\n[leiden on each (t, s)]")
    rows = []
    for (t, s), Z in z_grid.items():
        m = leiden_metrics(
            Z, data["ground_truth"],
            n_neighbors=args.leiden_n_neighbors,
            resolution=args.leiden_resolution,
            seed=args.leiden_seed,
        )
        rows.append({"sid": args.sid, "seed": args.seed, "t": t, "s": s, **m})
    grid_df = pd.DataFrame(rows).sort_values(["s", "t"]).reset_index(drop=True)
    grid_df.to_csv(out_dir / "grid_metrics.csv", index=False)
    print(grid_df.to_string(index=False))

    # ---- 7) ARI/NMI heatmaps + best (t, s) ----
    ari_pivot = _metric_pivot(grid_df, "ARI")
    nmi_pivot = _metric_pivot(grid_df, "NMI")
    nc_pivot = _metric_pivot(grid_df, "n_clusters")

    best_ari_row = grid_df.loc[grid_df["ARI"].idxmax()]
    best_nmi_row = grid_df.loc[grid_df["NMI"].idxmax()]
    best_df = pd.DataFrame([{
        "sid": args.sid,
        "seed": args.seed,
        "best_ARI": float(best_ari_row["ARI"]),
        "t_at_best_ARI": float(best_ari_row["t"]),
        "s_at_best_ARI": float(best_ari_row["s"]),
        "NMI_at_best_ARI": float(best_ari_row["NMI"]),
        "n_clusters_at_best_ARI": int(best_ari_row["n_clusters"]),
        "best_NMI": float(best_nmi_row["NMI"]),
        "t_at_best_NMI": float(best_nmi_row["t"]),
        "s_at_best_NMI": float(best_nmi_row["s"]),
        "ARI_at_best_NMI": float(best_nmi_row["ARI"]),
        "n_clusters_at_best_NMI": int(best_nmi_row["n_clusters"]),
        "K": args.K,
        "K_diff": args.K_diff,
        "alpha_max": args.alpha_max,
        "n_t": len(t_grid),
        "n_s": len(s_grid),
        "leiden_resolution": args.leiden_resolution,
        "leiden_n_neighbors": args.leiden_n_neighbors,
    }])
    best_df.to_csv(out_dir / "best_metrics.csv", index=False)

    print(f"\n[best by ARI] (t={best_ari_row['t']:.3f}, s={best_ari_row['s']:.3f})  "
          f"ARI={best_ari_row['ARI']:.4f}  NMI={best_ari_row['NMI']:.4f}  "
          f"k={int(best_ari_row['n_clusters'])}")
    print(f"[best by NMI] (t={best_nmi_row['t']:.3f}, s={best_nmi_row['s']:.3f})  "
          f"NMI={best_nmi_row['NMI']:.4f}  ARI={best_nmi_row['ARI']:.4f}  "
          f"k={int(best_nmi_row['n_clusters'])}")

    if not args.no_plots:
        annotate = (len(t_grid) * len(s_grid) <= 121)
        plot_heatmap(ari_pivot, "ARI heatmap on (t, s) grid", out_dir / "heatmap_ari.png", annotate=annotate)
        plot_heatmap(nmi_pivot, "NMI heatmap on (t, s) grid", out_dir / "heatmap_nmi.png", annotate=annotate)
        plot_heatmap(nc_pivot, "n_clusters heatmap on (t, s) grid", out_dir / "heatmap_n_clusters.png", cmap="plasma", annotate=annotate)
        plot_contour(ari_pivot, "ARI contour on (t, s) grid", out_dir / "contour_ari.png")
        plot_contour(nmi_pivot, "NMI contour on (t, s) grid", out_dir / "contour_nmi.png")
        plot_learned_overlay(ari_pivot, t_s_df, best_ari_row, out_dir / "learned_ts_overlay.png")

    # ---- 8) optional h5ad with all grid + learned-position latents ----
    if args.save_latents:
        a_out = ad.AnnData(X=np.zeros((data["X_raw"].shape[0], 1), dtype=np.float32))
        a_out.obs_names = list(data["obs_names"])
        a_out.obs["ground_truth"] = pd.Categorical(data["ground_truth"].astype(str))
        a_out.obsm["spatial"] = data["coords"]
        for (t, s), Z in z_grid.items():
            a_out.obsm[f"X_z_t{t:.2f}_s{s:.2f}"] = Z.astype(np.float32)
        learned_z = model.z_at_learned()
        for n, Z in learned_z.items():
            a_out.obsm[f"X_z_at_{n}"] = Z.astype(np.float32)
        a_out.uns["t_s_learned"] = {k: list(v) for k, v in t_s_learned.items()}
        a_out.uns["t_init"] = t_init
        a_out.uns["s_init"] = s_init
        a_out.uns["rayleigh"] = R_vals
        a_out.uns["intrinsic_dim_or_effective_rank"] = ID_vals
        a_out.write_h5ad(out_dir / "latents.h5ad")

    # ---- 9) summary ----
    with open(out_dir / "run_summary.txt", "w") as f:
        f.write(f"PARFAIT run summary  sid={args.sid} seed={args.seed}\n")
        f.write(f"args: {json.dumps(run_config, indent=2, default=_json_safe)}\n\n")
        f.write("=== Initial t/s table ===\n")
        f.write(init_df.to_string(index=False) + "\n\n")
        f.write("=== Learned (t_m, s_m) ===\n")
        f.write(t_s_df.to_string(index=False) + "\n\n")
        f.write("=== ARI/NMI on (t, s) grid ===\n")
        f.write(grid_df.to_string(index=False) + "\n\n")
        f.write("=== Best grid points ===\n")
        f.write(best_df.to_string(index=False) + "\n")

    print(f"\n=== DONE ===  output: {out_dir}")


if __name__ == "__main__":
    main()
