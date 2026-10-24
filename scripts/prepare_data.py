"""
CSTVAE demo single-slice data preparation.

Loads:
  - raw HVG counts from data/DLPFC_local/<sid>/<sid>_with_gt.h5ad
  - 7 FM embeddings: scgpt, loki_text, loki_image, phikon_v2, uni2_h,
    nicheformer, stpath
  - HVG-PCA computed on the fly (50d) as an 8th embedding
  - spatial coords + ground truth

NO PCA on FM embeddings; each is z-scored per feature within the slice.

STPath note: X_stpath is a unified multi-modal embedding (H&E morphology +
transcriptome + spatial coords + organ/tech) and is NOT directly comparable
with the purely transcriptomic (scgpt / loki_text / nicheformer) or purely
visual (loki_image / phikon_v2 / uni2_h) embeddings. Treat it as a separate
modality when interpreting downstream results.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA


def _zscore(X: np.ndarray) -> np.ndarray:
    return ((X - X.mean(0, keepdims=True))
            / (X.std(0, keepdims=True) + 1e-6)).astype(np.float32)


def select_hvg_single(a: ad.AnnData, n_top_genes: int = 2000,
                     flavor: str = "seurat_v3") -> List[str]:
    a2 = a.copy()
    sc.pp.highly_variable_genes(a2, n_top_genes=n_top_genes,
                                flavor=flavor, subset=False)
    return a2.var_names[a2.var["highly_variable"].values].tolist()


def _try_load_obsm(p: Path, key: str, order: List[str]) -> Optional[np.ndarray]:
    if not p.exists():
        return None
    a = sc.read_h5ad(str(p))
    if key not in a.obsm:
        # try fallback: any obsm key with 'uni' / 'path' / 'niche' / 'scgpt' / 'loki'
        for alt in a.obsm.keys():
            if any(s in alt.lower() for s in
                   ["uni", "path", "niche", "scgpt", "loki", "embed"]):
                key = alt
                break
        else:
            return None
    a = a[order]
    return np.nan_to_num(a.obsm[key], nan=0.0).astype(np.float32)


def _try_load_nicheformer(folder: Path, order: List[str]) -> Optional[np.ndarray]:
    """Special-case loader for the Nicheformer output folder structure."""
    h5 = folder / "embeddings.h5ad"
    npy = folder / "embeddings.npy"
    if h5.exists():
        a = sc.read_h5ad(str(h5))
        # try common obsm keys
        for k in ["X_nicheformer", "X_niche", "X_embedding", "X_emb"]:
            if k in a.obsm:
                if all(b in set(a.obs_names) for b in order):
                    return np.nan_to_num(a[order].obsm[k], nan=0.0).astype(np.float32)
                # row-aligned fallback
                arr = np.nan_to_num(a.obsm[k], nan=0.0).astype(np.float32)
                if arr.shape[0] == len(order):
                    return arr
        if a.X is not None and a.X.shape[1] in (256, 512, 768, 1024):
            X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
            if X.shape[0] == len(order):
                return X.astype(np.float32)
    if npy.exists():
        arr = np.load(str(npy)).astype(np.float32)
        if arr.shape[0] == len(order):
            return arr
    return None


def load_single_slice(
    sid: str,
    raw_root: Path,
    output_root: Path,
    n_hvg: int = 2000,
    n_pca: int = 50,
    pathology_specs: Tuple[Tuple[str, str, str], ...] = (
        ("phikon_v2", "DLPFC_{sid}_phikon_v2",  "X_path_phikon_v2"),
        ("uni2_h",    "DLPFC_{sid}_uni2-h",     "X_path_uni2_h"),
    ),
    use_hvg_pca: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Returns dict:
      X_raw         : (N, n_hvg) float32 (raw counts)
      hvg_genes     : list[str]
      coords        : (N, 2) float32
      ground_truth  : (N,) str
      log_lib       : (N,) float32
      obs_names     : list[str]
      embeddings    : dict[name -> (N, d_m)]      z-scored per feature
      emb_names     : list[str]                   ordered
      emb_dims      : list[int]                   ordered, matches emb_names
    """
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    # ---- Raw counts + canonical alignment ----
    raw_p = raw_root / sid / f"{sid}_with_gt.h5ad"
    a_raw = sc.read_h5ad(str(raw_p))
    canon_p = output_root / f"DLPFC_{sid}_loki_image_10x" / "embeddings.h5ad"
    a_canon = sc.read_h5ad(str(canon_p))
    common = set(a_canon.obs_names) & set(a_raw.obs_names)
    order = [b for b in a_canon.obs_names if b in common]
    a_raw_o = a_raw[order]
    a_canon_o = a_canon[order]
    if verbose:
        print(f"[align] N={len(order)}  (raw ∩ canonical loki_image)")

    # HVG selection
    hvg_genes = select_hvg_single(a_raw_o, n_top_genes=n_hvg)

    # raw counts
    Xfull = a_raw_o.X
    if hasattr(Xfull, "toarray"):
        Xfull = Xfull.toarray()
    Xfull = np.asarray(Xfull).astype(np.float32)
    log_lib = np.log(np.maximum(Xfull.sum(1), 1.0)).astype(np.float32)
    X_hvg = np.zeros((Xfull.shape[0], len(hvg_genes)), dtype=np.float32)
    for j, g in enumerate(hvg_genes):
        if g in a_raw_o.var_names:
            X_hvg[:, j] = Xfull[:, a_raw_o.var_names.get_loc(g)]

    # coords + GT
    coords = np.asarray(a_canon_o.obsm["spatial"], dtype=np.float32)
    if "ground_truth" in a_canon_o.obs:
        gt = a_canon_o.obs["ground_truth"].astype(str).to_numpy()
    elif "ground_truth" in a_raw_o.obs:
        gt = a_raw_o.obs["ground_truth"].astype(str).to_numpy()
    else:
        gt = np.array(["nan"] * len(order))

    # ---- Embeddings ----
    embeddings: Dict[str, np.ndarray] = {}

    # standard FMs (transcriptomic + image-LLM + multi-modal joint)
    standard = [
        ("scgpt",      f"DLPFC_{sid}_scgpt_with_spatial",    "X_scgpt_spatial"),
        ("loki_text",  f"DLPFC_{sid}_loki_text",             "X_loki_text"),
        ("loki_image", f"DLPFC_{sid}_loki_image_10x",        "X_loki_image"),
        ("stpath",     f"DLPFC_{sid}_stpath",                "X_stpath"),
    ]
    for name, folder, key in standard:
        p = output_root / folder / "embeddings.h5ad"
        arr = _try_load_obsm(p, key, order)
        if arr is None:
            if verbose:
                print(f"[skip] {name}: missing or unloadable {p}")
            continue
        embeddings[name] = arr
        if verbose:
            print(f"[ok ] {name:12s} {arr.shape}")

    # pathology FMs (Phikon, UNI2)
    for name, folder_tpl, key in pathology_specs:
        p = output_root / folder_tpl.format(sid=sid) / "embeddings.h5ad"
        arr = _try_load_obsm(p, key, order)
        if arr is None:
            if verbose:
                print(f"[skip] {name}: missing or unloadable {p}")
            continue
        embeddings[name] = arr
        if verbose:
            print(f"[ok ] {name:12s} {arr.shape}")

    # nicheformer (special folder convention)
    niche_folder = output_root / f"nicheformer_{sid}_test"
    arr = _try_load_nicheformer(niche_folder, order)
    if arr is None:
        if verbose:
            print(f"[skip] nicheformer: not found in {niche_folder}")
    else:
        embeddings["nicheformer"] = arr
        if verbose:
            print(f"[ok ] nicheformer  {arr.shape}")

    # HVG-PCA (computed on the fly)
    if use_hvg_pca:
        a_hvg = a_raw_o.copy()
        sc.pp.highly_variable_genes(a_hvg, n_top_genes=n_hvg,
                                    flavor="seurat_v3", subset=True)
        sc.pp.normalize_total(a_hvg, target_sum=1e4)
        sc.pp.log1p(a_hvg)
        Xhvg = a_hvg.X.toarray() if hasattr(a_hvg.X, "toarray") else np.asarray(a_hvg.X)
        embeddings["hvg_pca"] = PCA(n_components=n_pca, random_state=0)\
            .fit_transform(Xhvg).astype(np.float32)
        if verbose:
            print(f"[ok ] hvg_pca      {embeddings['hvg_pca'].shape}")

    # z-score each embedding per feature
    embeddings = {n: _zscore(E) for n, E in embeddings.items()}

    # Stable ordering
    canonical_order = [
        "hvg_pca", "scgpt", "nicheformer", "loki_text",
        "loki_image", "phikon_v2", "uni2_h", "stpath",
    ]
    emb_names = [n for n in canonical_order if n in embeddings]
    emb_dims = [embeddings[n].shape[1] for n in emb_names]
    if verbose:
        print(f"\n[done] {len(emb_names)} embeddings: {dict(zip(emb_names, emb_dims))}")

    return dict(
        X_raw=X_hvg,
        hvg_genes=hvg_genes,
        coords=coords,
        ground_truth=gt,
        log_lib=log_lib,
        obs_names=order,
        embeddings=embeddings,
        emb_names=emb_names,
        emb_dims=emb_dims,
    )
