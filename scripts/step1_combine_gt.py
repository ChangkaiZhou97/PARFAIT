#!/usr/bin/env python3
"""
Step 1: Combine ground-truth labels from SCE_<sid>.h5ad into the Visium
filtered_feature_bc_matrix.h5 to produce data/DLPFC_local/<sid>/<sid>_with_gt.h5ad.

Mirrors what your test notebook did for slice 151673 (cells 17-21):
    - sc.read_visium(...)  to load Visium with hires image + spatial/
    - read SCE_<sid>.h5ad which has obs['z'] = layer label
    - merge by barcode intersection
    - write back into the slice folder
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


ALL_SLICES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str,
                   default="/work/cxz220015/Anaconda/LLM/v1_260425/spatialFM-embedding")
    p.add_argument("--sce_root", type=str,
                   default="/groups/qiwei/changkai/data/DLPFC_10X/data",
                   help="dir with SCE_<sid>.h5ad files")
    p.add_argument("--slices", nargs="+", default=ALL_SLICES)
    p.add_argument("--overwrite", action="store_true",
                   help="re-make even if <sid>_with_gt.h5ad already exists")
    return p.parse_args()


def make_slice_with_gt(sid: str, repo: Path, sce_root: Path,
                       overwrite: bool = False) -> bool:
    """
    Produce data/DLPFC_local/<sid>/<sid>_with_gt.h5ad.
    Returns True on success, False on skip.
    """
    visium_dir = repo / "data" / "DLPFC_local" / sid
    out_path = visium_dir / f"{sid}_with_gt.h5ad"
    if out_path.exists() and not overwrite:
        print(f"  [skip {sid}] {out_path.name} already exists (use --overwrite)")
        return False

    if not visium_dir.exists():
        print(f"  [skip {sid}] visium dir missing: {visium_dir}")
        return False
    if not (visium_dir / "filtered_feature_bc_matrix.h5").exists():
        print(f"  [skip {sid}] missing filtered_feature_bc_matrix.h5")
        return False
    if not (visium_dir / "spatial").exists():
        print(f"  [skip {sid}] missing spatial/ folder")
        return False

    sce_path = sce_root / f"SCE_{sid}.h5ad"
    if not sce_path.exists():
        # Try alternate naming
        alt = sce_root / f"{sid}.h5ad"
        if alt.exists():
            sce_path = alt
        else:
            print(f"  [skip {sid}] missing SCE label file at {sce_path}")
            return False

    # Load Visium
    adata_10x = sc.read_visium(
        path=str(visium_dir),
        count_file="filtered_feature_bc_matrix.h5",
        library_id=sid,
        load_images=True,
    )
    adata_10x.var_names_make_unique()

    # Load SCE labels
    old = sc.read_h5ad(str(sce_path))
    common = adata_10x.obs_names.intersection(old.obs_names)
    print(f"  [{sid}] common barcodes: {len(common)} / {adata_10x.n_obs}")

    if len(common) == 0:
        print(f"  [skip {sid}] zero barcode intersection between Visium and SCE")
        return False

    # Merge ground_truth from SCE z column
    if "z" not in old.obs.columns:
        print(f"  [skip {sid}] SCE file has no obs['z'] column; "
              f"available: {old.obs.columns.tolist()}")
        return False

    adata_10x.obs["ground_truth"] = pd.NA
    adata_10x.obs.loc[common, "ground_truth"] = old.obs.loc[common, "z"].astype(str)
    adata_10x.obs["ground_truth"] = adata_10x.obs["ground_truth"].astype("category")

    counts = adata_10x.obs["ground_truth"].value_counts(dropna=False)
    print(f"  [{sid}] gt counts: {dict(counts)}")

    adata_10x.write_h5ad(out_path, compression="gzip")
    print(f"  [{sid}] saved: {out_path}")
    return True


def main():
    args = parse_args()
    repo = Path(args.repo)
    sce_root = Path(args.sce_root)

    print(f"=== Step 1: combine GT labels ===")
    print(f"repo:     {repo}")
    print(f"sce_root: {sce_root}")
    print(f"slices:   {args.slices}\n")

    succeeded = []
    for sid in args.slices:
        print(f"--- {sid} ---")
        if make_slice_with_gt(sid, repo, sce_root, overwrite=args.overwrite):
            succeeded.append(sid)

    print(f"\n=== done ===")
    print(f"  succeeded: {len(succeeded)}/{len(args.slices)}: {succeeded}")


if __name__ == "__main__":
    main()
