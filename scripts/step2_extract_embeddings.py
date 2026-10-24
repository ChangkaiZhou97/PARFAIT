#!/usr/bin/env python3
"""
Step 2: Per-slice batch FM-embedding extraction.

For each slice, checks whether each of the 7 FM embeddings is already present.
If not, extracts it. Carefully manages GPU memory: each backbone is loaded,
used, then deleted before the next backbone is loaded. The 7 backbones are:

    scgpt           (-> output/DLPFC_<sid>_scgpt_with_spatial/embeddings.h5ad)
    loki_text       (-> output/DLPFC_<sid>_loki_text/embeddings.h5ad)
    loki_image      (-> output/DLPFC_<sid>_loki_image_10x/embeddings.h5ad)
    phikon-v2       (-> output/DLPFC_<sid>_phikon_v2/embeddings.h5ad)
    uni2-h          (-> output/DLPFC_<sid>_uni2-h/embeddings.h5ad)
    nicheformer     (-> output/nicheformer_<sid>_test/embeddings.{h5ad,npy})
    stpath          (-> output/DLPFC_<sid>_stpath/embeddings.h5ad)

Pre-requisites:
    - <repo>/data/DLPFC_local/<sid>/<sid>_with_gt.h5ad  (from step1_combine_gt.py)
    - <repo>/data/DLPFC_local/<sid>/spatial/             (spaceranger output)
    - <repo>/data/DLPFC_local/ensembl_to_symbol_<sid>_mygene.csv  (or shared one)
      Falls back to using 151673's mapping if per-slice one is missing.

STPath additionally requires:
    - <repo>/model_weights/stpath/stfm.pth                 (~190 MB, from HF)
    - <repo>/model_weights/stpath/symbol2ensembl.json      (38,984 entries)
    - HuggingFace login for prov-gigapath (gated repo) — accept the EULA at
      https://huggingface.co/prov-gigapath/prov-gigapath and run
      `huggingface-cli login` (or set HF_TOKEN). The Gigapath tile features
      are computed inline on the first STPath run and cached to
      <output>/gigapath_features.h5 for reuse.
"""
from __future__ import annotations
import argparse, gc, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


ALL_SLICES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]


# ===================================================================
# Helpers
# ===================================================================
def gpu_cleanup():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_ensembl_mapping(slice_dir: Path, repo: Path, sid: str) -> Path:
    """
    Look for Ensembl->symbol mapping CSV. Try slice-specific first, fall back
    to a shared mapping (e.g. the one made for 151673).
    """
    candidates = [
        slice_dir.parent / f"ensembl_to_symbol_{sid}_mygene.csv",
        slice_dir.parent / f"ensembl_to_symbol_151673_mygene.csv",
        repo / "data" / "DLPFC_local" / f"ensembl_to_symbol_{sid}_mygene.csv",
        repo / "data" / "DLPFC_local" / f"ensembl_to_symbol_151673_mygene.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def prepare_scgpt_input(adata_10x: sc.AnnData, mapping_csv: Path) -> sc.AnnData:
    """Create scGPT-style input: var has ensembl_id and feature_name columns."""
    adata = adata_10x.copy()
    adata.var_names_make_unique()

    mapping = pd.read_csv(mapping_csv)
    # accept either column naming
    if "ensembl_clean" not in mapping.columns:
        mapping["ensembl_clean"] = mapping.get("ensembl_id", mapping.iloc[:, 0])

    # If mapping is per the 33538 reference geneset and matches adata.var_names,
    # use directly. Otherwise build a lookup:
    if mapping.shape[0] == adata.n_vars:
        # row-aligned mapping
        symbols = (
            mapping["symbol"].fillna(mapping["ensembl_clean"]).astype(str).values
        )
    else:
        # fallback: use Ensembl IDs from var.gene_ids if available
        if "gene_ids" in adata.var.columns:
            ens_ids = adata.var["gene_ids"].astype(str).values
        else:
            ens_ids = adata.var_names.astype(str).values
        ens_clean = pd.Series(ens_ids).str.replace(r"\.\d+$", "", regex=True).values
        lookup = {}
        for _, row in mapping.iterrows():
            k = str(row.get("ensembl_clean", row.iloc[0]))
            v = row.get("symbol")
            if pd.notna(v):
                lookup[k] = str(v)
        symbols = np.array(
            [lookup.get(e, e) for e in ens_clean], dtype=object
        )

    adata.var["ensembl_id"] = adata.var_names.astype(str)
    adata.var["feature_name"] = symbols.astype(str)

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    return adata


def prepare_loki_input(adata_10x: sc.AnnData, mapping_csv: Optional[Path]) -> sc.AnnData:
    """
    For Loki: var_names must be gene symbols (Loki reads var_names directly).
    If 10x var_names are already gene symbols (not Ensembl), keep as-is.
    """
    adata = adata_10x.copy()
    adata.var_names_make_unique()

    fraction_ensg = adata.var_names.astype(str).str.startswith("ENSG").mean()
    if fraction_ensg < 0.05:
        # already gene symbols (10x default)
        adata.var["feature_name"] = adata.var_names.astype(str)
        adata.var["ensembl_id"] = (
            adata.var["gene_ids"].astype(str)
            if "gene_ids" in adata.var.columns
            else adata.var_names.astype(str)
        )
    else:
        # need to remap
        if mapping_csv is None or not mapping_csv.exists():
            raise RuntimeError("var_names are Ensembl but no mapping CSV available")
        mapping = pd.read_csv(mapping_csv)
        if mapping.shape[0] == adata.n_vars:
            symbols = (
                mapping["symbol"].fillna(mapping["ensembl_clean"]).astype(str).values
            )
        else:
            ens_clean = (
                pd.Series(adata.var_names.astype(str))
                .str.replace(r"\.\d+$", "", regex=True).values
            )
            lookup = {str(r["ensembl_clean"]): str(r["symbol"])
                      for _, r in mapping.iterrows() if pd.notna(r.get("symbol"))}
            symbols = np.array([lookup.get(e, e) for e in ens_clean], dtype=object)
        adata.var["ensembl_id"] = adata.var_names.astype(str)
        adata.var_names = pd.Index(symbols.astype(str))
        adata.var_names_make_unique()
        adata.var["feature_name"] = adata.var_names.astype(str)

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    return adata


# ===================================================================
# Per-backbone runners
# ===================================================================
def run_scgpt(repo: Path, sid: str, slice_dir: Path) -> bool:
    out = repo / "output" / f"DLPFC_{sid}_scgpt_with_spatial"
    if (out / "embeddings.h5ad").exists():
        print(f"    [scgpt] exists, skip")
        return True

    sys.path.insert(0, str(repo))
    from src.adapters.scgpt_spatial import run as run_scgpt_adapter
    import torch

    mapping_csv = find_ensembl_mapping(slice_dir, repo, sid)
    if mapping_csv is None:
        print(f"    [scgpt] FAIL: no Ensembl->symbol mapping CSV available")
        return False

    visium = sc.read_h5ad(str(slice_dir / f"{sid}_with_gt.h5ad"))
    adata_in = prepare_scgpt_input(visium, mapping_csv)
    in_path = slice_dir / f"{sid}_scgpt_input.h5ad"
    adata_in.write_h5ad(in_path)

    try:
        run_scgpt_adapter(
            input_path=str(in_path),
            output_dir=str(out),
            model_dir=str(repo / "model_weights" / "scgpt_spatial"),
            gene_col="feature_name",
            max_length=1200,
            batch_size=16,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return True
    except Exception as e:
        print(f"    [scgpt] FAIL: {e}")
        return False
    finally:
        try:
            in_path.unlink()
        except Exception:
            pass


def run_loki(repo: Path, sid: str, slice_dir: Path,
             text_only: bool, output_subdir: str) -> bool:
    out = repo / "output" / output_subdir
    if (out / "embeddings.h5ad").exists():
        print(f"    [loki {'text' if text_only else 'image'}] exists, skip")
        return True

    sys.path.insert(0, str(repo))
    from src.adapters.loki import run as run_loki_adapter
    import torch

    mapping_csv = find_ensembl_mapping(slice_dir, repo, sid)
    visium = sc.read_h5ad(str(slice_dir / f"{sid}_with_gt.h5ad"))
    adata_in = prepare_loki_input(visium, mapping_csv)
    in_path = slice_dir / f"{sid}_loki_input.h5ad"
    adata_in.write_h5ad(in_path)

    try:
        run_loki_adapter(
            input_path=str(in_path),
            output_dir=str(out),
            model_dir=str(repo / "model_weights" / "loki"),
            spatial_dir=None if text_only else str(slice_dir / "spatial"),
            housekeeping_genes_path=None,
            library_id=None if text_only else sid,
            patch_size=16,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return True
    except Exception as e:
        print(f"    [loki] FAIL: {e}")
        return False
    finally:
        try:
            in_path.unlink()
        except Exception:
            pass


def run_pathology_fm(repo: Path, sid: str, slice_dir: Path, backbone: str,
                     output_subdir: str, patch_size_px: int = 96,
                     batch_size: int = 32) -> bool:
    """backbone: 'phikon-v2' or 'uni2-h'."""
    out = repo / "output" / output_subdir
    out.mkdir(parents=True, exist_ok=True)
    out_h5 = out / "embeddings.h5ad"
    if out_h5.exists():
        print(f"    [{backbone}] exists, skip")
        return True

    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "PoE"))
    try:
        from patch_embedder import extract_patch_embeddings
    except ImportError as e:
        print(f"    [{backbone}] FAIL: cannot import patch_embedder ({e})")
        return False

    try:
        extract_patch_embeddings(
            slice_dir=str(slice_dir),
            output_h5ad=str(out_h5),
            backbone=backbone,
            patch_size_px=patch_size_px,
            batch_size=batch_size,
        )
        return True
    except Exception as e:
        print(f"    [{backbone}] FAIL: {e}")
        return False


def run_nicheformer(repo: Path, sid: str, slice_dir: Path,
                    batch_size: int = 8) -> bool:
    out_dir = repo / "output" / f"nicheformer_{sid}_test"
    if (out_dir / "embeddings.h5ad").exists() or (out_dir / "embeddings.npy").exists():
        print(f"    [nicheformer] exists, skip")
        return True

    in_path = slice_dir / f"{sid}_with_gt.h5ad"
    if not in_path.exists():
        print(f"    [nicheformer] FAIL: {in_path} not found")
        return False

    embed_py = repo / "src" / "embed.py"
    if not embed_py.exists():
        print(f"    [nicheformer] FAIL: {embed_py} not found")
        return False

    cmd = [
        sys.executable, str(embed_py),
        "--model", "nicheformer",
        "--input", str(in_path),
        "--output", str(out_dir),
        "--model-dir", str(repo / "model_weights" / "nicheformer"),
        "--technology", "dissociated",
        "--batch-size", str(batch_size),
    ]
    print(f"    [nicheformer] running subprocess: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                              cwd=str(repo))
        if proc.returncode != 0:
            print(f"    [nicheformer] FAIL (rc={proc.returncode}):")
            print(f"      stdout tail: {proc.stdout[-500:]}")
            print(f"      stderr tail: {proc.stderr[-500:]}")
            return False
        return True
    except Exception as e:
        print(f"    [nicheformer] FAIL: {e}")
        return False


def run_stpath(repo: Path, sid: str, slice_dir: Path,
               organ_type: str = "Brain",
               tech_type: str = "Visium",
               gigapath_batch_size: int = 32,
               gigapath_precision: str = "fp32",
               save_imputed_expression: bool = False) -> bool:
    """
    STPath full-context joint embedding (H&E + transcriptome + coords + organ/tech).

    The adapter:
      - Reads HGNC gene symbols from var_names (uses prepare_loki_input).
      - Reads raw counts from .X (applies log1p internally; do NOT pre-normalize).
      - Reads spatial coords from --spatial-dir (Visium tissue_positions.csv).
      - Computes Gigapath tile features inline on first run and caches them
        to <out>/gigapath_features.h5 — subsequent runs reuse the cache.
      - Writes 512-d embeddings to obsm["X_stpath"] and to embeddings.{h5ad,npy,csv,tsv}.

    DLPFC slices are dorsolateral prefrontal cortex → organ_type="Brain",
    tech_type="Visium".
    """
    out = repo / "output" / f"DLPFC_{sid}_stpath"
    if (out / "embeddings.h5ad").exists():
        print(f"    [stpath] exists, skip")
        return True

    sys.path.insert(0, str(repo))
    from src.adapters.stpath import run as run_stpath_adapter
    import torch

    mapping_csv = find_ensembl_mapping(slice_dir, repo, sid)
    visium = sc.read_h5ad(str(slice_dir / f"{sid}_with_gt.h5ad"))
    # Reuse prepare_loki_input: same requirement (HGNC symbols in var_names,
    # raw counts in .X, sparse CSR). STPath's adapter handles log1p itself.
    adata_in = prepare_loki_input(visium, mapping_csv)
    in_path = slice_dir / f"{sid}_stpath_input.h5ad"
    adata_in.write_h5ad(in_path)

    try:
        run_stpath_adapter(
            input_path=str(in_path),
            output_dir=str(out),
            model_dir=str(repo / "model_weights" / "stpath"),
            spatial_dir=str(slice_dir / "spatial"),
            organ_type=organ_type,
            tech_type=tech_type,
            gigapath_batch_size=gigapath_batch_size,
            gigapath_precision=gigapath_precision,
            save_imputed_expression=save_imputed_expression,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return True
    except Exception as e:
        print(f"    [stpath] FAIL: {e}")
        return False
    finally:
        try:
            in_path.unlink()
        except Exception:
            pass


# ===================================================================
# Main
# ===================================================================
BACKBONES = ["scgpt", "loki_text", "loki_image", "phikon_v2", "uni2_h",
             "nicheformer", "stpath"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str,
                   default="/work/cxz220015/Anaconda/LLM/v1_260425/spatialFM-embedding")
    p.add_argument("--slices", nargs="+", default=ALL_SLICES)
    p.add_argument("--backbones", nargs="+", default=BACKBONES,
                   choices=BACKBONES, help="subset of backbones to run")

    p.add_argument("--phikon_patch", type=int, default=96)
    p.add_argument("--phikon_batch", type=int, default=32)
    p.add_argument("--uni_patch", type=int, default=96)
    p.add_argument("--uni_batch", type=int, default=16,
                   help="UNI2-h is much larger; smaller batch helps memory")
    p.add_argument("--niche_batch", type=int, default=8)

    # STPath-specific knobs. DLPFC is dorsolateral prefrontal cortex (Brain)
    # measured with Visium. Gigapath features are computed inline on first run
    # and cached to <out>/gigapath_features.h5; reused thereafter.
    p.add_argument("--stpath_organ", type=str, default="Brain",
                   help="STPath organ token (DLPFC: 'Brain')")
    p.add_argument("--stpath_tech", type=str, default="Visium",
                   help="STPath technology token ('Visium' for 10x Visium)")
    p.add_argument("--stpath_gigapath_batch", type=int, default=32,
                   help="batch size for inline Gigapath feature extraction")
    p.add_argument("--stpath_gigapath_precision", type=str, default="fp32",
                   choices=["fp32", "fp16"],
                   help="precision for inline Gigapath inference; fp16 halves "
                        "Gigapath VRAM but a sidecar keeps the precision it "
                        "was first written with")
    p.add_argument("--stpath_save_imputed", action="store_true",
                   help="also write imputed_expression.h5ad (38,984-gene "
                        "STPath vocab) alongside embeddings.h5ad")
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo)
    print(f"=== Step 2: extract FM embeddings ===")
    print(f"repo: {repo}")
    print(f"slices: {args.slices}")
    print(f"backbones: {args.backbones}\n")

    overall = []
    for sid in args.slices:
        print(f"\n{'='*70}\n=== slice {sid} ===\n{'='*70}")
        slice_dir = repo / "data" / "DLPFC_local" / sid
        gt_path = slice_dir / f"{sid}_with_gt.h5ad"
        if not gt_path.exists():
            print(f"  [skip {sid}] missing {gt_path.name}; run step1 first")
            overall.append({"sid": sid, "status": "skipped_no_gt"})
            continue

        per_bb = {"sid": sid}

        # Order: cheaper / smaller-memory first; pathology FMs last.
        if "scgpt" in args.backbones:
            print(f"\n  --- scgpt ---")
            t0 = time.time()
            ok = run_scgpt(repo, sid, slice_dir)
            per_bb["scgpt"] = ("ok" if ok else "fail") + f" ({time.time()-t0:.0f}s)"
            gpu_cleanup()

        if "loki_text" in args.backbones:
            print(f"\n  --- loki_text ---")
            t0 = time.time()
            ok = run_loki(repo, sid, slice_dir, text_only=True,
                          output_subdir=f"DLPFC_{sid}_loki_text")
            per_bb["loki_text"] = ("ok" if ok else "fail") + f" ({time.time()-t0:.0f}s)"
            gpu_cleanup()

        if "loki_image" in args.backbones:
            print(f"\n  --- loki_image ---")
            t0 = time.time()
            ok = run_loki(repo, sid, slice_dir, text_only=False,
                          output_subdir=f"DLPFC_{sid}_loki_image_10x")
            per_bb["loki_image"] = ("ok" if ok else "fail") + f" ({time.time()-t0:.0f}s)"
            gpu_cleanup()

        if "nicheformer" in args.backbones:
            print(f"\n  --- nicheformer ---")
            t0 = time.time()
            ok = run_nicheformer(repo, sid, slice_dir, batch_size=args.niche_batch)
            per_bb["nicheformer"] = ("ok" if ok else "fail") + f" ({time.time()-t0:.0f}s)"
            gpu_cleanup()

        if "phikon_v2" in args.backbones:
            print(f"\n  --- phikon_v2 ---")
            t0 = time.time()
            ok = run_pathology_fm(repo, sid, slice_dir, "phikon-v2",
                                  f"DLPFC_{sid}_phikon_v2",
                                  patch_size_px=args.phikon_patch,
                                  batch_size=args.phikon_batch)
            per_bb["phikon_v2"] = ("ok" if ok else "fail") + f" ({time.time()-t0:.0f}s)"
            gpu_cleanup()

        if "uni2_h" in args.backbones:
            print(f"\n  --- uni2_h ---")
            t0 = time.time()
            ok = run_pathology_fm(repo, sid, slice_dir, "uni2-h",
                                  f"DLPFC_{sid}_uni2-h",
                                  patch_size_px=args.uni_patch,
                                  batch_size=args.uni_batch)
            per_bb["uni2_h"] = ("ok" if ok else "fail") + f" ({time.time()-t0:.0f}s)"
            gpu_cleanup()

        if "stpath" in args.backbones:
            # Run last: Gigapath (~1.5B params) + STFM is the heaviest backbone
            # in terms of VRAM and disk (~190 MB STFM weights, gated Gigapath
            # download from HuggingFace).
            print(f"\n  --- stpath ---")
            t0 = time.time()
            ok = run_stpath(
                repo, sid, slice_dir,
                organ_type=args.stpath_organ,
                tech_type=args.stpath_tech,
                gigapath_batch_size=args.stpath_gigapath_batch,
                gigapath_precision=args.stpath_gigapath_precision,
                save_imputed_expression=args.stpath_save_imputed,
            )
            per_bb["stpath"] = ("ok" if ok else "fail") + f" ({time.time()-t0:.0f}s)"
            gpu_cleanup()

        overall.append(per_bb)

    print(f"\n\n=== summary ===")
    df = pd.DataFrame(overall)
    print(df.to_string(index=False))
    out_csv = repo / "output" / "embedding_extraction_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved summary to {out_csv}")


if __name__ == "__main__":
    main()
