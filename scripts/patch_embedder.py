"""
Pathology patch embedder for Visium spots.
==========================================

Extracts H&E patch embeddings around each Visium spot using a configurable
foundation-model backbone. Currently supports:

    - phikon-v2  (Owkin, ViT-L/16, DINOv2, 1024-d, public on HF)
    - phikon     (Owkin v1, ViT-B/16, iBOT, 768-d, public on HF)
    - uni        (Mahmood Lab, ViT-L/16, DINOv2, 1024-d, gated on HF)
    - uni2-h     (Mahmood Lab, ViT-H/14, DINOv2, 1536-d, gated on HF)

For Phikon-v2 / Phikon: no access request needed; just `huggingface-cli login`
once with any HF account.

For UNI / UNI2-h: must register on Hugging Face with INSTITUTIONAL email
(gmail/qq rejected), agree to terms, wait for approval.

Usage
-----
    from patch_embedder import extract_patch_embeddings

    out = extract_patch_embeddings(
        slice_dir="/path/to/151673/",          # contains spatial/ folder
        output_h5ad="...//151673_phikon2.h5ad",
        backbone="phikon-v2",
        patch_size_px=224,                     # patch side at full-res
        device="cuda",
        batch_size=64,
    )

Inputs expected in slice_dir:
    spatial/tissue_hires_image.png            (or tissue_lowres_image.png)
    spatial/scalefactors_json.json
    spatial/tissue_positions.csv              (or tissue_positions_list.csv)

Output:
    AnnData with obs_names = barcode, obsm['X_path_<backbone>'] = (N, d) embeddings,
    obsm['spatial'] = pixel coords. Drop-in for prepare_v4_data.py with
    pathology_obsm_key='X_path_<backbone>'.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import anndata as ad


# =============================================================================
# Backbone loaders
# =============================================================================
def _load_phikon_v2(device):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    mdl = AutoModel.from_pretrained("owkin/phikon-v2").to(device).eval()

    @torch.no_grad()
    def encode(pil_batch):
        # pil_batch: list of PIL Images
        inputs = proc(images=pil_batch, return_tensors="pt").to(device)
        out = mdl(**inputs)
        # CLS token: out.last_hidden_state[:, 0]
        return out.last_hidden_state[:, 0].cpu().numpy()

    return encode, 1024


def _load_phikon_v1(device):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("owkin/phikon")
    mdl = AutoModel.from_pretrained("owkin/phikon").to(device).eval()

    @torch.no_grad()
    def encode(pil_batch):
        inputs = proc(images=pil_batch, return_tensors="pt").to(device)
        out = mdl(**inputs)
        return out.last_hidden_state[:, 0].cpu().numpy()

    return encode, 768


def _load_uni(device):
    """UNI v1 (ViT-L/16)."""
    import timm
    from torchvision import transforms
    from huggingface_hub import login  # noqa: F401  user must be logged in

    mdl = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    ).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    @torch.no_grad()
    def encode(pil_batch):
        x = torch.stack([transform(im) for im in pil_batch]).to(device)
        # timm forward returns features pre-classifier when num_classes=0
        return mdl(x).cpu().numpy()

    return encode, 1024


def _load_uni2h(device):
    """UNI2-h (ViT-H/14, 1536-d). Needs HF approval."""
    import timm
    from torchvision import transforms

    timm_kwargs = dict(
        model_name="vit_giant_patch14_224",
        img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
        num_classes=0, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU, reg_tokens=8,
        dynamic_img_size=True,
    )
    mdl = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True,
                            **{k: v for k, v in timm_kwargs.items()
                               if k != "model_name"}).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    @torch.no_grad()
    def encode(pil_batch):
        x = torch.stack([transform(im) for im in pil_batch]).to(device)
        return mdl(x).cpu().numpy()

    return encode, 1536


_BACKBONES = {
    "phikon-v2": _load_phikon_v2,
    "phikon":    _load_phikon_v1,
    "uni":       _load_uni,
    "uni2-h":    _load_uni2h,
}


# =============================================================================
# Visium spatial helpers
# =============================================================================
def _load_visium_spatial(slice_dir: Path) -> Tuple[Image.Image, pd.DataFrame, dict]:
    """
    Load full-res H&E image (or hires), tissue_positions, and scalefactors.

    Returns: (PIL_image, positions_df, scalefactors_dict)
        positions_df has columns: barcode, in_tissue, array_row, array_col,
                                  pxl_row_in_fullres, pxl_col_in_fullres
        Coordinate system depends on which image we open (we open hires).
    """
    spatial = slice_dir / "spatial"

    sf = json.load(open(spatial / "scalefactors_json.json"))
    # Try hires first, fall back to lowres
    img_path = spatial / "tissue_hires_image.png"
    sf_key = "tissue_hires_scalef"
    if not img_path.exists():
        img_path = spatial / "tissue_lowres_image.png"
        sf_key = "tissue_lowres_scalef"
    img = Image.open(img_path).convert("RGB")

    # tissue_positions.csv (newer) vs tissue_positions_list.csv (legacy)
    tp_path = spatial / "tissue_positions.csv"
    if not tp_path.exists():
        tp_path = spatial / "tissue_positions_list.csv"
        df = pd.read_csv(tp_path, header=None,
                         names=["barcode", "in_tissue",
                                "array_row", "array_col",
                                "pxl_row_in_fullres", "pxl_col_in_fullres"])
    else:
        df = pd.read_csv(tp_path)
        # New format has different column names sometimes
        col_map = {
            "barcode": "barcode",
            "in_tissue": "in_tissue",
            "array_row": "array_row",
            "array_col": "array_col",
            "pxl_row_in_fullres": "pxl_row_in_fullres",
            "pxl_col_in_fullres": "pxl_col_in_fullres",
        }
        df = df.rename(columns=col_map)

    # Convert to image-space coords for the image we actually opened
    scale = sf[sf_key]
    df["pxl_row_in_img"] = (df["pxl_row_in_fullres"] * scale).round().astype(int)
    df["pxl_col_in_img"] = (df["pxl_col_in_fullres"] * scale).round().astype(int)

    # Spot diameter in image pixels (Visium spec: spot_diameter_fullres in sf)
    spot_diam_full = sf.get("spot_diameter_fullres", 100.0)
    df.attrs = {"spot_diameter_in_img_px": spot_diam_full * scale}

    return img, df, sf


def _crop_patch(img: Image.Image, row_px: int, col_px: int, half: int) -> Image.Image:
    """Center-crop a square patch of 2*half pixels around (col_px, row_px)."""
    W, H = img.size
    left = col_px - half
    upper = row_px - half
    right = col_px + half
    lower = row_px + half
    # Clip to image; pad with white if necessary
    if left < 0 or upper < 0 or right > W or lower > H:
        # Pad on the boundary
        pad_l = max(0, -left); pad_t = max(0, -upper)
        pad_r = max(0, right - W); pad_b = max(0, lower - H)
        # paste into white canvas
        canvas = Image.new("RGB", (right - left, lower - upper), color=(255, 255, 255))
        valid = img.crop((max(0, left), max(0, upper),
                          min(W, right), min(H, lower)))
        canvas.paste(valid, (pad_l, pad_t))
        return canvas
    return img.crop((left, upper, right, lower))


# =============================================================================
# Main extractor
# =============================================================================
def extract_patch_embeddings(
    slice_dir,
    output_h5ad: Optional[str] = None,
    backbone: str = "phikon-v2",
    patch_size_px: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    only_in_tissue: bool = True,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Extract H&E embeddings for every Visium spot.

    Parameters
    ----------
    slice_dir : path containing the spatial/ folder
    backbone  : one of {'phikon-v2', 'phikon', 'uni', 'uni2-h'}
    patch_size_px : full crop edge in image pixels. If None, uses 1.0 * spot
                    diameter on the hires image (typical = ~36-50 px), which
                    is then resized by the model preprocessor to 224. Pass
                    a larger number (e.g. 96) if you want to capture context
                    around the spot.

    Returns
    -------
    AnnData with obs_names=barcode, obsm['X_path_<backbone>']=(N, d) and
    obsm['spatial']=(N, 2) pixel coords for the IMAGE used for cropping.
    """
    slice_dir = Path(slice_dir)
    if backbone not in _BACKBONES:
        raise ValueError(f"backbone must be one of {list(_BACKBONES)}, got {backbone}")

    img, df, sf = _load_visium_spatial(slice_dir)
    if verbose:
        print(f"image size: {img.size}    spots: {len(df)}    "
              f"in_tissue: {int(df['in_tissue'].sum())}")

    if only_in_tissue:
        df = df[df["in_tissue"] == 1].reset_index(drop=True)

    if patch_size_px is None:
        patch_size_px = max(48, int(round(df.attrs["spot_diameter_in_img_px"])))
    half = patch_size_px // 2
    if verbose:
        print(f"patch size: {patch_size_px} px (half={half})")

    if verbose:
        print(f"loading backbone: {backbone} ...")
    encode_fn, emb_dim = _BACKBONES[backbone](device)
    if verbose:
        print(f"backbone loaded; embedding dim = {emb_dim}")

    N = len(df)
    feats = np.empty((N, emb_dim), dtype=np.float32)

    # Mini-batch over spots
    batch_imgs, batch_idx = [], []
    for i in range(N):
        row = df.iloc[i]
        patch = _crop_patch(img, int(row["pxl_row_in_img"]),
                            int(row["pxl_col_in_img"]), half)
        batch_imgs.append(patch); batch_idx.append(i)
        if len(batch_imgs) >= batch_size:
            f = encode_fn(batch_imgs)
            feats[batch_idx] = f
            batch_imgs, batch_idx = [], []
            if verbose and (i % (batch_size * 10) == 0):
                print(f"  encoded {i + 1}/{N}")
    if batch_imgs:
        f = encode_fn(batch_imgs)
        feats[batch_idx] = f

    obsm_key = f"X_path_{backbone.replace('-', '_')}"
    a = ad.AnnData(
        X=np.zeros((N, 1), dtype=np.float32),
        obs=pd.DataFrame({
            "in_tissue": df["in_tissue"].values,
            "array_row": df["array_row"].values,
            "array_col": df["array_col"].values,
        }),
    )
    a.obs_names = df["barcode"].astype(str).values
    a.obsm[obsm_key] = feats
    a.obsm["spatial"] = np.stack(
        [df["pxl_col_in_img"].values, df["pxl_row_in_img"].values],
        axis=-1,
    ).astype(np.float32)
    a.uns["patch_embedder"] = {
        "backbone": backbone,
        "patch_size_px": int(patch_size_px),
        "embedding_dim": int(emb_dim),
        "image_used": "hires" if (slice_dir / "spatial" / "tissue_hires_image.png").exists() else "lowres",
    }

    if output_h5ad:
        a.write_h5ad(output_h5ad)
        if verbose:
            print(f"saved: {output_h5ad}  shape={feats.shape}")
    return a
