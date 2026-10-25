# PARFAIT

This repository contains a DLPFC workflow for training PARFAIT on
spatial foundation-model embeddings and benchmarking embeddings with Leiden
clustering.

## Files

- `spatialFM-embedding/` (*Git submodule*): external submodule linking to [`Qingtian-Zou/spatialFM-embedding`](https://github.com/Qingtian-Zou/spatialFM-embedding), providing the dedicated pipeline and model wrappers for extracting spatial foundation-model embeddings.
- `scripts/parfait_model.py`: PARFAIT model definition.
- `scripts/prepare_data.py`: helper imported by `run_parfait.py`; loads raw DLPFC
  h5ad files, aligns spot order, loads precomputed embeddings, and computes the
  `hvg_pca` baseline.
- `scripts/run_parfait.py`: trains PARFAIT on one slice, evaluates the `(t, s)`
  grid, and writes `grid_metrics.csv` / `best_metrics.csv`.
- `scripts/benchmark_embeddings.py`: clusters each embedding with Leiden across
  DLPFC samples and reports ARI/NMI.
- `scripts/step1_combine_gt.py`: optional preprocessing from 10x counts plus
  reference labels to `<sid>_with_gt.h5ad`.
- `scripts/step2_extract_embeddings.py`: optional batch extraction of source
  foundation-model embeddings.
- `scripts/patch_embedder.py`: helper used by `step2_extract_embeddings.py` for
  Phikon/UNI pathology image embeddings.

## Install

```bash
pip install -r requirements.txt
```

For GPU runs, install the PyTorch build that matches your CUDA version before
running PARFAIT.

Embedding extraction has extra model-specific requirements and pretrained
weights. See "Extract Embeddings" below.

## Data Layout

Raw DLPFC files are expected under:

```text
data/DLPFC_local/<sid>/<sid>_with_gt.h5ad
data/DLPFC_local/<sid>/spatial/
```

If using the provided `DLPFC_local.zip`, unzip it into `data/DLPFC_local`:

```bash
mkdir -p data/DLPFC_local
unzip data/DLPFC_local.zip -d data/DLPFC_local
```

After extraction, each slice should look like:

```text
data/DLPFC_local/151673/
  151673_with_gt.h5ad
  filtered_feature_bc_matrix.h5
  spatial/
```

`filtered_feature_bc_matrix.h5` is only needed if you want to regenerate
`<sid>_with_gt.h5ad` with `step1_combine_gt.py`.

## Optional: Build `*_with_gt.h5ad`

Skip this step if `data/DLPFC_local/<sid>/<sid>_with_gt.h5ad` already exists.

```bash
python scripts/step1_combine_gt.py \
  --repo . \
  --sce_root data/DLPFC_local \
  --slices 151673
```

`step1_combine_gt.py` combines:

```text
data/DLPFC_local/<sid>/filtered_feature_bc_matrix.h5
data/DLPFC_local/SCE_<sid>.h5ad
```

and writes:

```text
data/DLPFC_local/<sid>/<sid>_with_gt.h5ad
```

## Extract Embeddings

PARFAIT expects precomputed embeddings under `output/`. The default layout is:

```text
output/DLPFC_<sid>_scgpt_with_spatial/embeddings.h5ad
output/DLPFC_<sid>_loki_text/embeddings.h5ad
output/DLPFC_<sid>_loki_image_10x/embeddings.h5ad
output/DLPFC_<sid>_phikon_v2/embeddings.h5ad
output/DLPFC_<sid>_uni2-h/embeddings.h5ad
output/DLPFC_<sid>_stpath/embeddings.h5ad
output/nicheformer_<sid>_test/embeddings.h5ad
```

To extract embeddings for one slice:

```bash
python scripts/step2_extract_embeddings.py \
  --repo . \
  --slices 151673
```

To extract only selected embeddings:

```bash
python scripts/step2_extract_embeddings.py \
  --repo . \
  --slices 151673 \
  --backbones scgpt loki_text loki_image phikon_v2 uni2_h nicheformer stpath
```

`step2_extract_embeddings.py` skips embeddings that already exist.

Important prerequisites:

- The upstream embedding adapters must be available under `src/`.
- Pretrained weights must be placed under `model_weights/`. For ST foundation models, see [Qingtian-Zou/spatialFM-embedding: Extract embeddings from spatial transcriptomics foundation models.](https://github.com/Qingtian-Zou/spatialFM-embedding). Phikon-v2: [owkin/phikon-v2 · Hugging Face](https://huggingface.co/owkin/phikon-v2). UNI2-h: [MahmoodLab/UNI2-h · Hugging Face](https://huggingface.co/MahmoodLab/UNI2-h). 
- scGPT-spatial expects `model_weights/scgpt_spatial/`.
- Loki expects `model_weights/loki/`.
- Nicheformer expects `model_weights/nicheformer/`.
- STPath expects `model_weights/stpath/stfm.pth` and
  `model_weights/stpath/symbol2ensembl.json`.
- Phikon/UNI extraction uses Hugging Face models through `patch_embedder.py`;
  UNI/UNI2-h may require Hugging Face access approval.  

## Train One Slice

Fast smoke test:

```bash
python scripts/run_parfait.py \
  --repo . \
  --sid 151673 \
  --seed 0 \
  --K 32 \
  --n_steps 200 \
  --n_t 5 \
  --n_s 5 \
  --no_plots \
  --output_root output/DLPFC_parfait_finegrid_smoke
```

Full PARFAIT run with `K=32`:

```bash
python scripts/run_parfait.py \
  --repo . \
  --sid 151673 \
  --seed 0 \
  --K 32 \
  --n_hvg 2000 \
  --n_neighbors_spatial 6 \
  --K_diff 4 \
  --alpha_max 0.1 \
  --lambda_spatial 0.0 \
  --lambda_path 0.0 \
  --beta_strength_t 4.0 \
  --beta_strength_s 4.0 \
  --n_steps 10000 \
  --lr 1e-3 \
  --kl_warmup_steps 1000 \
  --kl_ramp_steps 2000 \
  --n_t 21 \
  --n_s 21 \
  --leiden_n_neighbors 15 \
  --leiden_resolution 0.5 \
  --leiden_seed 0 \
  --output_root output/DLPFC_parfait_finegrid
```

Main per-run outputs:

```text
output/DLPFC_parfait_finegrid/<sid>/seed_00/
  grid_metrics.csv
  best_metrics.csv
  t_s_table.csv
  rayleigh_table.csv
  run_summary.txt
```

## Aggregate PARFAIT Runs

After running multiple slices or seeds, aggregate existing PARFAIT outputs:

```bash
python scripts/run_parfait.py \
  --repo . \
  --aggregate_only \
  --output_root output/DLPFC_parfait_finegrid
```

Aggregate outputs are written to:

```text
output/DLPFC_parfait_finegrid/aggregate/
```

## Benchmark Embeddings

Run Leiden clustering on raw/source embeddings and compare clusters with DLPFC
manual layer labels:

```bash
python scripts/benchmark_embeddings.py \
  --repo . \
  --samples all \
  --embedding_root output \
  --parfait_root output/DLPFC_parfait_finegrid \
  --no_include_traditional_baselines \
  --output_dir output/DLPFC_embedding_ari_nmi
```

Main benchmark outputs:

```text
output/DLPFC_embedding_ari_nmi/dlpfc_embedding_ari_nmi_long.csv
output/DLPFC_embedding_ari_nmi/dlpfc_embedding_ari_nmi_summary.csv
output/DLPFC_embedding_ari_nmi/dlpfc_embedding_ari_nmi_report.md
```
