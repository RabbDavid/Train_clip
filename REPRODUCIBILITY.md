# Reproducibility Notes

This repo is intentionally small: code and docs are committed, while datasets, model weights, and run outputs stay local.

## Python Environment

Recommended:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install CUDA PyTorch from the official selector:

```text
https://pytorch.org/get-started/locally/
```

Then install the repo dependencies:

```powershell
python -m pip install -r requirements.txt
```

Check the GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Manual Model Downloads

DFN2B/OpenCLIP:

```text
https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16
MODEL/DFN2B-CLIP-ViT-B-16/
```

StreetCLIP:

```text
https://huggingface.co/geolocal/StreetCLIP
MODEL/StreetCLIP/
```

The scripts use local model paths and do not download weights at runtime.

## Dataset Layout

```text
TRAIN_DATASET/koglab_levi/
  argentina/
  argentina_test/
  ...
```

Folders without `_test` are training images. Folders with `_test` are test images. If no validation folders exist, the scripts create a stratified validation split from the training folders.

## Runs To Report

DFN2B zero-shot:

```powershell
python train_clip_country.py `
  --data-root TRAIN_DATASET/koglab_levi `
  --model-dir MODEL/DFN2B-CLIP-ViT-B-16 `
  --zero-shot-only `
  --batch-size 256 `
  --num-workers 8 `
  --prefetch-factor 4 `
  --amp-dtype bf16
```

DFN2B fine-tune:

```powershell
python train_clip_country.py `
  --data-root TRAIN_DATASET/koglab_levi `
  --model-dir MODEL/DFN2B-CLIP-ViT-B-16 `
  --epochs 12 `
  --batch-size 192 `
  --unfreeze-visual-layers -1 `
  --lr-visual 1e-5 `
  --lr-logit-scale 5e-5 `
  --layer-decay 0.75 `
  --label-smoothing 0.05 `
  --proj-l2 1e-4 `
  --amp-dtype bf16 `
  --num-workers 8 `
  --prefetch-factor 4
```

StreetCLIP fine-tune:

```powershell
python Code/train_streetclip_country.py `
  --data-root TRAIN_DATASET/koglab_levi `
  --model-dir MODEL/StreetCLIP `
  --epochs 8 `
  --batch-size 64 `
  --grad-accum-steps 2 `
  --unfreeze-vision-layers 4 `
  --lr-head 1e-3 `
  --lr-vision 1e-5 `
  --amp-dtype bf16 `
  --num-workers 8 `
  --prefetch-factor 4 `
  --attn-implementation sdpa
```

Save the resulting `config.json`, `epoch_metrics.csv`, `test_report_*.txt`, `test_predictions_*.csv`, and `confusion_matrix_*.csv` files for the final paper.
