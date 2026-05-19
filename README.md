# CLIP Country Classifier

Fine-tune an OpenCLIP model to classify Google Street View images by country.

## Folder Layout

Put the Google Drive image downloads here after cloning:

```text
Train_clip/
  train_clip_country.py
  requirements.txt
  Code/
    train_streetclip_country.py
    README_StreetCLIP.md
  MODEL/
    DFN2B-CLIP-ViT-B-16/
      open_clip_config.json
      open_clip_pytorch_model.bin
  TRAIN_DATASET/
    koglab_levi/
      argentina/
      argentina_test/
      australia/
      australia_test/
      ...
      usa/
      usa_test/
```

Folders without `_test` are training images. Folders with `_test` are test images.

Do not commit `MODEL/`, `TRAIN_DATASET/`, `data/`, `runs_clip/`, or model checkpoints.

## Install

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install PyTorch for CUDA from the official selector:

```text
https://pytorch.org/get-started/locally/
```

For a modern NVIDIA GPU, this is likely:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install the rest:

```bash
pip install -r requirements.txt
```

Check the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Model

Download the CLIP model manually from:

```text
https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16
```

Put it here:

```text
Train_clip/MODEL/DFN2B-CLIP-ViT-B-16/
```

The folder must contain:

```text
MODEL/DFN2B-CLIP-ViT-B-16/open_clip_config.json
MODEL/DFN2B-CLIP-ViT-B-16/open_clip_pytorch_model.bin
```

The Python script loads this local folder and does not download the model.

## Zero-Shot Baseline

```bash
python train_clip_country.py \
  --data-root TRAIN_DATASET/koglab_levi \
  --model-dir MODEL/DFN2B-CLIP-ViT-B-16 \
  --zero-shot-only \
  --batch-size 256 \
  --num-workers 8 \
  --amp-dtype bf16
```

## Main Fine-Tune

```bash
python train_clip_country.py \
  --data-root TRAIN_DATASET/koglab_levi \
  --model-dir MODEL/DFN2B-CLIP-ViT-B-16 \
  --epochs 12 \
  --batch-size 192 \
  --unfreeze-visual-layers -1 \
  --lr-visual 1e-5 \
  --lr-logit-scale 5e-5 \
  --layer-decay 0.75 \
  --label-smoothing 0.05 \
  --proj-l2 1e-4 \
  --amp-dtype bf16 \
  --num-workers 8
```

Outputs are saved to:

```text
runs_clip/<timestamp>/
```

Important output files:

```text
best.pt
last.pt
test_report_clip.txt
test_predictions_detailed_clip.csv
wiseft_metrics.csv
config.json
```

## StreetCLIP Version

StreetCLIP is the geolocation-specialized challenger model.

Manual download page:

```text
https://huggingface.co/geolocal/StreetCLIP
```

Put it here:

```text
Train_clip/MODEL/StreetCLIP/
```

Then run:

```bash
python Code/train_streetclip_country.py \
  --data-root TRAIN_DATASET/koglab_levi \
  --model-dir MODEL/StreetCLIP \
  --epochs 8 \
  --batch-size 64 \
  --grad-accum-steps 2 \
  --unfreeze-vision-layers 4 \
  --lr-head 1e-3 \
  --lr-vision 1e-5 \
  --amp-dtype bf16 \
  --num-workers 8 \
  --prefetch-factor 4
```

More details:

```text
Code/README_StreetCLIP.md
```
