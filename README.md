# CLIP Country Classifier

Fine-tune an OpenCLIP model to classify Google Street View images by country.

## Folder Layout

Put the Google Drive image downloads here after cloning:

```text
Train_clip/
  train_clip_country.py
  requirements.txt
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

Do not commit `TRAIN_DATASET/`, `data/`, `runs_clip/`, or model checkpoints.

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

Primary CLIP model:

```text
hf-hub:apple/DFN2B-CLIP-ViT-B-16
```

Model page:

```text
https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16
```

The model downloads automatically on first run.

## Zero-Shot Baseline

```bash
python train_clip_country.py \
  --data-root TRAIN_DATASET/koglab_levi \
  --model hf-hub:apple/DFN2B-CLIP-ViT-B-16 \
  --zero-shot-only \
  --batch-size 256 \
  --num-workers 8 \
  --amp-dtype bf16
```

## Main Fine-Tune

```bash
python train_clip_country.py \
  --data-root TRAIN_DATASET/koglab_levi \
  --model hf-hub:apple/DFN2B-CLIP-ViT-B-16 \
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
