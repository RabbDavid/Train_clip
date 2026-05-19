# CLIP Fine-Tuning Plan

## Goal

Train a CLIP-based Google Street View country classifier and compare it against the previous frozen DINOv2/ViT baseline.

The new code is `train_clip_country.py`. It uses OpenCLIP directly:

- country classes are represented by text prompt prototypes,
- images are classified by image-text similarity,
- the text tower is frozen by default,
- the visual tower is fine-tuned,
- zero-shot, fine-tuned, and WiSE-FT-style interpolated checkpoints are evaluated.

## Model Choice

Primary recommendation:

```text
MODEL/DFN2B-CLIP-ViT-B-16
```

Manual download page:

```text
https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16
```

Expected local files:

```text
MODEL/DFN2B-CLIP-ViT-B-16/open_clip_config.json
MODEL/DFN2B-CLIP-ViT-B-16/open_clip_pytorch_model.bin
```

Why:

- It is still a reasonable ViT-B/16-sized CLIP model, not a giant ViT-H/G model.
- It is directly usable through OpenCLIP.
- The model card reports strong zero-shot metrics, including ImageNet 76.236% and Country211 19.545%.
- It is more modern than the original OpenAI CLIP ViT-B/16 and should be a better starting point for geographic visual cues.

Fallback if the Apple license is uncomfortable:

```text
MODEL/CLIP-ViT-B-16-DataComp.XL-s13B-b90K
```

Fallback download page:

```text
https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K
```

This is MIT-licensed and reports 73.5% ImageNet zero-shot accuracy.

## Why Not Muon First?

Muon and variants are interesting, but the current evidence is still mostly language-model-centric or fresh-pretraining-oriented. For this project, AdamW with discriminative layer-wise learning rates is the safer high-performance choice. The code uses:

- AdamW,
- bf16 AMP,
- fused AdamW when CUDA supports it,
- cosine learning rate schedule with warmup,
- layer-wise LR decay,
- gradient clipping,
- prompt-ensemble CLIP text prototypes,
- optional WiSE-FT interpolation.

PyTorch already dispatches scaled dot-product attention to fused CUDA implementations such as FlashAttention when the model/operator path supports it, so the first priority is a clean PyTorch 2.x CUDA install rather than a custom FlashAttention dependency.

## First GPU Commands

Install:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Check CUDA:

```powershell
@'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
print("flash available:", torch.backends.cuda.is_flash_attention_available() if torch.cuda.is_available() else False)
'@ | python -
```

Zero-shot sanity baseline:

```powershell
python train_clip_country.py `
  --data-root TRAIN_DATASET/koglab_levi `
  --model-dir MODEL/DFN2B-CLIP-ViT-B-16 `
  --zero-shot-only `
  --batch-size 256 `
  --num-workers 8 `
  --amp-dtype bf16
```

Main fine-tune:

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
  --num-workers 8
```

Conservative run if full fine-tuning overfits:

```powershell
python train_clip_country.py `
  --data-root TRAIN_DATASET/koglab_levi `
  --model-dir MODEL/DFN2B-CLIP-ViT-B-16 `
  --epochs 10 `
  --batch-size 256 `
  --unfreeze-visual-layers 4 `
  --lr-visual 3e-5 `
  --lr-logit-scale 1e-4 `
  --amp-dtype bf16 `
  --num-workers 8
```

## Sources Used

- OpenCLIP README and model loading notes: https://github.com/mlfoundations/open_clip
- OpenCLIP pretrained model overview: https://deepwiki.com/mlfoundations/open_clip/3-using-pretrained-models
- DFN2B CLIP ViT-B/16 model card: https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16
- DataComp CLIP ViT-B/16 model card: https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K
- OpenAI CLIP introduction: https://openai.com/index/clip/
- WiSE-FT repository: https://github.com/mlfoundations/wise-ft
- WiSE-FT paper: https://arxiv.org/abs/2109.01903
- PyTorch SDPA tutorial: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
- MuonAll paper: https://arxiv.org/abs/2511.06086
