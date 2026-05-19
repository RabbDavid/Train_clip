# StreetCLIP Country Classifier

This folder contains the StreetCLIP version of the country classifier.

StreetCLIP is different from the DFN2B OpenCLIP model:

- DFN2B script: `train_clip_country.py`
- StreetCLIP script: `Code/train_streetclip_country.py`
- StreetCLIP uses Hugging Face `transformers`
- StreetCLIP is trained for street-level geolocation, so it is the strongest task-specific challenger

## Manual Model Download

Download this model manually:

```text
https://huggingface.co/geolocal/StreetCLIP
```

Put it here:

```text
Train_clip/MODEL/StreetCLIP/
```

Expected files include:

```text
MODEL/StreetCLIP/config.json
MODEL/StreetCLIP/preprocessor_config.json
MODEL/StreetCLIP/model.safetensors
```

or:

```text
MODEL/StreetCLIP/pytorch_model.bin
```

The script uses `local_files_only=True`, so it does not download the model.

## Dataset

Use the same merged dataset:

```text
Train_clip/TRAIN_DATASET/koglab_levi/
  argentina/
  argentina_test/
  australia/
  australia_test/
  ...
  usa/
  usa_test/
```

Folders without `_test` are train folders. Folders with `_test` are test folders.

## Recommended RTX 5000 Pro Run

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
  --prefetch-factor 4 \
  --attn-implementation sdpa
```

Effective batch size is:

```text
batch-size * grad-accum-steps = 64 * 2 = 128
```

If VRAM is comfortable, try:

```bash
--batch-size 96 --grad-accum-steps 2
```

If it runs out of memory:

```bash
--batch-size 32 --grad-accum-steps 4
```

## Why This Script Is Efficient

- parallel image loading with `num_workers`
- pinned memory when CUDA is available
- persistent DataLoader workers
- prefetching batches in workers
- non-blocking GPU transfers
- bf16 mixed precision
- fused AdamW when available
- gradient accumulation for a larger effective batch
- channels-last image tensors
- PyTorch/Hugging Face SDPA attention by default
- cosine LR schedule with warmup
- only the last N vision layers are unfrozen by default

`flash_attention_2` is exposed as an optional flag, but it is not the default because it needs a separate compatible `flash-attn` install. Use it only if the GPU machine already has it working:

```bash
--attn-implementation flash_attention_2
```

## Outputs

Outputs go to:

```text
runs_streetclip/<timestamp>/
```

Important files:

```text
best.pt
last.pt
config.json
epoch_metrics.csv
test_report_streetclip.txt
test_predictions_streetclip.csv
confusion_matrix_streetclip.csv
```
