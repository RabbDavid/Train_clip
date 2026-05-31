# Bende GPU Run Request

This file is the exact request for the person/Claude controlling the GPU
machine. The goal is to run the final analysis and send back the artifacts
needed for one final PDF.

## What We Need Back

Send back:

```text
FINAL_RESULTS_CLIP_COUNTRY.zip
preflight_report.txt
```

Also paste the text contents of:

```text
analysis_outputs/streetclip_attention/attention_summary.txt
analysis_outputs/dfn2b_attention/attention_summary.txt
analysis_outputs/attention_comparison/attention_comparison_summary.txt
analysis_outputs/levi_dino_comparison/comparison_summary.txt
analysis_outputs/streetclip_concepts/concept_attention_summary.txt
analysis_outputs/dfn2b_concepts/concept_attention_summary.txt
analysis_outputs/dino_sae/SAE_INTERPRETATION_NOTE.txt
```

If any file is missing, say exactly which one and why.

## Expected Folder Layout On The GPU Machine

Run everything from the repo root:

```text
Train_clip/
```

Expected folders:

```text
Train_clip/
  MODEL/
    DFN2B-CLIP-ViT-B-16/
    StreetCLIP/

  TRAIN_DATASET/
    koglab_levi/
      argentina/
      argentina_test/
      ...

  runs_clip/
    <dfn2b_run_with_best.pt>/

  runs_streetclip/
    <streetclip_run_with_best.pt>/

  Modellek, scriptek/
    5_dino_geo.weights.h5

  Results/
    0_test_predictions_detailed.csv
    ...
    5_test_predictions_detailed.csv
```

The `Modellek, scriptek/` and `Results/` folders are Levi's exported files. If
they are somewhere else, use those paths in the commands below.

## Step 1. Pull Code And Install

```bash
git pull
pip install -r requirements.txt
```

If PyTorch CUDA is missing, install it from the official selector:

```text
https://pytorch.org/get-started/locally/
```

For recent CUDA this is usually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Step 2. Preflight

```bash
python analysis/preflight_check.py \
  --dino-h5 "Modellek, scriptek/5_dino_geo.weights.h5"
```

If this fails, do not start the long run. Fix the missing path/package first.

## Step 3. Recommended Final Run

The commands are shown in bash style with `\` line continuations. In
PowerShell, either put the command on one line or replace `\` with PowerShell
backticks.

This is the preferred final command. It runs:

- StreetCLIP attention rollout,
- DFN2B attention rollout,
- attention comparison,
- CLIPSeg concept attribution,
- Levi DINO attention analysis,
- DINO SAE,
- Levi DINO vs CLIP result comparison,
- final packaging zip.

```bash
python analysis/run_gpu_analysis_workflow.py \
  --run-concepts \
  --run-dino \
  --run-dino-sae \
  --dino-h5 "Modellek, scriptek/5_dino_geo.weights.h5" \
  --levi-results-dir Results \
  --max-samples-per-country 120 \
  --concept-max-samples 1200 \
  --batch-size 4 \
  --save-panels 120 \
  --dino-sae-max-per-country 200 \
  --dino-sae-patches-per-image 64 \
  --dino-sae-hidden 4096 \
  --dino-sae-epochs 20 \
  --amp-dtype bf16 \
  --heatmap-norm mass \
  --vmax-multiplier 4.0 \
  --heatmap-alpha 0.38 \
  --cmap viridis
```

Why these settings:

- `mass` heatmap scaling avoids exaggerated colors.
- `batch-size 4` is safer for StreetCLIP attention extraction.
- `120` samples per country gives stronger statistics than tiny examples while
  keeping runtime reasonable.
- DINO SAE uses more tokens than the quick demo but should still be manageable.

## Step 4. If Time Is Very Short

Run this smaller version:

```bash
python analysis/run_gpu_analysis_workflow.py \
  --run-concepts \
  --run-dino \
  --dino-h5 "Modellek, scriptek/5_dino_geo.weights.h5" \
  --levi-results-dir Results \
  --max-samples-per-country 60 \
  --concept-max-samples 500 \
  --batch-size 4 \
  --save-panels 80 \
  --amp-dtype bf16 \
  --heatmap-norm mass \
  --cmap viridis
```

Then run SAE separately if possible:

```bash
python legacy_dino/sae_quick.py \
  --h5 "Modellek, scriptek/5_dino_geo.weights.h5" \
  --data-root TRAIN_DATASET/koglab_levi \
  --out-dir analysis_outputs/dino_sae \
  --split train \
  --max-per-country 120 \
  --patches-per-image 32 \
  --hidden 2048 \
  --epochs 12
```

Then package:

```bash
python analysis/package_final_outputs.py \
  --streetclip-run runs_streetclip/<RUN_FOLDER> \
  --dfn2b-run runs_clip/<RUN_FOLDER> \
  --analysis-root analysis_outputs \
  --out-dir FINAL_RESULTS \
  --include-checkpoints \
  --zip
```

Replace `<RUN_FOLDER>` with the real folder names if needed.

## Step 5. Check That These Outputs Exist

```text
analysis_outputs/streetclip_attention/attention_metrics.csv
analysis_outputs/streetclip_attention/panels/
analysis_outputs/dfn2b_attention/attention_metrics.csv
analysis_outputs/dfn2b_attention/panels/
analysis_outputs/attention_comparison/attention_comparison_summary.txt
analysis_outputs/attention_comparison/attention_metric_auc_for_correctness.csv
analysis_outputs/attention_comparison/country_accuracy_attention_correlation.csv
analysis_outputs/streetclip_concepts/concept_attention_summary.txt
analysis_outputs/dfn2b_concepts/concept_attention_summary.txt
analysis_outputs/dino_attention/attention_summary.txt
analysis_outputs/dino_examples/
analysis_outputs/dino_sae/feature_summary.csv
analysis_outputs/dino_sae/SAE_INTERPRETATION_NOTE.txt
analysis_outputs/levi_dino_comparison/comparison_summary.txt
FINAL_RESULTS_CLIP_COUNTRY.zip
```

## Do Not Do These

- Do not use `--heatmap-norm minmax` for paper figures.
- Do not delete the run folders.
- Do not invent test-over-epoch curves. Test accuracy is final-only unless the
  training script explicitly evaluated test every epoch.
- Do not describe SAE boxes as object detections.

## Short Explanation For The Runner

Attention rollout:

```text
where the model routes spatial information
```

SAE:

```text
which recurring internal patch features exist in DINO activations
```

CLIPSeg concept attribution:

```text
approximate overlap between attention mass and human concepts like road/sky/tree
```

Final PDF needs all three, plus training curves, test accuracy, per-country
performance, and Levi DINO comparison.
