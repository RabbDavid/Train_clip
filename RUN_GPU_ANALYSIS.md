# GPU Analysis Runbook

This is the checklist for the GPU-side machine after training has finished.

The goal is to produce the missing project evidence:

- CLIP/StreetCLIP attention rollout examples,
- quantitative attention focus metrics,
- optional ontology-level concept attribution,
- one packaged `FINAL_RESULTS_CLIP_COUNTRY.zip`.

## 1. Pull Latest Code

```bash
git pull
pip install -r requirements.txt
```

Expected local folders:

```text
Train_clip/
  MODEL/
    DFN2B-CLIP-ViT-B-16/
    StreetCLIP/
  TRAIN_DATASET/
    koglab_levi/
  runs_clip/
    20260519-213048/
  runs_streetclip/
    20260519-221734/
```

If the run folder names differ, replace them in the commands below.

## 1A. Simplest One-Command Workflow

First run the preflight check:

```bash
python analysis/preflight_check.py
```

Then run the main attention workflow:

```bash
python analysis/run_gpu_analysis_workflow.py --skip-concepts
```

If CLIPSeg concept attribution is also desired and internet/model cache is
available:

```bash
python analysis/run_gpu_analysis_workflow.py --run-concepts
```

If the DINO `.h5` weights are present and DINO reproduction should also run:

```bash
python analysis/run_gpu_analysis_workflow.py --skip-concepts --run-dino
```

The sections below show the same steps manually.

## 2. Run Attention Rollout Metrics

StreetCLIP:

```bash
python analysis/attention_rollout_clip.py \
  --model streetclip \
  --data-root TRAIN_DATASET/koglab_levi \
  --model-dir MODEL/StreetCLIP \
  --checkpoint runs_streetclip/20260519-221734/best.pt \
  --out-dir analysis_outputs/streetclip_attention \
  --max-samples-per-country 60 \
  --batch-size 8 \
  --save-panels 80 \
  --amp-dtype bf16
```

DFN2B-CLIP:

```bash
python analysis/attention_rollout_clip.py \
  --model dfn2b \
  --data-root TRAIN_DATASET/koglab_levi \
  --model-dir MODEL/DFN2B-CLIP-ViT-B-16 \
  --checkpoint runs_clip/20260519-213048/best.pt \
  --out-dir analysis_outputs/dfn2b_attention \
  --max-samples-per-country 60 \
  --batch-size 8 \
  --save-panels 80 \
  --amp-dtype bf16
```

These runs create:

```text
analysis_outputs/<model>_attention/
  attention_metrics.csv
  attention_summary.txt
  attention_summary_by_correctness.csv
  attention_summary_by_country.csv
  attention_entropy_correct_vs_wrong.png
  attention_entropy_vs_confidence.png
  panels/
  heatmaps/
```

## 3. Compare Attention Runs

```bash
python analysis/summarize_attention_runs.py \
  --streetclip-metrics analysis_outputs/streetclip_attention/attention_metrics.csv \
  --dfn2b-metrics analysis_outputs/dfn2b_attention/attention_metrics.csv \
  --out-dir analysis_outputs/attention_comparison
```

This creates cross-model charts/tables:

```text
analysis_outputs/attention_comparison/
  attention_comparison_summary.txt
  attention_summary_by_model.csv
  attention_summary_by_model_and_correctness.csv
  attention_summary_by_country.csv
  attention_entropy_distribution.png
  attention_entropy_correct_vs_wrong_models.png
  attention_sample_accuracy_by_country.png
```

## 4. Optional DINO Reproduction

The previous DINO/ViT-S/14 classifier tools are included in:

```text
legacy_dino/
```

They require the classmate Keras/PyTorch-backend weights file:

```text
dino_geo_28_countries_full.weights.h5
```

Place that file in the repo root, then run:

```bash
python legacy_dino/attention_quantitative_eval.py \
  --h5 dino_geo_28_countries_full.weights.h5 \
  --data-root TRAIN_DATASET/koglab_levi \
  --out-dir analysis_outputs/dino_attention \
  --per-country 60 \
  --batch-note sampled_60_per_country
```

For qualitative DINO examples:

```bash
python legacy_dino/run_viz.py \
  --h5 dino_geo_28_countries_full.weights.h5 \
  --data-root TRAIN_DATASET/koglab_levi \
  --out-dir analysis_outputs/dino_examples \
  --per-country 2 \
  --misclassified 1 \
  --heatmap-norm mass
```

This keeps the DINO comparison reproducible without committing the large `.h5`
weights file.

For the older DINO SAE experiment:

```bash
python legacy_dino/sae_quick.py \
  --h5 dino_geo_28_countries_full.weights.h5 \
  --data-root TRAIN_DATASET/koglab_levi \
  --out-dir analysis_outputs/dino_sae \
  --max-per-country 60 \
  --patches-per-image 16 \
  --epochs 12
```

SAE caveat: this script trains a sparse autoencoder on final-layer DINO patch
tokens and produces top-activating patch contact sheets. These sheets are
feature exemplars, not direct object labels.

## 5. Optional Object/Concept Attribution

This approximates the "road / sky / trees / grass / building" ontology using
CLIPSeg. It may download `CIDAS/clipseg-rd64-refined` from Hugging Face unless
the model is already cached.

StreetCLIP concept attribution:

```bash
python analysis/object_concept_attention.py \
  --attention-metrics analysis_outputs/streetclip_attention/attention_metrics.csv \
  --out-dir analysis_outputs/streetclip_concepts \
  --clipseg-model CIDAS/clipseg-rd64-refined \
  --no-local-files-only \
  --max-samples 500
```

DFN2B concept attribution:

```bash
python analysis/object_concept_attention.py \
  --attention-metrics analysis_outputs/dfn2b_attention/attention_metrics.csv \
  --out-dir analysis_outputs/dfn2b_concepts \
  --clipseg-model CIDAS/clipseg-rd64-refined \
  --no-local-files-only \
  --max-samples 500
```

Outputs:

```text
concept_attention.csv
concept_attention_summary_by_correctness.csv
concept_attention_correct_vs_wrong.png
concept_attention_summary.txt
```

Interpretation caveat: CLIPSeg masks are approximate open-vocabulary masks, not
manual ground truth segmentation.

## 6. Package Final Outputs

```bash
python analysis/package_final_outputs.py \
  --streetclip-run runs_streetclip/20260519-221734 \
  --dfn2b-run runs_clip/20260519-213048 \
  --analysis-root analysis_outputs \
  --out-dir FINAL_RESULTS \
  --include-checkpoints \
  --zip
```

Send back:

```text
FINAL_RESULTS_CLIP_COUNTRY.zip
```

Also paste these into chat:

```text
StreetCLIP attention summary:
cat analysis_outputs/streetclip_attention/attention_summary.txt

DFN2B attention summary:
cat analysis_outputs/dfn2b_attention/attention_summary.txt

Comparison:
cat analysis_outputs/attention_comparison/attention_comparison_summary.txt
```

## Notes For The Paper

Validation curves are available per epoch. Test accuracy is final-only by
design. We should not invent a test-over-epochs graph.

Attention rollout is a spatial focus diagnostic. It is useful evidence, but it
is not a causal proof of why the model predicted a class.
