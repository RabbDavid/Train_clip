# Local Levi DINO Notes

This is for David's laptop, not the GPU machine.

## Local Files Found

Downloaded zips:

```text
C:\Users\Dávid\Downloads\Modellek, scriptek-20260531T190547Z-3-001.zip
C:\Users\Dávid\Downloads\Results-20260531T190638Z-3-001.zip
C:\Users\Dávid\Downloads\Test_pictures-20260531T190650Z-3-001.zip
C:\Users\Dávid\Downloads\KogLab_DINO.docx
```

Already extracted working folder:

```text
C:\tmp\levi_new_dino\
  Modellek, scriptek\
    5_dino_geo.weights.h5
  Results\
    0_test_predictions_detailed.csv
    ...
    5_test_predictions_detailed.csv
```

Local dataset path:

```text
C:\Users\Dávid\Desktop\VisualTransformerHomework\TRAIN_DATASET\koglab_levi
```

## What Can Run Locally

Your local Python currently has CPU-only PyTorch:

```text
torch 2.11.0+cpu
CUDA: False
```

So:

- single-image Levi DINO attention rollout: yes, easy;
- small DINO SAE smoke test: yes, easy;
- full StreetCLIP/DFN2B attention: better on GPU;
- serious SAE with many images/tokens: better on GPU;
- CLIPSeg concept attribution on many images: better on GPU.

## Verified Local Attention Smoke Command

This already worked locally:

```powershell
$img = Get-ChildItem -Path 'C:\Users\Dávid\Desktop\VisualTransformerHomework\TRAIN_DATASET\koglab_levi\argentina_test' -File | Select-Object -First 1 -ExpandProperty FullName
python legacy_dino\run_viz.py --h5 'C:\tmp\levi_new_dino\Modellek, scriptek\5_dino_geo.weights.h5' --image $img --out-dir 'C:\tmp\levi_new_dino\local_pc_smoke_viz' --heatmap-norm mass --img-size 224
```

Outputs:

```text
C:\tmp\levi_new_dino\local_pc_smoke_viz\...\__rollout.jpg
C:\tmp\levi_new_dino\local_pc_smoke_viz\...\__per_head_last.jpg
```

## Verified Local SAE Smoke Command

This also worked locally, but it is only a smoke test, not final evidence:

```powershell
python legacy_dino\sae_quick.py --h5 'C:\tmp\levi_new_dino\Modellek, scriptek\5_dino_geo.weights.h5' --data-root 'C:\Users\Dávid\Desktop\VisualTransformerHomework\TRAIN_DATASET\koglab_levi' --out-dir 'C:\tmp\levi_new_dino\local_pc_sae_smoke' --split train --max-per-country 2 --patches-per-image 4 --hidden 64 --epochs 1 --batch-size 32 --top-k-features 3 --top-k-patches 3 --feature-summary-top-k 5
```

Outputs:

```text
C:\tmp\levi_new_dino\local_pc_sae_smoke\
  loss.csv
  feature_summary.csv
  top_features.csv
  feature_*.jpg
  SAE_INTERPRETATION_NOTE.txt
```

## Recommended Use

Use local runs for:

- checking that Levi's `.h5` model loads;
- generating one or two quick DINO attention examples;
- confirming SAE output format.

Use GPU runs for final PDF evidence.

