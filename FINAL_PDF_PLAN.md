# Final PDF Plan

Goal: produce one final Hungarian PDF that replaces the earlier
`Ertelmezheto_GeoLab_AI_V2.pdf` style report, but with the new CLIP results,
Levi's newest DINO comparison, stronger quantitative evaluation, and clearer
interpretability wording.

## Final PDF Should Answer

1. What is the task?
   - Country classification from Google Street View / Google Maps images.
   - 22 country classes.
   - Train folders are `country/`; test folders are `country_test/`.

2. What models were compared?
   - StreetCLIP fine-tuned classifier: strongest final model.
   - DFN2B-CLIP ViT-B/16 fine-tuned baseline.
   - Levi's newest DINOv2 ViT-S/14 model, especially `5_dino_geo.weights.h5`.

3. How did training happen?
   - Include model size, layer details, trainable layers, optimizer settings,
     epochs, validation curves, final test accuracy, per-country results.
   - Use `runs_streetclip/<run>/config.json`, `epoch_metrics.csv`,
     `test_report_streetclip.txt`, and prediction/confusion CSVs.
   - Use `runs_clip/<run>/config.json`, `epoch_metrics.csv`,
     `test_report_clip.txt`, and prediction/confusion CSVs.

4. What did Levi's DINO comparison show?
   - Use `analysis_outputs/levi_dino_comparison/`.
   - Overall comparison should include StreetCLIP, DFN2B-CLIP, and Levi-DINO-5.
   - Earlier observed result: StreetCLIP was about `0.8015` test accuracy,
     DFN2B-CLIP about `0.6653`, Levi-DINO-5 about `0.6354`.
   - Recompute from files in the final run and use the recomputed values.

5. What did attention rollout show?
   - Attention rollout is spatial focus / information-flow evidence.
   - It is not causal proof of the final country decision.
   - Heatmap metrics use normalized attention mass.
   - Paper heatmaps use conservative `mass` color scaling, not per-image minmax.
   - Use full or large-sample metrics from:
     `analysis_outputs/streetclip_attention/`,
     `analysis_outputs/dfn2b_attention/`,
     `analysis_outputs/dino_attention/`.

6. What did concept attribution show?
   - CLIPSeg approximates concepts such as road, sky, trees, grass, buildings,
     vehicles, traffic signs.
   - Prefer `attention_lift_vs_uniform`, not raw `attention_mass`, because raw
     mass is biased by mask size.
   - Use:
     `analysis_outputs/streetclip_concepts/`,
     `analysis_outputs/dfn2b_concepts/`.

7. What did SAE show?
   - SAE input: DINO final-layer patch-token activation vectors.
   - SAE output: sparse feature activations and reconstructed patch vectors.
   - Yellow boxes are top-activating patch examples for a feature.
   - They are not direct object detections and not causal proof.
   - Use `analysis_outputs/dino_sae/feature_summary.csv`,
     `top_features.csv`, contact sheets, and `SAE_INTERPRETATION_NOTE.txt`.

8. How do attention rollout and SAE connect?
   - Attention rollout says where the model routes spatial information.
   - SAE says what recurring internal patch features exist.
   - Strong wording: "high-attention regions often overlap with patches whose
     SAE features appear road/sign/vegetation-like."
   - Avoid: "the SAE proves the model predicted country X because of object Y."

9. How is the work reproducible?
   - Include GitHub URL: `https://github.com/RabbDavid/Train_clip.git`.
   - Include exact commit hash from `FINAL_RESULTS/repo_info.txt`.
   - Include `requirements.txt` and `FINAL_RESULTS/environment.txt`.
   - Mention large local assets are not committed: dataset, model folders,
     checkpoints, Levi `.h5` weights.

## External Download Checklist

```text
GitHub code:
https://github.com/RabbDavid/Train_clip.git

DFN2B-CLIP:
https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16

StreetCLIP:
https://huggingface.co/geolocal/StreetCLIP

Dataset:
Google Drive folders supplied by the group; extract under TRAIN_DATASET/koglab_levi/

Levi DINO model/results:
Modellek, scriptek-20260531T190547Z-3-001.zip
Results-20260531T190638Z-3-001.zip
KogLab_DINO.docx
```

## Required Files From GPU Run

The final PDF cannot be completed cleanly without these:

```text
FINAL_RESULTS_CLIP_COUNTRY.zip
analysis_outputs/
runs_streetclip/<run>/
runs_clip/<run>/
preflight_report.txt
```

Inside the zip/output we need at minimum:

```text
FINAL_RESULTS/
  streetclip/
    config.json
    epoch_metrics.csv
    test_report_streetclip.txt
    test_predictions_streetclip.csv
    confusion_matrix_streetclip.csv

  dfn2b_clip/
    config.json
    epoch_metrics.csv
    test_report_clip.txt
    test_predictions_detailed_clip.csv
    confusion_matrix_clip.csv
    wiseft_metrics.csv

  analysis_outputs/
    streetclip_attention/
    dfn2b_attention/
    attention_comparison/
    streetclip_concepts/
    dfn2b_concepts/
    dino_attention/
    dino_examples/
    dino_sae/
    levi_dino_comparison/

  documentation/
    README.md
    RUN_GPU_ANALYSIS.md
    INTERPRETABILITY_NOTES.md
    requirements.txt

  environment.txt
  repo_info.txt
```

## Final PDF Structure

1. Abstract
2. Task and dataset
3. Models
   - StreetCLIP
   - DFN2B-CLIP
   - Levi DINOv2
4. Training setup and reproducibility
5. Training curves and final test results
6. Per-country performance and confusion patterns
7. Levi DINO comparison
8. Attention rollout method
9. Attention rollout quantitative results
10. Qualitative attention examples with conservative colors
11. CLIPSeg concept attribution
12. SAE method and output interpretation
13. SAE results
14. Relationship between attention and internal features
15. Limitations
16. Conclusion
17. Reproducibility appendix

## Important Wording Rules

Use:

```text
attention mass
spatial focus diagnostic
class-agnostic rollout
SAE feature exemplar
approximate CLIPSeg concept mask
attention lift compared with uniform expectation
```

Avoid:

```text
the model thinks
the heatmap proves
the SAE detected the object
red means 90 percent importance
this object caused the prediction
```
