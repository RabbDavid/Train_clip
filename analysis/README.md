# Analysis Scripts

These scripts are for post-training interpretability and quantitative analysis.

Main workflow:

1. `attention_rollout_clip.py`
   - runs attention rollout for StreetCLIP or DFN2B-CLIP,
   - saves heatmaps/panels,
   - computes entropy and concentration metrics.

2. `summarize_attention_runs.py`
   - combines StreetCLIP and DFN2B attention metrics,
   - creates comparison tables and charts.

3. `object_concept_attention.py`
   - optional CLIPSeg-based concept attribution,
   - estimates how much attention mass falls on concepts such as road, sky,
     trees, grass, buildings, vehicles, and signs.

4. `package_final_outputs.py`
   - collects training outputs, analysis outputs, environment info, and repo info
     into `FINAL_RESULTS_CLIP_COUNTRY.zip`.

5. `preflight_check.py`
   - checks that data, model folders, checkpoints, and Python packages exist.

6. `run_gpu_analysis_workflow.py`
   - one-command wrapper for the GPU-side remote run.

7. `compare_levi_dino_results.py`
   - compares Levi's exported DINO prediction CSVs with DFN2B/StreetCLIP
     prediction CSVs.

Full commands are in:

```text
RUN_GPU_ANALYSIS.md
```
