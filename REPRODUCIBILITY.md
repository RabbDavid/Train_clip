# Reproducibility Notes

Tested local environment:

- Python 3.10.11
- torch 2.11.0
- torchvision 0.26.0
- timm 1.0.26
- open_clip_torch 3.3.0
- numpy 2.2.6
- pandas 2.3.3
- pillow 12.1.1
- tqdm 4.67.3
- matplotlib 3.10.9
- seaborn 0.13.2
- h5py 3.16.0

Install:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Regenerate the DINO attention figures used in the report:

```powershell
python make_paper_attention_assets.py
python attention_region_numbers.py
```

Run the quantitative attention analysis:

```powershell
python attention_quantitative_eval.py --per-country 30 --out-dir paper_assets/quant_attention --batch-note sampled_30_per_country
```

For the full test-set analysis, use `--per-country 0`.

Compile the Hungarian report:

```powershell
pdflatex -interaction=nonstopmode paper_hu.tex
```

Train a CLIP-style timm backbone on a prepared `data/train`, `data/val`, `data/test` layout:

```powershell
python train.py --data-root data --backbone vit_base_patch16_clip_224 --img-size 224 --epochs 20 --batch-size 128 --lr-backbone 2e-5 --lr-head 5e-4 --amp-dtype bf16
```

Evaluate a trained checkpoint:

```powershell
python eval.py --ckpt runs/<run-id>/best.pt --data-root data
```
