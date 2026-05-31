# Interpretability Notes

This project uses three different analysis layers. They answer different
questions, so the report should not describe them as the same kind of evidence.

## Attention Rollout

Attention rollout combines attention matrices through the transformer layers,
including residual connections, to estimate how information can flow from patch
tokens to the CLS/global representation. It is useful for asking:

- is the model's spatial focus concentrated or diffuse?
- do correct and wrong predictions differ in attention entropy or top-k mass?
- do StreetCLIP and DFN2B-CLIP show different focus patterns on the same image?

It should not be presented as a complete causal explanation of the prediction.
Attention weights can be weak explanations on their own, and rollout is a
post-hoc diagnostic rather than an intervention.

Also important: the rollout used here is class-agnostic. It measures how patch
information flows into the global image representation, not specifically which
pixels supported "Hungary" versus "Poland". Class-specific transformer
attribution methods usually add gradients/relevance propagation; they are more
ambitious and more fragile to implement across different CLIP/DINO codepaths.
For this project, we therefore use rollout as a systematic focus metric and
avoid claiming it is the full reason for a class decision.

Implementation detail: saved `heatmaps/*.npy` files are normalized attention
mass. Metrics are computed from this mass, not from display colors. Paper panels
default to `--heatmap-norm mass`, where the color maximum is a fixed multiple of
uniform attention. This avoids exaggerating small within-image contrasts.

Use `--heatmap-norm minmax` only when you want to inspect local maxima in a
single image. Do not use min-max panels as cross-image or cross-model evidence.

## Sparse Autoencoder

An SAE does not directly output "spots on the image." The SAE is trained on a
chosen internal activation space, here final-layer DINO patch-token vectors. It
learns a sparse dictionary of feature directions:

```text
patch activation vector -> sparse feature activations -> reconstructed vector
```

The contact-sheet images are a visualization of the feature activations. For a
given SAE feature, the script finds the patch tokens that activate that feature
most strongly, then draws yellow boxes around the corresponding image patches.
`feature_summary.csv` adds basic checks for each feature: activation sparsity,
top-country concentration, and mean top-patch position. These numbers help catch
boring features such as "mostly bottom-left patch" or "mostly one country"
before we over-interpret a contact sheet.

So the correct interpretation is:

- the yellow boxes are top examples for an internal feature,
- the feature may correspond to a human-readable concept only after inspecting
  many top examples,
- the feature is not automatically a class label, object label, or explanation
  of a particular prediction,
- stronger claims require additional validation, such as class/country
  enrichment, concept consistency, or ablation.

This matches the Anthropic-style dictionary-learning idea at a small scale: the
SAE decomposes activations into sparse features that can sometimes be interpreted
from top-activating examples. In our case, because the activation units are image
patch tokens, the natural exemplar view is image patches.

## CLIPSeg Concept Attribution

CLIPSeg is used as an approximate open-vocabulary segmenter for concepts such as
`road`, `sky`, `trees`, `grass`, `building`, and `traffic sign`. The script
measures how much rollout attention mass falls inside each predicted mask.

This is closer to the teacher's "human ontology" request, but it is still not
manual ground truth. Phrase masks can be noisy or overlapping. The safe wording
is:

```text
We approximate concept-level attention by intersecting rollout mass with
CLIPSeg masks for human-chosen visual concepts.
```

## Good Paper Wording

Use cautious language:

- "attention mass is more concentrated on..."
- "rollout suggests the model routes more spatial attention through..."
- "SAE feature exemplars often contain..."
- "CLIPSeg-based concept attribution estimates..."

Avoid overclaiming:

- "the model thinks this is Hungary because..."
- "the SAE found the object..."
- "the red heatmap proves causal importance..."

## References

- Abnar and Zuidema, 2020, "Quantifying Attention Flow in Transformers":
  https://arxiv.org/abs/2005.00928
- Jain and Wallace, 2019, "Attention is not Explanation":
  https://arxiv.org/abs/1902.10186
- Chefer, Gur, and Wolf, 2021, "Transformer Interpretability Beyond Attention
  Visualization": https://arxiv.org/abs/2012.09838
- Anthropic, 2023, "Towards Monosemanticity":
  https://transformer-circuits.pub/2023/monosemantic-features/
- CLIPSeg, 2021/2022, "Image Segmentation Using Text and Image Prompts":
  https://arxiv.org/abs/2112.10003
