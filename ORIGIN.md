# Origin: notebooks → YAML sweeps

The YAML experiment framework under `configs/experiments/` and
`paleo_ocr/experiments/` is a **later rewrite** of notebook-driven pipelines.
Knowing this lineage helps reviewers understand why some hyperparameters differ
slightly from early notebook cells and why Stage B lives as a separate script.

## Stage B stylization

| Item | Path / fact |
|------|-------------|
| Production script | `scripts/style_adapt_sd15_controlnet_ip_multigpu.py` (copied from the project notebooks tree) |
| Original launch | Full-pipeline notebook cell launching 8-GPU launcher mode |
| Output corpus | 200k images under Stage B `manifest.jsonl` |
| Stack | SD1.5 + ControlNet Canny + IP-Adapter img2img, 512² |
| Annotation transfer | Full Stage A JSONL row copied; boxes/chars/text unchanged; only image path + style meta updated |
| Not used | `paleo_ocr/style_adapt.py` diffusion CLI (scaffold / RuntimeError) |

## Detector (YOLO)

Original recipe lived in the main Paleo OCR notebook:

- export Stage B (or Stage A) synth train + real evaluation manifests to YOLO layout;
- train large YOLO backbones (e.g. YOLO26-X) with AdamW, mosaic, strong aug;
- later entry-aware / val-split finetune.

The YAML sweeps reuse those manifests and approximate the augmentation knobs
(mosaic 0.4, degrees 3, fliplr 0.35, …) but **do not** include the notebook’s
custom Albumentations Ultralytics patch. Sequential Stage B regime: **10e synth
→ 90e real**.

## Classifier (ConvNeXt-L and others)

Original **ConvNeXt-L** training was notebook-first (ImageFolder export from
manifests, ~20e synth then lower-LR finetune). The YAML sweep defaults to
`convnext_base` and overrides via `--classifier-model`, with a standardized
**20e Stage A/B → 20/60/120e real** schedule at imgsz 224 (SwinV2 @256).

## OCR baselines

Tesseract `heb` and Kraken Hebrew manuscript checkpoints were first exercised
in notebook / Colab baseline cells; the package CLI is
`paleo_ocr.paleo_ocr_baseline`.

## Implication for the supplement

- Training/evaluation reproducibility for the **paper tables** is via YAML sweeps.
- Stage B **regeneration** is via the multi-GPU diffusion script, not the classic helper.
- Notebooks are historical provenance; this repository ships the executable
  scripts and configs needed for follow-up work without requiring the original
  notebook environment.
