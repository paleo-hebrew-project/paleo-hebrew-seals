# PaleoHebrew-Seals: Reproducibility Supplement

Anonymous reproducibility package for the submission
**“Preserving Low-Resource Cultural Heritage with AI: A Dataset and Synthetic-to-Real Study of Paleo-Hebrew Seals.”**

This repository contains the training, evaluation, and synthetic-generation code
used to produce the detector, classifier, and OCR-transfer experiments reported
in the paper. It is intentionally self-contained: the Python package, experiment
configurations, sweep scripts, aggregation utilities, and the open-source
Paleo-Hebrew fonts are all included. Real images, synthetic images, and text
lexicons are distributed separately as a data release (see `data/README.md`) and
are referenced by relative path from the repository root.

The codebase is organized so that every experiment is driven by a YAML
configuration under `configs/experiments/`, launched by a sweep script under
`scripts/`, and aggregated by utilities in `paleo_ocr/experiments/`. All paths in
the configurations are relative to the repository root, so the same commands work
on any machine after the data release is placed under `data/`.

## What this repository covers

- **Real benchmark + synthetic corpus handling.** Manifest builders, group-split
  construction, and leakage-controlled train/evaluation splitting.
- **Stage A generation.** Structural, lexicon-aware, font-based rendering of
  Paleo-Hebrew inscriptions with exact character-level bounding boxes.
- **Stage B generation.** Structure-preserving diffusion stylization that adapts
  Stage A renders toward seal-like appearance while retaining the supervision.
- **Detector study.** Seven architectures (YOLOv8-M/X, YOLO11-M/X, YOLO26-M/X,
  RT-DETR-L) under four regimes: Stage A only (A), Stage A + real (A+R),
  Stage A + Stage B + real (A+B+R), and real only (R).
- **Classifier study.** Sixteen complete timm backbone sweeps under real-only
  (R40), Stage A only (A), and A / A+B followed by 20, 60, or 120 real epochs.
- **OCR transfer baselines.** Tesseract and Kraken Hebrew checkpoints evaluated
  on the real benchmark to quantify the domain gap.
- **Aggregation and evaluation.** Run aggregation, standalone detector and
  classifier evaluation, end-to-end pipeline evaluation, and CER/WER metrics.

## Repository layout

```
.
├── paleo_ocr/                 # Python package
│   ├── experiments/          # YAML-driven experiment orchestration + aggregation
│   ├── train_detect.py        # Ultralytics detector training core
│   ├── train_classify.py      # timm classifier training core
│   ├── synthetic_v_2_generator.py  # Stage A structural generator
│   ├── style_adapt.py         # Stage B style adaptation
│   ├── dataset_manifest.py    # real manifest builder
│   ├── build_group_split ...  # (in scripts/) entry-disjoint split builder
│   ├── predict_*.py, extract_crops.py, ocr_*.py, end_2_end_infer.py
│   └── paleo_labeler.py        # annotation interface
├── configs/experiments/       # all sweep + single-regime YAMLs (relative paths)
├── scripts/                   # parallel sweep launchers + aggregation
├── fonts/                     # open-source Paleo-Hebrew TTF fonts (Stage A)
├── data/                      # place the data release here (see data/README.md)
├── REPRODUCIBILITY.md         # seeds, hyperparameters, hardware, regime definitions
├── requirements.txt
└── pyproject.toml
```

## Quick start

1. **Create an environment and install dependencies.**

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

   For detector training install a CUDA build of PyTorch matching your driver,
   e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124`.

2. **Obtain the data release** and place it under `data/` (or symlink the
   manifests/images to the relative paths listed in `configs/experiments/`).
   See `data/README.md` for the expected layout.

3. **Run a single detector regime** (Stage A + real, one backbone):

   ```bash
   DETECTOR_GPUS="0" \
   DETECTOR_MODELS="yolov8m.pt" \
     bash scripts/run_parallel_detector_stagea_sweep.sh
   ```

4. **Run the classifier Stage A sweep** at a given real-finetune length:

   ```bash
   CLASSIFIER_GPUS="0 1" CLASSIFIER_PHASE1_EPOCHS=120 \
     bash scripts/run_parallel_classifier_stagea_backbones.sh
   ```

5. **Aggregate results:**

   ```bash
   bash scripts/aggregate_ultralytics_detect_runs.sh
   python -m paleo_ocr.experiments.aggregate_runs
   ```

See `REPRODUCIBILITY.md` for the full regime definitions, hyperparameters,
seeds, checkpoint-selection rules, and the exact commands that reproduce each
table in the paper.

## Regimes

| Code    | Pretrain                 | Real fine-tune | Used for            |
|---------|--------------------------|----------------|---------------------|
| `A`     | Stage A (structural)     | —              | detector ablation   |
| `A+R`   | Stage A                  | yes            | detector, classifier|
| `A+B+R` | Stage A + Stage B        | yes            | detector, classifier|
| `R`     | —                        | yes            | real-only reference |
| `R40`   | —                        | 40 epochs      | classifier reference|

## License

Code is released under the MIT license. The Paleo-Hebrew fonts under `fonts/`
are governed by their respective open-source licenses (SIL OFL and GPL with Font
Exception); see the font files and the dataset documentation for terms. Real
images and synthetic outputs carry the license terms documented in the data
release, not a single blanket license.

## AI-use disclosure

Generative AI assisted drafting and language editing of the accompanying paper.
The authors verified and remain responsible for all claims, results, and code.
