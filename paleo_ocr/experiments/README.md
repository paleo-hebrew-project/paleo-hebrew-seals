# Paleo-Hebrew detector/classifier experiments

## Data paths (Stage A / Stage B / real)

| Source | Path |
|--------|------|
| Stage A (structural) | `runs/synthetic_v2_parallel_advanced_20260222/all_manifest.jsonl` |
| Stage B (styled) | `notebooks/syn_v2_styled_advanced_20260222/manifest.jsonl` |
| Real train / val | `notebooks/paleo_ocr_part2/yolo_22_from_val_split_singlecls/manifest_train.jsonl` + `manifest_val.jsonl` |

See [`MODEL_MATRIX.md`](MODEL_MATRIX.md) for **parallel detector** (YOLO vs RT-DETR, etc.) and **classifier backbone** (ConvNeXt vs EfficientNet, Swin, ViT, ResNet) tables.
See [`SWEEP_RUNBOOK.md`](SWEEP_RUNBOOK.md) for a practical launch guide with ready-to-run commands, logging tips, and the recommended `paper` profile workflow.

## Setup

```bash
cd /path/to/paleo-hebrew-project
pip install -r requirements_experiments.txt   # PyYAML
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

## Two-phase workflow

1. **Detector sweep** — compare architectures and training regimes; pick best mAP@50 on real val.
2. **Classifier sweep** — use the same GT crops (manifest-based) or re-run `predict_detect` with the chosen `best.pt` for end-to-end ablations. Train classifiers with different **timm** backbones in parallel.

Parallel launch (8 GPUs example — edit `GPUS` in script):

```bash
bash scripts/run_parallel_detector_sweep.sh
# After analysis:
bash scripts/run_parallel_classifier_backbones.sh
```

Stage-A-only ablations on the same real train/val definitions:

```bash
bash scripts/run_parallel_detector_stagea_sweep.sh
bash scripts/run_parallel_classifier_stagea_backbones.sh
# Classifier regimes:
CLASSIFIER_PHASE1_EPOCHS=20 bash scripts/run_parallel_classifier_stagea_backbones.sh
CLASSIFIER_PHASE1_EPOCHS=60 bash scripts/run_parallel_classifier_stagea_backbones.sh
CLASSIFIER_PHASE1_EPOCHS=120 bash scripts/run_parallel_classifier_stagea_backbones.sh
```

Detector augmentation ablation (same schedule/backbone family, only aug changes):

```bash
bash scripts/run_detector_augmentation_ablation.sh
```

## Single experiment

```bash
python -m paleo_ocr.experiments.run_experiment --config configs/experiments/detector_yolo_synth_b_val_real.yaml
python -m paleo_ocr.experiments.run_experiment --config configs/experiments/classifier_convnext_synthb_val_real.yaml
```

Backbone sweep without duplicating YAML (output dir `experiment_name` becomes `base__model_slug` unless you pass `--experiment-name`):

```bash
python -m paleo_ocr.experiments.run_experiment \
  --config configs/experiments/sweep_detector_stageb_base.yaml --detector-model yolo11l.pt
python -m paleo_ocr.experiments.run_experiment \
  --config configs/experiments/sweep_classifier_stageb_base.yaml --classifier-model efficientnet_b0
```

Sweep scripts `scripts/run_parallel_detector_sweep.sh` and `scripts/run_parallel_classifier_backbones.sh` use these bases and loop over many models. Override `DETECTOR_BASE_CONFIG` / `CLASSIFIER_BASE_CONFIG` / `CLASSIFIER_SWINV2_BASE_CONFIG` to point at custom YAMLs. For the reviewer-requested Stage-A isolation, use `scripts/run_parallel_detector_stagea_sweep.sh` and `scripts/run_parallel_classifier_stagea_backbones.sh`.

For detector sweep, the default base now uses **sequential** `synth_stage_b -> real` with explicit real train/val manifests from `notebooks/paleo_ocr_part2/yolo_22_from_val_split_singlecls/` (to avoid evaluating on the full mixed manifest).

Important: notebook `Paleo_OCR.ipynb` also patches Ultralytics with custom Albumentations transforms. This repository sweep currently ablates built-in Ultralytics augment knobs (`mosaic`, `degrees`, `scale`, etc.), not that custom patch.

Classifier regime override:

```bash
python -m paleo_ocr.experiments.classifier_train --config configs/experiments/classifier_convnext_synthb_val_real.yaml --regime real_only
```

## Ultralytics training folders (`runs/detect/...`)

If you trained with `project: runs/detect` from the repo (often under `scripts/runs/detect/runs/detect/<name>/`), aggregate **args.yaml**, **results.csv**, and **`yolo_dataset/export_meta.json`** into one table and verify every run used the same **val** manifest(s):

```bash
python -m paleo_ocr.experiments.aggregate_yolo_runs \
  --root scripts/runs/detect/runs/detect \
  --repo-root . \
  --out-csv yolo_runs_summary.csv
# or:
bash scripts/aggregate_ultralytics_detect_runs.sh
```

Exit code **3** if val definitions differ across runs. Stdout is CSV; sanity messages go to stderr.

## Evaluate checkpoints

```bash
python -m paleo_ocr.experiments.eval_detector --weights runs/detect/<run>/weights/best.pt --data-yaml runs/paleo_experiments/<exp>/yolo_dataset/data.yaml
python -m paleo_ocr.experiments.eval_classifier --checkpoint runs/paleo_experiments/<exp>/classifier_run/best.pt --model convnext_base --data runs/paleo_experiments/<exp>/cls_imagefolder
```

## Aggregate metrics

```bash
python -m paleo_ocr.experiments.aggregate_runs --root runs/paleo_experiments --out-csv runs_summary.csv
```

`aggregate_runs` includes `detector_model` and `classifier_model` from each run’s `config_resolved.json`. It merges **detector** `metrics_detector.json` (if you saved eval in the experiment root) and **classifier** `classifier_run/metrics_best.json` after training, or `classifier_run/metrics_classifier.json` after a standalone `eval_classifier` run.

### Comparing evaluations

- **CSV:** Open `runs_summary.csv` in a spreadsheet, or sort in the shell:
  ```bash
  python -m paleo_ocr.experiments.compare_runs --csv runs_summary.csv --sort-by macro_f1 --top 15
  python -m paleo_ocr.experiments.compare_runs --csv runs_summary.csv --sort-by map50 --top 10
  ```
- **Per-epoch curves (classifier):** `classifier_run/metrics_history.jsonl` — one JSON object per epoch (loss, acc, macro F1, per-class metrics).
- **Ultralytics detector:** training/val curves under `runs/detect/<run_name>/` (Ultralytics’ own `results.csv` / plots).

### Logging

- **Console:** training prints epoch lines; eval scripts print JSON summaries.
- **Artifacts:** `run_metadata.json`, `config_resolved.json`, `metrics_*.json`, `train_summary.json` (classifier).
- **TensorBoard (classifier):** set in YAML under `classifier`:
  ```yaml
  classifier:
    tensorboard: true
  ```
  Then: `tensorboard --logdir runs/paleo_experiments/<exp>/classifier_run/tensorboard`
  Requires `pip install tensorboard`.
- **External loggers:** not wired by default; you can wrap runs with Weights & Biases / MLflow by logging the same JSON files or using their PyTorch callbacks in a fork of `classify_core.py`.

## Other scripts

```bash
bash scripts/run_detector_experiments.sh
bash scripts/run_classifier_experiments.sh
```

Edit `epochs`, `batch`, and manifest paths in `configs/experiments/*.yaml` as needed.
