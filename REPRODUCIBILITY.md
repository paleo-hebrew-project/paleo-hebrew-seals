# Reproducibility

This document specifies the experimental regimes, hyperparameters, seeds,
checkpoint-selection rules, hardware, and exact commands needed to reproduce the
tables in the paper. All paths are relative to the repository root.

## 1. Data

Place the data release under `data/` so that the relative manifest paths in
`configs/experiments/*.yaml` resolve. The expected layout is documented in
`data/README.md`. The three source manifests are:

| Source id     | Relative path                                              | Role            |
|---------------|------------------------------------------------------------|-----------------|
| `real`        | `notebooks/manifest_hebrew_unambiguous.jsonl`              | real benchmark  |
| `synth_stage_a` | `runs/synthetic_v2_parallel_advanced_20260222/all_manifest.jsonl` | Stage A renders |
| `synth_stage_b` | `notebooks/syn_v2_styled_advanced_20260222/manifest.jsonl`        | Stage B styled  |

The entry-disjoint group split used for the headline detector and classifier
tables is built deterministically by:

```bash
python scripts/build_group_split_manifest.py
```

## 2. Regimes

| Code    | Phase 0 (pretrain)            | Phase 1 (real fine-tune) | Sweep base YAML                                  |
|---------|-------------------------------|--------------------------|--------------------------------------------------|
| `A`     | Stage A, 10 ep (detector) / 20 ep (classifier) | none        | `sweep_*_stagea_base.yaml`                       |
| `A+R`   | Stage A                       | real, 90 ep (det) / 20,60,120 (cls) | `sweep_*_stagea_base.yaml`               |
| `A+B+R` | Stage A + Stage B             | real, 90 ep (det) / 20,60,120 (cls) | `sweep_*_stageb_base.yaml`               |
| `R`     | none                          | real, 100 ep (det) / 40 ep (cls) | `sweep_*_real_only_*.yaml`                  |
| `R40`   | none                          | real, 40 ep (cls)         | `sweep_classifier_real_only_40e_base.yaml`       |

The `A+R` real-finetune length for classifiers is controlled by
`CLASSIFIER_PHASE1_EPOCHS` (20, 60, or 120); the experiment directory is
suffixed `__real{N}e` so the regimes coexist on disk.

## 3. Hyperparameters

### Detector (Ultralytics)

| Parameter        | Value                                  |
|------------------|----------------------------------------|
| Image size       | 640                                    |
| Optimizer        | AdamW, cosine LR                       |
| Phase-0 LR       | 0.005                                  |
| Phase-1 LR       | 0.005                                  |
| Phase-0 epochs    | 10 (Stage A)                           |
| Phase-1 epochs    | 90 (real fine-tune) / 100 (real only)  |
| Early stopping   | patience 15 (phase 1)                  |
| Close mosaic     | last 10 (phase 0), last 50 (phase 1)   |
| Augmentation     | mosaic 0.4, fliplr 0.35, degrees 3, scale 0.4, shear 2, randaugment |
| Batch size       | 128 default; **32 for YOLOv8-X and YOLO26-X** (OOM-safe) |
| Seed             | 42                                     |
| Single class     | true                                   |
| Shared cache     | `runs/paleo_experiments/_shared_yolo_cache` (or `DETECTOR_SHARED_CACHE_ROOT`) |

For large backbones that exceed 80 GB at `batch=128`, pass
`DETECTOR_BATCH=32` and `PYTORCH_ALLOC_CONF=expandable_segments:True`.

### Classifier (timm)

| Parameter        | Value (224 backbones)   | Value (SwinV2 @256)   |
|------------------|-------------------------|------------------------|
| Image size       | 224                     | 256                    |
| Batch size       | 256                     | 256                    |
| Optimizer        | AdamW                   | AdamW                  |
| Phase-0 LR       | 0.0003                  | 0.0003                 |
| Phase-1 LR       | 0.0001                  | 0.0001                 |
| Weight decay      | 0.05                    | 0.05                   |
| Label smoothing  | 0.05                    | 0.05                   |
| Warmup           | 1 epoch                 | 1 epoch                |
| Schedule         | cosine                  | cosine                 |
| Freeze backbone  | 1 epoch                 | 1 epoch                |
| AMP              | true                    | true                   |
| Phase-0 epochs   | 20 (Stage A)            | 20 (Stage A)           |
| Phase-1 epochs   | 20 / 60 / 120           | 20 / 60 / 120          |
| Real-only epochs | 40                      | 40                     |
| Seed             | 42                      | 42                     |
| Best metric      | val_acc1                | val_acc1               |

## 4. Model grids

- **Detectors (paper profile):** `yolov8m.pt yolov8x.pt yolo11m.pt yolo11x.pt
  yolo26m.pt yolo26x.pt rtdetr-l.pt`.
- **Classifiers (16 complete sweeps):** ConvNeXt-T/S/B/L, EfficientNet-B0/B1/B2
  + tf_efficientnet_b3_ns, ResNet-34/101, Swin-T/S/B (W7-224), SwinV2-T/S
  (W8-256), ViT-S/B (16, 224). ResNet-50, SwinV2-B, and SwinV2-S are omitted from
  the cross-architecture analysis because at least one regime is incomplete.

## 5. Checkpoint selection

Each reported row uses the **best** checkpoint by the regime's validation
metric (`mAP50-95` for detectors, `val_acc1` for classifiers). The evaluation
split participated in checkpoint selection in parts of the project, so the
results are comparative validation results rather than a blind test; this is
stated in the paper.

## 6. Hardware

- 6 × NVIDIA H100 80 GB, 1.1 TB RAM.
- One job per GPU; the sweep scripts queue models across GPUs.
- Detector shared YOLO export cache on local storage (`/tmp` or
  `DETECTOR_SHARED_CACHE_ROOT`) when the project lives on a slow network share.

## 7. Commands to reproduce the paper tables

### Detector table (Table 2 in the paper)

```bash
# A+R  (Stage A + real)
DETECTOR_GPUS="0 1 2 3 4 5" \
  bash scripts/run_parallel_detector_stagea_sweep.sh

# A+B+R (Stage A + Stage B + real)
DETECTOR_GPUS="0 1 2 3 4 5" \
  bash scripts/run_parallel_detector_sweep.sh

# R   (real only)
DETECTOR_GPUS="0 1 2 3 4 5" \
  bash scripts/run_parallel_detector_sweep_real_only.sh
DETECTOR_GPUS="0 1 2 3 4 5" \
  bash scripts/run_parallel_detector_sweep_real_only_non_yolo.sh
```

For YOLOv8-X / YOLO26-X add `DETECTOR_BATCH=32` and
`PYTORCH_ALLOC_CONF=expandable_segments:True`.

### Classifier table (Table 3 in the paper)

```bash
for N in 20 60 120; do
  CLASSIFIER_GPUS="0 1 2 3 4 5" CLASSIFIER_PHASE1_EPOCHS=$N \
    bash scripts/run_parallel_classifier_stagea_backbones.sh
  CLASSIFIER_GPUS="0 1 2 3 4 5" CLASSIFIER_PHASE1_EPOCHS=$N \
    bash scripts/run_parallel_classifier_backbones.sh
done

# R40 (real only reference)
CLASSIFIER_GPUS="0 1 2 3 4 5" \
  bash scripts/run_parallel_classifier_backbones_real_only_40e.sh
```

### Aggregation

```bash
bash scripts/aggregate_ultralytics_detect_runs.sh     # detector runs -> CSV
python -m paleo_ocr.experiments.aggregate_runs          # all runs -> runs_summary.csv
python -m paleo_ocr.experiments.compare_runs           # sort / top-k
```

### OCR transfer baselines

```bash
python -m paleo_ocr.paleo_ocr_baseline --manifest notebooks/manifest_hebrew_unambiguous.jsonl \
    --engines tesseract_heb kraken_midrash kraken_biblia kraken_ashkenazi \
              kraken_italian kraken_sephardi
```

## 8. Reproducibility checklist (for the submission system)

- **Code:** this repository (anonymized).
- **Data:** real benchmark and synthetic corpus are provided as an anonymized
  data supplement; manifests and split files are included.
- **Seeds:** all experiments use `seed: 42` (config).
- **Hyperparameters:** fully specified in the YAMLs and in §3 above.
- **Checkpoint selection:** best by validation metric (§5).
- **Hardware:** specified in §6.
- **Runtime:** detector phase-1 ≈ 0.02–0.05 h/epoch on H100; classifier 20 ep
  Stage A + 120 ep real ≈ a few hours per backbone on one H100.
- **Expected results:** detector mAP50-95 and classifier Acc1/macro-F1 match
  Tables 2–3 of the paper within seed-induced variance.
