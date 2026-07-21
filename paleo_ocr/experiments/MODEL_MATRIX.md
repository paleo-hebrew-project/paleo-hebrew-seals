# Detector and classifier model matrix (parallel experiments)

Manifests (project defaults):

| Role | Path |
|------|------|
| Stage A (structural) | `runs/synthetic_v2_parallel_advanced_20260222/all_manifest.jsonl` |
| Stage B (styled) | `notebooks/syn_v2_styled_advanced_20260222/manifest.jsonl` |
| Real train / val (for detector sweep) | `notebooks/paleo_ocr_part2/yolo_22_from_val_split_singlecls/manifest_train.jsonl` + `manifest_val.jsonl` |

## Sweep mechanics

- **Stage B base YAMLs:** `configs/experiments/sweep_detector_stageb_base.yaml`, `sweep_classifier_stageb_base.yaml`, `sweep_classifier_swinv2_256.yaml`
- **Stage A ablation YAMLs:** `configs/experiments/sweep_detector_stagea_base.yaml`, `sweep_classifier_stagea_base.yaml`, `sweep_classifier_swinv2_256_stagea.yaml`
- **CLI:** `python -m paleo_ocr.experiments.run_experiment --config BASE.yaml --detector-model …` or `--classifier-model …`
- **Shell:** `scripts/run_parallel_detector_sweep.sh` and `scripts/run_parallel_classifier_backbones.sh` loop over many backbones and shard jobs across `GPUS=(0..7)` by default.
  - Detector sweep profiles: `DETECTOR_SWEEP_PROFILE=paper` (default) or `full`
  - Optional overrides: `DETECTOR_MODELS="..."`, `DETECTOR_GPUS="0 1"`, `DETECTOR_SWEEP_LOG_DIR=logs/detector_sweep`, `DETECTOR_EXTRA_CONFIGS=none`
- **Runbook:** see `SWEEP_RUNBOOK.md` for step-by-step launch commands, monitoring, and the recommended `paper` setup.

This mirrors common **domain-generalization / backbone comparison** tables (many architectures × one training recipe).

## Phase 1 — Detector sweep (Ultralytics)

All use `YOLO(model).train(...)` or RT-DETR under the same API. Default sweep (`sweep_detector_stageb_base.yaml`) is now **sequential**: Stage B synth pretrain → real finetune (train split), evaluated on the held-out real val split.

| Family | Checkpoints in sweep (examples) |
|--------|----------------------------------|
| YOLOv8 | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt` |
| YOLO11 | `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt` |
| YOLO26 | `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt` |
| RT-DETR | `rtdetr-l.pt`, `rtdetr-x.pt` |

For the Stage-A isolation requested by reviewers, run the same backbone grid with `scripts/run_parallel_detector_stagea_sweep.sh` instead of the legacy single-model `detector_stage_a_only_valreal.yaml`.

**Note:** Large models (`yolo11x`, `yolo26x`, `rtdetr-x`) may need a smaller `detector.batch` in the base YAML. Other stacks (Faster R-CNN, MMDetection) are not wired here.

### Augmentation ablation (detector)

- `configs/experiments/sweep_detector_stageb_base.yaml` → **strongaug**
- `configs/experiments/sweep_detector_stageb_lightaug.yaml` → **lightaug**
- `configs/experiments/sweep_detector_stageb_noaug.yaml` → **noaug**
- Run: `bash scripts/run_detector_augmentation_ablation.sh`

This ablates built-in Ultralytics augment knobs (`mosaic`, `degrees`, `scale`, `fliplr`, ...). The notebook’s custom patched Albumentations pipeline is a separate axis.

## Phase 2 — Classifier sweep (timm)

Default image size **224** and batch **16** in `sweep_classifier_stageb_base.yaml` for cross-family fairness. **Swin V2** uses `sweep_classifier_swinv2_256.yaml` (imgsz **256**, batch **8**) because those checkpoints match 256× inputs.

| Family | timm names (sweep) |
|--------|---------------------|
| ConvNeXt | `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large` |
| EfficientNet | `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `tf_efficientnet_b3_ns` |
| ResNet | `resnet34`, `resnet50`, `resnet101` |
| Swin (v1) | `swin_tiny_patch4_window7_224`, `swin_small_patch4_window7_224`, `swin_base_patch4_window7_224` |
| Swin V2 | `swinv2_tiny_window8_256`, `swinv2_small_window8_256`, `swinv2_base_window12_256` |
| ViT | `vit_small_patch16_224`, `vit_base_patch16_224` |

Pretraining follows each timm default (typically ImageNet-1k / 21k / CLIP variants depending on the string). For **Laion / custom pretrain** (e.g. some ConvNeXt variants), add the exact timm model name to the `MODELS` array in `run_parallel_classifier_backbones.sh` and tune batch size.

## Suggested order

1. `DETECTOR_SWEEP_PROFILE=paper bash scripts/run_parallel_detector_sweep.sh`
2. Choose best detector; optionally refresh crops with that `best.pt`
3. `bash scripts/run_parallel_detector_stagea_sweep.sh`
4. `bash scripts/run_parallel_classifier_backbones.sh`
5. `bash scripts/run_parallel_classifier_stagea_backbones.sh`
6. `python -m paleo_ocr.experiments.aggregate_runs --root runs/paleo_experiments --out-csv runs_summary.csv`

## Legacy per-model YAMLs

`configs/experiments/detector_*.yaml` and `classifier_*_stageb.yaml` remain for hand-tuned runs; sweeps prefer the base + CLI pattern above.
