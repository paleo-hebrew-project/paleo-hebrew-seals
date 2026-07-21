# Reproducibility

## 1. Data layout

All YAML configs resolve manifests under `data/`:

```
data/
├── real/
│   ├── manifest.jsonl              # 307 real images
│   ├── manifest_train.jsonl        # detector train split
│   ├── manifest_val.jsonl          # evaluation split (150)
│   ├── manifest_group_split.jsonl  # entry-disjoint group split
│   └── images/
├── stage_a/
│   ├── manifest.jsonl
│   └── images/
├── stage_b/
│   ├── manifest.jsonl
│   └── images/
├── lexicons/                       # Stage A text resources
└── SHA256SUMS
```

```bash
python scripts/verify_data_release.py --root data --sha256
```

## 2. Regime naming

| Code | Phase 0 | Phase 1 | Notes |
|------|---------|---------|-------|
| `A` | Stage A | — | structural synth only |
| `A+R` | Stage A | real | |
| `B+R` | **Stage B only** | real | Stage B images are derived from Stage A. Paper shorthand: **A+B+R**. Not joint A∪B training. |
| `R` | — | real 100e (det) | |
| `R40` | — | real 40e (cls) | |

Detector Stage B sequential: **10 + 90** epochs (not 10+40).
Classifier Stage B sequential: **20 + 120** epochs (override real length with `CLASSIFIER_PHASE1_EPOCHS`).

## 3. Hyperparameters

### Detector

| Parameter | Value |
|-----------|-------|
| imgsz | 640 |
| optimizer | AdamW, cosine |
| phase LR | 0.005 |
| Stage A / B pretrain | 10e |
| real finetune | 90e (patience 15) |
| real only | 100e |
| batch | 128; **32 for YOLOv8-X / YOLO26-X** |
| seed | 42 |
| aug | mosaic 0.4, fliplr 0.35, degrees 3, scale 0.4, shear 2, randaugment |

### Classifier

| Parameter | 224 backbones | SwinV2 @256 |
|-----------|---------------|-------------|
| imgsz / batch | 224 / 256 | 256 / 256 |
| phase-0 / phase-1 LR | 3e-4 / 1e-4 | same |
| wd / label smooth | 0.05 / 0.05 | same |
| freeze / warmup | 1e / 1e | same |
| Stage A/B pretrain | 20e | 20e |
| real finetune | 20 / 60 / 120 | same |
| R40 | 40e | 40e |
| seed | 42 | 42 |
| best metric | val_acc1 | val_acc1 |

## 4. Model grids

**Detectors:** YOLOv8-M/X, YOLO11-M/X, YOLO26-M/X, RT-DETR-L.

**Classifiers (16 complete sweeps):** ConvNeXt-T/S/B/L; EfficientNet-B0/B1/B2/B3-NS;
ResNet-34/101; Swin-T/S/B; SwinV2-T; ViT-S/B.

**Excluded:** ResNet-50; SwinV2-B; SwinV2-S (incomplete regimes).

## 5. Checkpoint selection & evaluation split

Best checkpoint by validation metric (`mAP50-95` / `val_acc1`).
The 150-image partition is an **evaluation / comparative validation split**,
not a sealed test set (see paper, README, `results/run_manifest.json`).

## 6. Seeds and variation

All published runs used **seed=42**, **one run per configuration**.
Seed-induced variance was **not** estimated for the full grid. For the
camera-ready / checklist we recommend:

- 3 seeds for 2–3 representative detector families;
- 3 seeds for 3 classifier families;
- bootstrap CIs clustered by `row_id`;
- for the remainder, state one run per configuration explicitly.

## 7. Stage B generation

Real pipeline: `scripts/style_adapt_sd15_controlnet_ip_multigpu.py`
(SD1.5 + ControlNet Canny + IP-Adapter). Corpus hyperparameters are documented
in `SYSTEM_REQUIREMENTS.md` and `ORIGIN.md`.
`paleo_ocr/style_adapt.py` diffusion mode is a non-functional scaffold.

## 8. Reproduce paper tables

```bash
python scripts/reproduce_tables.py --output results
# -> results/table_detector.csv
# -> results/table_classifier.csv
# -> results/ocr_baselines.csv
# -> results/run_manifest.json
```

Live aggregation after training:

```bash
bash scripts/aggregate_ultralytics_detect_runs.sh
python -m paleo_ocr.experiments.aggregate_runs --root runs/paleo_experiments --out-csv results/runs_summary.csv
```

The aggregator reads `classifier_run/phase_*/metrics_best.json` (last phase)
as well as the legacy flat layout.

## 9. AAAI reproducibility checklist notes

- Code: this repository (anonymized ZIP for OpenReview).
- Data: anonymized release under `data/` + SHA-256.
- Seeds / hypers / hardware: this document + YAMLs.
- Runs per result: 1 (seed 42); variation not estimated for full grid.
- Evaluation split role: comparative validation (not blind test).
