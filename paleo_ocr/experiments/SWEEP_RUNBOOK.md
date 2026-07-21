# Detector Sweep Runbook

This guide explains how to launch, monitor, and summarize detector backbone sweeps from the repository root.

It is written around:

- `scripts/run_parallel_detector_sweep.sh`
- `configs/experiments/sweep_detector_stageb_base.yaml`
- `python -m paleo_ocr.experiments.run_experiment`

The default detector base uses a two-phase sequential schedule:

1. `synth_stage_b` pretrain
2. `real` finetune

It also uses a shared YOLO export cache, so identical dataset exports are built once and then reused across models.

## Before You Start

Commands below assume:

- your environment is already activated
- you are in the repo root
- manifest paths inside `configs/experiments/sweep_detector_stageb_base.yaml` are valid on your machine

Minimal setup:

```bash
cd .
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

If needed:

```bash
pip install -r requirements_experiments.txt
```

## What The Sweep Script Does

`scripts/run_parallel_detector_sweep.sh` launches one detector experiment per backbone and shards jobs across the GPUs listed in `DETECTOR_GPUS`.

Important knobs:

- `DETECTOR_SWEEP_PROFILE=paper|full`
- `DETECTOR_MODELS="..."` to override the profile completely
- `DETECTOR_GPUS="0 1 2 3"` to choose visible GPUs
- `DETECTOR_SWEEP_LOG_DIR=...` to save one log per job
- `DETECTOR_BASE_CONFIG=...` to switch between Stage B, Stage A, real-only, or group-split bases

Stage-A ablations use a dedicated wrapper, `scripts/run_parallel_detector_stagea_sweep.sh`, so every backbone is trained and validated on the same split definitions as the Stage-B sweep.

## Recommended Launches

### 1. Paper Subset

Use this when you want the compact detector table for the article.

```bash
DETECTOR_SWEEP_PROFILE=paper \
DETECTOR_GPUS="0 1 2 3 4 5 6 7" \
DETECTOR_SWEEP_LOG_DIR="logs/detector_sweep_paper" \
DETECTOR_EXTRA_CONFIGS=none \
bash scripts/run_parallel_detector_sweep.sh
```

The shell script's `paper` profile currently launches these backbones:

- `yolov8m.pt`
- `yolov8x.pt`
- `yolo11m.pt`
- `yolo11x.pt`
- `rtdetr-l.pt`
- `yolo26m.pt`
- `yolo26x.pt`

### 2. Paper Subset Stage-A Ablation

Use this when you want the reviewer-requested Stage-A-only isolation on the same backbone grid and real train/val split as the Stage-B paper subset.

```bash
DETECTOR_SWEEP_PROFILE=paper \
DETECTOR_GPUS="0 1 2 3 4 5 6 7" \
DETECTOR_SWEEP_LOG_DIR="logs/detector_stagea_paper" \
bash scripts/run_parallel_detector_stagea_sweep.sh
```

### 3. Full Grid

Use this when you want the full family benchmark, not just the compact paper subset.

```bash
DETECTOR_SWEEP_PROFILE=full \
DETECTOR_GPUS="0 1 2 3 4 5 6 7" \
DETECTOR_SWEEP_LOG_DIR="logs/detector_sweep_full" \
DETECTOR_EXTRA_CONFIGS=none \
bash scripts/run_parallel_detector_sweep.sh
```

### 4. Small Custom Subset

Useful for smoke tests or limited GPU time.

```bash
DETECTOR_MODELS="yolov8m.pt yolo11m.pt rtdetr-l.pt" \
DETECTOR_GPUS="0 1 2" \
DETECTOR_SWEEP_LOG_DIR="logs/detector_sweep_custom" \
DETECTOR_EXTRA_CONFIGS=none \
bash scripts/run_parallel_detector_sweep.sh
```

### 5. Single Backbone Without The Shell Wrapper

Useful for debugging one model or one YAML.

```bash
python -m paleo_ocr.experiments.run_experiment \
  --config configs/experiments/sweep_detector_stageb_base.yaml \
  --detector-model yolo11m.pt
```

### 6. Non-YOLO Detector Scripts

For non-YOLO detector experiments, use:

- `scripts/run_parallel_detector_sweep_phase0_non_yolo.sh`
- `scripts/run_parallel_detector_sweep_phase1_non_yolo.sh`
- `scripts/run_parallel_detector_sweep_non_yolo_full.sh`

These scripts default to a broader non-YOLO spec set than just the two RT-DETR `.pt` weights:

- `rtdetr-l.yaml`
- `rtdetr-x.yaml`
- `rtdetr-resnet50.yaml`
- `rtdetr-resnet101.yaml`

This is the intended single-pack non-YOLO comparison in the current repo: official Ultralytics RT-DETR family including the ResNet50/101 variants.

You can still override the list entirely via:

- `DETECTOR_MODELS="rtdetr-l.yaml rtdetr-resnet50.yaml"`

Example:

```bash
DETECTOR_GPUS="0 1" \
DETECTOR_SWEEP_LOG_DIR="logs/non_yolo_phase0" \
bash scripts/run_parallel_detector_sweep_phase0_non_yolo.sh
```

One-command full run:

```bash
DETECTOR_GPUS="0 1 2 3" \
DETECTOR_SWEEP_LOG_DIR="logs/detector_sweep_non_yolo_full" \
bash scripts/run_parallel_detector_sweep_non_yolo_full.sh
```

This wrapper runs phase 0 first and then phase 1, writing logs into:

- `logs/detector_sweep_non_yolo_full/phase0`
- `logs/detector_sweep_non_yolo_full/phase1`

### 7. Real-Only 100-Epoch Detector Sweep

If you want to skip synth pretraining completely and train only on the real train subset for 100 epochs, use:

- `configs/experiments/sweep_detector_real_only_100e_base.yaml`

YOLO-like / mixed Ultralytics sweep:

```bash
DETECTOR_BASE_CONFIG=configs/experiments/sweep_detector_real_only_100e_base.yaml \
DETECTOR_SWEEP_PROFILE=paper \
DETECTOR_GPUS="0 1 2 3 4 5 6 7" \
DETECTOR_SWEEP_LOG_DIR="logs/detector_real_only_100e_paper" \
DETECTOR_EXTRA_CONFIGS=none \
bash scripts/run_parallel_detector_sweep.sh
```

Official non-YOLO RT-DETR family pack:

```bash
DETECTOR_BASE_CONFIG=configs/experiments/sweep_detector_real_only_100e_base.yaml \
DETECTOR_MODELS="rtdetr-l.yaml rtdetr-x.yaml rtdetr-resnet50.yaml rtdetr-resnet101.yaml" \
DETECTOR_GPUS="0 1 2 3" \
DETECTOR_SWEEP_LOG_DIR="logs/detector_real_only_100e_non_yolo" \
bash scripts/run_parallel_detector_sweep.sh
```

If you only want the ResNet RT-DETR variants:

```bash
DETECTOR_BASE_CONFIG=configs/experiments/sweep_detector_real_only_100e_base.yaml \
DETECTOR_MODELS="rtdetr-resnet50.yaml rtdetr-resnet101.yaml" \
DETECTOR_GPUS="0 1" \
DETECTOR_SWEEP_LOG_DIR="logs/detector_real_only_100e_resnet_only" \
bash scripts/run_parallel_detector_sweep.sh
```

## Monitoring A Running Sweep

If you set `DETECTOR_SWEEP_LOG_DIR`, the main terminal only prints job dispatch and then waits for child processes. This is normal. The detailed output goes into per-job log files.

Start with:

```bash
sed -n '1,20p' logs/detector_sweep_paper/jobs.tsv
```

Then follow one job:

```bash
tail -f logs/detector_sweep_paper/000_model_yolov8m.pt.log
```

Useful patterns in the logs:

- `[yolo-cache] build ...` means the shared YOLO export is being created for that phase
- `[yolo-cache] wait for shared build ...` means another job is already building that same export
- `[yolo-cache] reuse ...` means the dataset was already built and is being reused
- Ultralytics epoch lines mean actual training has started

Useful quick searches:

```bash
rg -n "\\[yolo-cache\\]|Traceback|ERROR|epoch" logs/detector_sweep_paper
```

## Where Outputs Go

### Experiment Outputs

Per-experiment metadata goes under:

- `runs/paleo_experiments/<experiment_name>/`

Typical files there:

- `config_used.yaml`
- `config_resolved.json`
- `run_metadata.json`
- `metrics_detector.json`
- `detector_sequential_log.json`
- `yolo_dataset/dataset_refs.json`

### Shared Dataset Cache

Shared YOLO exports go under:

- `runs/paleo_experiments/_shared_yolo_cache/`

This cache is keyed by manifests and export settings. With the current setup, identical phases are built once and reused by all matching jobs.

### Ultralytics Training Runs

Ultralytics writes phase runs under the configured `detector.project`, typically:

- `runs/detect/<run_name>_ph0/`
- `runs/detect/<run_name>_ph1/`

## After The Sweep

Aggregate experiment-level results:

```bash
python -m paleo_ocr.experiments.aggregate_runs \
  --root runs/paleo_experiments \
  --out-csv runs_summary.csv
```

Then compare the best detector runs:

```bash
python -m paleo_ocr.experiments.compare_runs \
  --csv runs_summary.csv \
  --sort-by map50 \
  --top 10
```

If you want to inspect Ultralytics run folders directly:

```bash
python -m paleo_ocr.experiments.aggregate_yolo_runs \
  --root runs/detect \
  --repo-root . \
  --out-csv yolo_runs_summary.csv
```

Once you choose the best detector, the usual next step is:

```bash
bash scripts/run_parallel_classifier_backbones.sh
```

## Notes For Papers

For paper-quality tables, keep the recipe fixed and vary only the backbone family you are comparing.

Recommended practice:

1. Use `DETECTOR_SWEEP_PROFILE=paper` explicitly, even though it is the default.
2. Use `DETECTOR_EXTRA_CONFIGS=none` if the paper table is supposed to contain only backbone rows.
3. Keep the same base YAML for the whole table, usually `configs/experiments/sweep_detector_stageb_base.yaml`.
4. Keep the same validation manifest across all runs.
5. Save logs to a dedicated folder such as `logs/detector_sweep_paper`.
6. Export a dedicated summary CSV such as `runs_summary_paper.csv`.

Example article run:

```bash
DETECTOR_SWEEP_PROFILE=paper \
DETECTOR_GPUS="0 1 2 3 4 5 6 7" \
DETECTOR_SWEEP_LOG_DIR="logs/detector_sweep_paper" \
DETECTOR_EXTRA_CONFIGS=none \
bash scripts/run_parallel_detector_sweep.sh

python -m paleo_ocr.experiments.aggregate_runs \
  --root runs/paleo_experiments \
  --out-csv runs_summary_paper.csv
```

Do not mix these into the same comparison table unless that is intentional:

- `strongaug` vs `lightaug` vs `noaug`
- different val manifests
- paper subset vs full grid
- backbone rows vs extra control configs

## Common Issues

### Resuming after a failed phase (sequential runs)

If phase 0 finished but phase 1 crashed (for example because weights were looked up under the wrong folder), re-run **only phase 1** after updating the code, or pass weights explicitly:

```bash
python -m paleo_ocr.experiments.run_experiment \
  --config configs/experiments/sweep_detector_stageb_base.yaml \
  --detector-model yolo11m.pt \
  --detector-start-phase 1 \
  --detector-init-weights /home/.../runs/detect/runs/detect/det_sweep_seq_real_strongaug__yolo11m.pt_ph02/weights/best.pt
```

If you omit `--detector-init-weights`, the runner searches under `detector.project` for an Ultralytics run whose folder name matches `run_name_ph{N-1}` or `run_name_ph{N-1}2`, `...3` (Ultralytics increments names when a folder already exists).

Run from the **repository root** so `detector.project: runs/detect` resolves predictably.

### "The terminal is quiet for a long time"

If `DETECTOR_SWEEP_LOG_DIR` is set, the main terminal is not the right place to watch progress. Use the per-job log files.

### "Training did not start immediately"

On the first run, some time may be spent in shared YOLO export creation. Look for `[yolo-cache] build` in the job logs. Later runs with the same manifests should show `[yolo-cache] reuse`.

### "Disk usage grows too quickly"

The current detector base uses:

- `use_symlinks: true` for source images
- `shared_dataset_cache: true` for reusable YOLO exports

That avoids rebuilding the same phase dataset for every model in the sweep.

### "Large backbones run out of memory"

Reduce `detector.batch` in the base YAML or create a smaller custom YAML and point the script at it with:

```bash
DETECTOR_BASE_CONFIG=/abs/path/to/your_detector_config.yaml \
bash scripts/run_parallel_detector_sweep.sh
```
