# Reproduce tables

## Frozen tables (no training required)

```bash
python scripts/reproduce_tables.py --output results
```

Writes:

| File | Content |
|------|---------|
| `results/table_detector.csv` | mAP50–95 for A / A+R / B+R / R (+ Δ) |
| `results/table_classifier.csv` | Acc1 / macro-F1 for 16 complete sweeps |
| `results/ocr_baselines.csv` | CER / WER / chrF / EM |
| `results/run_manifest.json` | split role, seed, exclusions, regime glossary |

These numbers match the paper tables. The 150-image partition is an
**evaluation / comparative validation split**, not a sealed test set.

## After re-running training

```bash
bash scripts/aggregate_ultralytics_detect_runs.sh
python -m paleo_ocr.experiments.aggregate_runs \
  --root runs/paleo_experiments \
  --out-csv results/runs_summary.csv
python scripts/reproduce_tables.py --output results --runs runs/paleo_experiments --from-runs
```

Sequential classifier metrics are read from
`classifier_run/phase_*/metrics_best.json` (last phase wins).

## Naming reminder

`B+R` in CSVs = Phase 0 on Stage B styled images + real finetune.
Paper text may say `A+B+R` for the same training regime because Stage B is
derived from Stage A; the training set of Phase 0 does **not** mix Stage A and
Stage B images.
