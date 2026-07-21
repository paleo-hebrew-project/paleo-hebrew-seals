#!/usr/bin/env python3
"""Build the paper tables from frozen results/ and/or live run directories.

Produces:
  results/table_detector.csv
  results/table_classifier.csv
  results/ocr_baselines.csv   (copied if present)
  results/run_manifest.json

Usage:
  # Prefer frozen published numbers (default):
  python scripts/reproduce_tables.py --output results

  # Rebuild classifier rows from live sequential runs:
  python scripts/reproduce_tables.py --runs runs/paleo_experiments --output results --from-runs
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


DETECTOR_FROZEN = [
    # model, A_map50_95, AR_map50_95, BR_map50_95, R_map50_95, delta_BR_minus_R
    ("YOLOv8-M", 0.091, 0.101, 0.599, 0.499, 0.100),
    ("YOLOv8-X", 0.094, 0.513, 0.132, 0.182, -0.050),
    ("YOLO11-M", 0.073, 0.091, 0.587, 0.344, 0.243),
    ("YOLO11-X", 0.076, 0.097, 0.589, 0.205, 0.384),
    ("YOLO26-M", 0.094, 0.524, 0.587, 0.533, 0.054),
    ("YOLO26-X", 0.071, 0.238, 0.404, 0.511, -0.107),
    ("RT-DETR-L", 0.028, 0.031, 0.450, 0.010, 0.440),
]

# Acc1 / macro-F1 percent. Complete 16-backbone sweeps only.
# Columns: R40, A_only, A+120R, B+120R  (B = Stage-B styled pretrain = paper A+B+R)
CLASSIFIER_FROZEN = [
    ("ConvNeXt-L", "ConvNeXt", 77.57, 71.34, 47.31, 40.63, 74.11, 66.16, 76.45, 71.60),
    ("ConvNeXt-S", "ConvNeXt", 10.69, 4.02, 41.78, 33.12, 70.54, 60.86, 75.96, 69.77),
    ("ConvNeXt-B", "ConvNeXt", 76.85, 71.45, 45.84, 39.83, 73.55, 64.40, 75.40, 70.73),
    ("ConvNeXt-T", "ConvNeXt", 13.50, 1.08, 44.51, 37.79, 73.20, 66.90, 74.28, 66.34),
    ("EfficientNet-B0", "EfficientNet", 54.90, 40.03, 50.24, 43.59, 71.80, 63.74, 68.89, 61.74),
    ("EfficientNet-B1", "EfficientNet", 66.00, 52.05, 44.93, 36.69, 72.78, 65.87, 70.90, 60.87),
    ("EfficientNet-B2", "EfficientNet", 49.76, 34.42, 50.17, 41.65, 73.34, 63.96, 70.10, 61.24),
    ("EfficientNet-B3-NS", "EfficientNet", 53.94, 39.64, 49.83, 42.65, 73.34, 66.35, 71.95, 63.83),
    ("ResNet-34", "ResNet", 48.31, 25.88, 47.87, 41.62, 71.03, 63.96, 52.97, 32.55),
    ("ResNet-101", "ResNet", 46.06, 26.18, 49.34, 42.44, 72.57, 64.39, 51.45, 31.26),
    ("Swin-T", "Swin", 70.98, 60.55, 44.44, 37.19, 72.08, 65.07, 71.62, 62.23),
    ("Swin-S", "Swin", 73.55, 64.72, 47.38, 40.32, 75.44, 68.24, 74.52, 67.69),
    ("Swin-B", "Swin", 75.64, 68.96, 44.93, 38.34, 74.39, 66.32, 75.32, 68.97),
    ("SwinV2-T", "SwinV2", 72.19, 63.19, 42.55, 36.02, 72.64, 65.59, 69.45, 59.96),
    ("ViT-B/16", "ViT", 73.23, 64.23, 39.19, 33.83, 71.31, 62.89, 73.47, 65.01),
    ("ViT-S/16", "ViT", 67.36, 57.65, 40.38, 34.34, 70.82, 62.29, 68.57, 60.60),
]

OCR_FROZEN = [
    ("Tesseract (heb)", 3.400, 7.029, 1.817, 0.000),
    ("Kraken MiDRASH_Gen_01", 2.832, 5.294, 1.811, 0.000),
    ("Kraken BiblIA_01", 2.634, 4.568, 1.046, 0.000),
    ("Kraken Ashkenazi_01", 1.209, 1.784, 0.865, 0.000),
    ("Kraken Italian_01", 1.186, 1.767, 1.049, 0.000),
    ("Kraken Sephardi_01", 1.673, 2.878, 1.225, 0.000),
]

EXCLUDED_CLASSIFIERS = [
    "ResNet-50 (incomplete regime)",
    "SwinV2-B (incomplete / unavailable timm name in env)",
    "SwinV2-S (incomplete regime)",
]


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_frozen(out: Path) -> None:
    det_rows = []
    for m, a, ar, br, r, d in DETECTOR_FROZEN:
        det_rows.append(
            {
                "model": m,
                "A_mAP50_95": a,
                "AplusR_mAP50_95": ar,
                "BplusR_mAP50_95": br,
                "R_mAP50_95": r,
                "delta_BplusR_minus_R": d,
                "note": (
                    "B+R = Phase0 on Stage-B styled images then real finetune; "
                    "Stage B images are derived from Stage A. Paper shorthand A+B+R."
                ),
            }
        )
    _write_csv(
        out / "table_detector.csv",
        [
            "model",
            "A_mAP50_95",
            "AplusR_mAP50_95",
            "BplusR_mAP50_95",
            "R_mAP50_95",
            "delta_BplusR_minus_R",
            "note",
        ],
        det_rows,
    )

    cls_rows = []
    for row in CLASSIFIER_FROZEN:
        (
            name,
            family,
            r40a,
            r40f,
            aa,
            af,
            a120a,
            a120f,
            b120a,
            b120f,
        ) = row
        cls_rows.append(
            {
                "model": name,
                "family": family,
                "R40_acc": r40a,
                "R40_macroF1": r40f,
                "A_only_acc": aa,
                "A_only_macroF1": af,
                "Aplus120R_acc": a120a,
                "Aplus120R_macroF1": a120f,
                "Bplus120R_acc": b120a,
                "Bplus120R_macroF1": b120f,
            }
        )
    _write_csv(
        out / "table_classifier.csv",
        [
            "model",
            "family",
            "R40_acc",
            "R40_macroF1",
            "A_only_acc",
            "A_only_macroF1",
            "Aplus120R_acc",
            "Aplus120R_macroF1",
            "Bplus120R_acc",
            "Bplus120R_macroF1",
        ],
        cls_rows,
    )

    ocr_rows = [
        {"method": m, "CER": cer, "WER": wer, "chrF": chrf, "EM": em}
        for m, cer, wer, chrf, em in OCR_FROZEN
    ]
    _write_csv(
        out / "ocr_baselines.csv",
        ["method", "CER", "WER", "chrF", "EM"],
        ocr_rows,
    )

    manifest = {
        "split_name": "evaluation_split",
        "split_role": (
            "comparative validation / evaluation split; participated in checkpoint "
            "selection; NOT a sealed blind test set"
        ),
        "n_real_images": 307,
        "n_evaluation_images": 150,
        "n_train_images": 157,
        "seed": 42,
        "n_classifier_complete_sweeps": 16,
        "excluded_classifier_backbones": EXCLUDED_CLASSIFIERS,
        "regime_naming": {
            "A": "Stage A structural synth only",
            "A+R": "Stage A pretrain then real finetune",
            "B+R": (
                "Phase 0 trains on Stage B styled images only, then real finetune. "
                "Stage B images are produced from Stage A. Paper shorthand: A+B+R."
            ),
            "R": "real only",
            "R40": "real only, 40 epochs (classifier reference)",
        },
        "runs_per_configuration": 1,
        "variation": (
            "Single seed (42). Seed-induced variance was not estimated for the full "
            "grid; a small multi-seed subset is recommended for the camera-ready."
        ),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="results")
    ap.add_argument("--runs", type=str, default=None, help="optional live runs root")
    ap.add_argument(
        "--from-runs",
        action="store_true",
        help="also call paleo_ocr.experiments.aggregate_runs if --runs is set",
    )
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_frozen(out)
    print(f"Wrote frozen paper tables under {out}/")

    if args.from_runs and args.runs:
        from paleo_ocr.experiments import aggregate_runs as agg

        # Reuse aggregator CLI entry by writing a side CSV
        import sys

        sys.argv = ["aggregate_runs", "--root", args.runs, "--out-csv", str(out / "runs_summary.csv")]
        agg.main()
        print(f"Also wrote live aggregation to {out / 'runs_summary.csv'}")


if __name__ == "__main__":
    main()
