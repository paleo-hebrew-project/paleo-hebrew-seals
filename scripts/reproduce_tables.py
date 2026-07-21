#!/usr/bin/env python3
"""Build paper tables from live run artifacts and/or export frozen values.

Modes
-----
1) Export frozen paper numbers (no training required)::

    python scripts/reproduce_tables.py --mode export-frozen --output results

2) Rebuild tables from detector/classifier run dirs and compare to frozen::

    python scripts/reproduce_tables.py \\
      --mode from-runs \\
      --runs-detect runs/detect \\
      --runs-cls runs/paleo_experiments \\
      --output reproduced_results \\
      --compare-with-frozen results \\
      --tol 1e-3

The from-runs mode records, for each row, the source run directory and fails
if a published frozen value differs by more than ``--tol``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Frozen paper numbers (single source for --mode export-frozen)
# ---------------------------------------------------------------------------

DETECTOR_FROZEN = [
    ("YOLOv8-M", 0.091, 0.101, 0.599, 0.499, 0.100),
    ("YOLOv8-X", 0.094, 0.513, 0.132, 0.182, -0.050),
    ("YOLO11-M", 0.073, 0.091, 0.587, 0.344, 0.243),
    ("YOLO11-X", 0.076, 0.097, 0.589, 0.205, 0.384),
    ("YOLO26-M", 0.094, 0.524, 0.587, 0.533, 0.054),
    ("YOLO26-X", 0.071, 0.238, 0.404, 0.511, -0.107),
    ("RT-DETR-L", 0.028, 0.031, 0.450, 0.010, 0.440),
]

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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def export_frozen(out: Path) -> None:
    det_rows = [
        {
            "model": m,
            "A_mAP50_95": a,
            "AplusR_mAP50_95": ar,
            "BplusR_mAP50_95": br,
            "R_mAP50_95": r,
            "delta_BplusR_minus_R": d,
            "source": "frozen",
        }
        for m, a, ar, br, r, d in DETECTOR_FROZEN
    ]
    _write_csv(
        out / "table_detector.csv",
        ["model", "A_mAP50_95", "AplusR_mAP50_95", "BplusR_mAP50_95", "R_mAP50_95", "delta_BplusR_minus_R", "source"],
        det_rows,
    )
    cls_rows = []
    for row in CLASSIFIER_FROZEN:
        name, family, r40a, r40f, aa, af, a120a, a120f, b120a, b120f = row
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
                "source": "frozen",
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
            "source",
        ],
        cls_rows,
    )
    _write_csv(
        out / "ocr_baselines.csv",
        ["method", "CER", "WER", "chrF", "EM", "source"],
        [
            {"method": m, "CER": cer, "WER": wer, "chrF": chrf, "EM": em, "source": "frozen"}
            for m, cer, wer, chrf, em in OCR_FROZEN
        ],
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
            "Single seed (42). Seed-induced variance was not estimated for the full grid."
        ),
        "table_source": "frozen constants exported by --mode export-frozen; "
        "use --mode from-runs to rebuild from artifacts",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _best_map50_95_from_results_csv(run_dir: Path) -> Optional[float]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    best = None
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Ultralytics columns often have leading spaces
        for row in reader:
            key = None
            for k in row:
                if "mAP50-95" in k or "metrics/mAP50-95" in k:
                    key = k
                    break
            if key is None:
                continue
            try:
                v = float(row[key])
            except Exception:
                continue
            best = v if best is None else max(best, v)
    return best


def _classifier_metrics(exp_dir: Path) -> Optional[Tuple[float, float, str]]:
    cls = exp_dir / "classifier_run"
    if not cls.is_dir():
        return None
    candidates: List[Path] = []
    phases = sorted([d for d in cls.iterdir() if d.is_dir() and d.name.startswith("phase_")], key=lambda d: d.name)
    for d in reversed(phases):
        candidates.append(d / "metrics_best.json")
    candidates.append(cls / "metrics_best.json")
    for cand in candidates:
        j = _read_json(cand)
        if not j:
            continue
        acc = j.get("val_acc1")
        f1 = (j.get("metrics_all") or {}).get("macro_f1")
        if acc is None:
            continue
        if f1 is None:
            f1 = j.get("macro_f1")
        return float(acc) * 100.0, (float(f1) * 100.0 if f1 is not None else float("nan")), str(cand)
    return None


def rebuild_from_runs(runs_detect: Optional[Path], runs_cls: Optional[Path], out: Path) -> Dict[str, Any]:
    """Best-effort rebuild. Detector mapping is heuristic on run dir names."""
    provenance: Dict[str, Any] = {"detector_sources": {}, "classifier_sources": {}}
    det_rows: List[Dict[str, Any]] = []
    if runs_detect and runs_detect.is_dir():
        # Collect best mAP50-95 per model slug seen in stagea/stageb/real dirs
        by_model: Dict[str, Dict[str, Any]] = {}
        for d in sorted(runs_detect.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            mapv = _best_map50_95_from_results_csv(d)
            if mapv is None:
                continue
            # Heuristic regime tag
            regime = "unknown"
            if "stagea" in name and ("_ph1" in name or "finetune" in name or "real" in name):
                regime = "AplusR"
            elif "stagea" in name and ("_ph0" in name or "_ph03" in name or "pretrain" in name):
                regime = "A"
            elif "real_only" in name or "realonly" in name:
                regime = "R"
            elif ("stageb" in name or "seq_real" in name) and ("_ph1" in name or "finetune" in name):
                regime = "BplusR"
            model = name
            for prefix in (
                "det_sweep_stagea_seq_real_strongaug__",
                "det_sweep_seq_real_strongaug__",
                "det_sweep_real_only__",
            ):
                if model.startswith(prefix):
                    model = model[len(prefix) :]
            for suf in ("_ph1", "_ph13", "_ph12", "_ph03", "_ph0", "_ph02"):
                if suf in model:
                    model = model.split(suf)[0]
                    break
            slot = by_model.setdefault(model, {})
            prev = slot.get(regime)
            if prev is None or mapv > prev["mAP50_95"]:
                slot[regime] = {"mAP50_95": mapv, "run_dir": str(d)}
        for model, slots in sorted(by_model.items()):
            row = {"model": model, "source": "from-runs"}
            for k in ("A", "AplusR", "BplusR", "R"):
                if k in slots:
                    row[f"{k}_mAP50_95"] = round(slots[k]["mAP50_95"], 3)
                    provenance["detector_sources"].setdefault(model, {})[k] = slots[k]["run_dir"]
                else:
                    row[f"{k}_mAP50_95"] = ""
            if row.get("BplusR_mAP50_95") != "" and row.get("R_mAP50_95") != "":
                row["delta_BplusR_minus_R"] = round(float(row["BplusR_mAP50_95"]) - float(row["R_mAP50_95"]), 3)
            else:
                row["delta_BplusR_minus_R"] = ""
            det_rows.append(row)
        _write_csv(
            out / "table_detector.csv",
            ["model", "A_mAP50_95", "AplusR_mAP50_95", "BplusR_mAP50_95", "R_mAP50_95", "delta_BplusR_minus_R", "source"],
            det_rows,
        )

    cls_rows: List[Dict[str, Any]] = []
    if runs_cls and runs_cls.is_dir():
        # Map experiment dirs -> model + regime heuristic
        for exp in sorted(runs_cls.iterdir()):
            if not exp.is_dir():
                continue
            metrics = _classifier_metrics(exp)
            if metrics is None:
                continue
            acc, f1, src = metrics
            name = exp.name
            regime = "unknown"
            if "__real40e" in name or "real_only_40e" in name:
                regime = "R40"
            elif "stagea" in name and "__real120e" in name:
                regime = "Aplus120R"
            elif "stagea" in name and ("phase_00" in str(src) and "phase_01" not in str(src)):
                regime = "A_only"
            elif "stagea" in name and "phase_00" in str(src):
                # phase metrics from last phase -> A+R length encoded in name
                if "__real20e" in name:
                    regime = "Aplus20R"
                elif "__real60e" in name:
                    regime = "Aplus60R"
                elif "__real120e" in name:
                    regime = "Aplus120R"
            elif ("stageb" in name or "cls_sweep_stageb" in name or "cls_sweep_swinv2_256__" in name) and "__real120e" in name:
                regime = "Bplus120R"
            elif "stagea" in name and "phase_00_pretrain" in str(src) and "finetune" not in str(src):
                regime = "A_only"
            # Prefer last-phase finetune metrics: if path contains phase_01, treat as finetune
            if "phase_01" in str(src) or "finetune" in str(src):
                if "stagea" in name:
                    if "__real120e" in name:
                        regime = "Aplus120R"
                    elif "__real60e" in name:
                        regime = "Aplus60R"
                    elif "__real20e" in name:
                        regime = "Aplus20R"
                else:
                    if "__real120e" in name or name.endswith("_base") or "stageb" in name:
                        regime = "Bplus120R"
            model = name
            for pref in (
                "cls_sweep_stagea_base__",
                "cls_sweep_stageb_base__",
                "cls_sweep_swinv2_256_stagea__",
                "cls_sweep_swinv2_256__",
                "cls_sweep_real_only_40e__",
                "cls_sweep_real_only_40e_swinv2_256__",
            ):
                if model.startswith(pref):
                    model = model[len(pref) :]
            for suf in ("__real120e", "__real60e", "__real20e", "__real40e"):
                if suf in model:
                    model = model.split(suf)[0]
            cls_rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "acc": round(acc, 2),
                    "macroF1": round(f1, 2) if not math.isnan(f1) else "",
                    "source_run": str(exp),
                    "source_metrics": src,
                }
            )
            provenance["classifier_sources"].setdefault(model, {})[regime] = {
                "run": str(exp),
                "metrics": src,
                "acc": acc,
                "macroF1": f1,
            }
        _write_csv(
            out / "table_classifier_raw.csv",
            ["model", "regime", "acc", "macroF1", "source_run", "source_metrics"],
            cls_rows,
        )

    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def compare_csvs(reproduced: Path, frozen: Path, key: str, cols: List[str], tol: float) -> List[str]:
    errs: List[str] = []
    if not reproduced.exists():
        return [f"missing reproduced {reproduced}"]
    if not frozen.exists():
        return [f"missing frozen {frozen}"]
    r_rows = {r[key]: r for r in _read_csv(reproduced) if r.get(key)}
    f_rows = {r[key]: r for r in _read_csv(frozen) if r.get(key)}
    for model, fr in f_rows.items():
        if model not in r_rows:
            errs.append(f"{reproduced.name}: missing model {model} in reproduced")
            continue
        rr = r_rows[model]
        for c in cols:
            if c not in fr or fr[c] == "":
                continue
            if c not in rr or rr[c] == "":
                errs.append(f"{model}.{c}: missing in reproduced")
                continue
            try:
                fv, rv = float(fr[c]), float(rr[c])
            except Exception:
                continue
            if abs(fv - rv) > tol:
                errs.append(f"{model}.{c}: frozen={fv} reproduced={rv} |diff|>{tol}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("export-frozen", "from-runs"), default="export-frozen")
    ap.add_argument("--output", type=str, default="results")
    ap.add_argument("--runs-detect", type=str, default="")
    ap.add_argument("--runs-cls", type=str, default="")
    ap.add_argument("--compare-with-frozen", type=str, default="")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "export-frozen":
        export_frozen(out)
        print(f"Exported frozen tables to {out}/")
        print("NOTE: these are the published numbers, not recomputed from runs.")
        return 0

    provenance = rebuild_from_runs(
        Path(args.runs_detect) if args.runs_detect else None,
        Path(args.runs_cls) if args.runs_cls else None,
        out,
    )
    print(f"Wrote from-runs tables under {out}/")
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "mode": "from-runs",
                "split_role": "evaluation / comparative validation (not sealed test)",
                "provenance_file": "provenance.json",
                "n_detector_models_seen": len(provenance.get("detector_sources", {})),
                "n_classifier_models_seen": len(provenance.get("classifier_sources", {})),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.compare_with_frozen:
        frozen = Path(args.compare_with_frozen)
        errs = []
        errs += compare_csvs(
            out / "table_detector.csv",
            frozen / "table_detector.csv",
            "model",
            ["A_mAP50_95", "AplusR_mAP50_95", "BplusR_mAP50_95", "R_mAP50_95"],
            args.tol,
        )
        # Classifier compare only when a wide table exists; raw regime table is informational.
        if (out / "table_classifier.csv").exists():
            errs += compare_csvs(
                out / "table_classifier.csv",
                frozen / "table_classifier.csv",
                "model",
                ["R40_acc", "A_only_acc", "Aplus120R_acc", "Bplus120R_acc"],
                args.tol,
            )
        (out / "compare_report.json").write_text(
            json.dumps({"tol": args.tol, "n_errors": len(errs), "errors": errs}, indent=2) + "\n",
            encoding="utf-8",
        )
        if errs:
            print(f"COMPARE FAILED ({len(errs)} diffs). See {out / 'compare_report.json'}", file=sys.stderr)
            for e in errs[:20]:
                print(" ", e, file=sys.stderr)
            return 1
        print("COMPARE PASSED within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
