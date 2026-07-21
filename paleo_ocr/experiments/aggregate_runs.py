"""Scan experiment output dirs and append rows to runs_summary.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="e.g. runs/paleo_experiments")
    p.add_argument("--out-csv", type=str, default="runs_summary.csv")
    return p.parse_args()


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows: List[Dict[str, Any]] = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        meta = _read_json(exp_dir / "run_metadata.json")
        cfg = _read_json(exp_dir / "config_resolved.json")
        det_m = _read_json(exp_dir / "metrics_detector.json") if (exp_dir / "metrics_detector.json").exists() else None
        cls_dir = exp_dir / "classifier_run"
        cls_m: Optional[Dict[str, Any]] = None
        if cls_dir.is_dir():
            if (cls_dir / "metrics_best.json").exists():
                cls_m = _read_json(cls_dir / "metrics_best.json")
            elif (cls_dir / "metrics_classifier.json").exists():
                cls_m = _read_json(cls_dir / "metrics_classifier.json")

        row: Dict[str, Any] = {"experiment_dir": str(exp_dir.name)}
        if cfg:
            row["task"] = cfg.get("task")
            row["experiment_name"] = cfg.get("experiment_name")
            row["seed"] = cfg.get("seed")
            row["mixing_mode"] = (cfg.get("mixing") or {}).get("mode")
            det_cfg = cfg.get("detector") or {}
            cls_cfg = cfg.get("classifier") or {}
            row["detector_model"] = det_cfg.get("model")
            row["detector_regime"] = det_cfg.get("regime")
            row["classifier_model"] = cls_cfg.get("model")
        if meta:
            row["git_hash"] = meta.get("git_hash")
        if det_m:
            row["map50"] = det_m.get("mAP50")
            row["map50_95"] = det_m.get("mAP50_95")
        if cls_m:
            ma = cls_m.get("metrics_all") or {}
            if ma:
                row["val_acc1"] = cls_m.get("val_acc1")
                row["macro_f1"] = ma.get("macro_f1")
                row["weighted_f1"] = ma.get("weighted_f1")
            else:
                # Standalone eval_classifier output (metrics_classifier.json): flat keys
                row["val_acc1"] = cls_m.get("val_acc1")
                row["macro_f1"] = cls_m.get("macro_f1")
                row["weighted_f1"] = cls_m.get("weighted_f1")
        rows.append(row)

    if not rows:
        print("No runs found.")
        return

    fieldnames = sorted({k for r in rows for k in r.keys()})
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
