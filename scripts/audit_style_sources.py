#!/usr/bin/env python3
"""Audit Stage B style sources against the evaluation-split row_id set.

Writes:
  data/style_sources/leakage_audit.json

Usage:
  python scripts/audit_style_sources.py \\
    --style-manifest data/style_sources/manifest.jsonl \\
    --excluded-row-ids data/real/evaluation_row_ids.txt \\
    --out data/style_sources/leakage_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_excluded(path: Path) -> Set[str]:
    out: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def _row_id(rec: dict) -> Optional[str]:
    rid = rec.get("row_id")
    if rid is None:
        rid = (rec.get("meta") or {}).get("row_id")
    return None if rid is None else str(rid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style-manifest", type=str, required=True)
    ap.add_argument("--excluded-row-ids", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/style_sources/leakage_audit.json")
    args = ap.parse_args()

    style_path = Path(args.style_manifest)
    excl_path = Path(args.excluded_row_ids)
    if not style_path.exists():
        print(f"ERROR: missing {style_path}", file=sys.stderr)
        return 2
    if not excl_path.exists():
        print(f"ERROR: missing {excl_path}", file=sys.stderr)
        return 2

    excluded = _load_excluded(excl_path)
    style_ids: Set[str] = set()
    missing_rid = 0
    n = 0
    for rec in _iter_jsonl(style_path):
        n += 1
        rid = _row_id(rec)
        if rid is None:
            missing_rid += 1
        else:
            style_ids.add(rid)

    overlap = sorted(style_ids & excluded)
    report: Dict[str, Any] = {
        "style_manifest": str(style_path),
        "excluded_row_ids_file": str(excl_path),
        "n_style_records": n,
        "n_style_row_ids": len(style_ids),
        "n_style_missing_row_id": missing_rid,
        "n_excluded_row_ids": len(excluded),
        "n_overlap": len(overlap),
        "overlap_row_ids": overlap,
        "passed": len(overlap) == 0 and missing_rid == 0,
        "notes": (
            "passed=true requires zero overlap and every style record carrying a row_id. "
            "Regenerate Stage B only with --style-manifest and --excluded-row-ids."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_overlap", "n_style_missing_row_id", "passed")}, indent=2))
    if not report["passed"]:
        print(f"FAILED audit -> {out}", file=sys.stderr)
        return 1
    print(f"PASSED audit -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
