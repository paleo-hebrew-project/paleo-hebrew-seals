"""Print or sort rows from aggregate_runs output (runs_summary.csv) for quick comparison."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare experiment rows from runs_summary.csv")
    p.add_argument("--csv", type=str, required=True, help="Path to runs_summary.csv from aggregate_runs")
    p.add_argument(
        "--sort-by",
        type=str,
        default=None,
        help="Numeric column to sort by (e.g. macro_f1, map50, val_acc1). Descending unless --asc",
    )
    p.add_argument("--asc", action="store_true", help="Sort ascending")
    p.add_argument("--top", type=int, default=0, help="Print only first N rows after sort (0 = all)")
    return p.parse_args()


def _to_float(x: Any) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    args = parse_args()
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    with path.open(encoding="utf-8", newline="") as f:
        rows: List[Dict[str, Any]] = list(csv.DictReader(f))

    if not rows:
        print("No rows in CSV.")
        return

    if args.sort_by:
        key = args.sort_by
        if key not in rows[0]:
            raise SystemExit(f"Unknown column {key!r}. Available: {sorted(rows[0].keys())}")
        rev = not args.asc
        rows.sort(key=lambda r: _to_float(r.get(key)), reverse=rev)

    if args.top and args.top > 0:
        rows = rows[: args.top]

    fieldnames = list(rows[0].keys())
    w = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})


if __name__ == "__main__":
    main()
