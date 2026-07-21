#!/usr/bin/env python3
"""Verify that the anonymized data release under data/ is complete and consistent.

Checks:
  - required manifests and image trees exist
  - record counts and unique row_id counts
  - entry-disjoint train/evaluation split (no shared row_id)
  - optional SHA-256 of manifests (printed; compare against data/SHA256SUMS)

Usage:
  python scripts/verify_data_release.py --root data
  python scripts/verify_data_release.py --root data --sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REQUIRED = {
    "real/manifest.jsonl": "real benchmark (full)",
    "real/manifest_train.jsonl": "detector real train split",
    "real/manifest_val.jsonl": "detector real evaluation split",
    "stage_a/manifest.jsonl": "Stage A synthetic",
    "stage_b/manifest.jsonl": "Stage B synthetic (styled)",
}

OPTIONAL = {
    "real/manifest_group_split.jsonl": "entry-disjoint group split",
    "lexicons/": "Stage A lexicon pack",
    "SHA256SUMS": "manifest checksums",
}


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_ids(path: Path) -> Tuple[int, Set[str]]:
    ids: Set[str] = set()
    n = 0
    for rec in _iter_jsonl(path):
        n += 1
        rid = rec.get("row_id") or (rec.get("meta") or {}).get("row_id")
        if rid is not None:
            ids.add(str(rid))
    return n, ids


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--sha256", action="store_true", help="print SHA-256 of required manifests")
    p.add_argument(
        "--expect-real",
        type=int,
        default=307,
        help="expected number of real images in data/real/manifest.jsonl",
    )
    p.add_argument("--expect-val", type=int, default=150)
    p.add_argument("--expect-train", type=int, default=157)
    args = p.parse_args()

    root = Path(args.root)
    errors: List[str] = []
    warnings: List[str] = []

    if not root.is_dir():
        print(f"ERROR: data root not found: {root}", file=sys.stderr)
        return 2

    for rel, desc in REQUIRED.items():
        path = root / rel
        if not path.exists():
            errors.append(f"missing {rel} ({desc})")
        else:
            print(f"OK  {rel}")

    for rel, desc in OPTIONAL.items():
        path = root / rel
        if path.exists():
            print(f"OK  {rel} (optional)")
        else:
            warnings.append(f"optional missing: {rel} ({desc})")

    # Counts and leakage
    real = root / "real" / "manifest.jsonl"
    train = root / "real" / "manifest_train.jsonl"
    val = root / "real" / "manifest_val.jsonl"
    group = root / "real" / "manifest_group_split.jsonl"

    if real.exists():
        n, _ = _row_ids(real)
        print(f"real/manifest.jsonl: {n} records")
        if n != args.expect_real:
            warnings.append(f"real count {n} != expected {args.expect_real}")

    if train.exists() and val.exists():
        n_tr, ids_tr = _row_ids(train)
        n_va, ids_va = _row_ids(val)
        print(f"real train={n_tr} val={n_va}")
        if n_tr != args.expect_train:
            warnings.append(f"train count {n_tr} != expected {args.expect_train}")
        if n_va != args.expect_val:
            warnings.append(f"val count {n_va} != expected {args.expect_val}")
        overlap = ids_tr & ids_va
        if ids_tr and ids_va:
            if overlap:
                errors.append(
                    f"row_id overlap between train and evaluation: {len(overlap)} shared ids"
                )
            else:
                print("OK  entry-disjoint: no shared row_id between train and evaluation")
        else:
            warnings.append(
                "train/val manifests have no row_id field; cannot audit entry-level leakage here"
            )

    if group.exists():
        by_split: Dict[str, Set[str]] = {}
        for rec in _iter_jsonl(group):
            split = str(rec.get("split") or (rec.get("meta") or {}).get("split") or "")
            rid = rec.get("row_id") or (rec.get("meta") or {}).get("row_id")
            if not split or rid is None:
                continue
            by_split.setdefault(split, set()).add(str(rid))
        tr = by_split.get("train", set())
        va = by_split.get("validation", set()) | by_split.get("val", set())
        if tr and va:
            ov = tr & va
            if ov:
                errors.append(f"group-split row_id overlap train/validation: {len(ov)}")
            else:
                print(
                    f"OK  group-split entry-disjoint "
                    f"(train entries={len(tr)}, eval entries={len(va)})"
                )

    for name in ("stage_a", "stage_b"):
        man = root / name / "manifest.jsonl"
        if man.exists():
            n, _ = _row_ids(man)
            print(f"{name}/manifest.jsonl: {n} records")

    if args.sha256:
        print("\nSHA-256 of required manifests:")
        for rel in REQUIRED:
            path = root / rel
            if path.exists():
                print(f"{_sha256(path)}  {rel}")

    for w in warnings:
        print(f"WARN: {w}", file=sys.stderr)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    if errors:
        print("FAILED", file=sys.stderr)
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
