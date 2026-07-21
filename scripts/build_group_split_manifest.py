#!/usr/bin/env python3
"""Build a deterministic group-based train/validation split for a JSONL manifest.

Default behavior:
- group rows by ``row_id``
- preserve the original record order
- target the original number of validation records as closely as possible
- assign the whole group to either ``train`` or ``validation``

This is intended as a safer backup split for classifier validation, where
different photos/views of the same inscription should not cross train/val.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _count_validation_records(rows: List[Dict[str, Any]]) -> int:
    count = 0
    for rec in rows:
        split = str(rec.get("split", "")).lower()
        if split in {"validation", "val"}:
            count += 1
    return count


def _stable_group_order(groups: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    return sorted(
        groups.items(),
        key=lambda kv: hashlib.sha1(kv[0].encode("utf-8")).hexdigest(),
    )


def _assign_validation_groups(
    ordered_groups: List[Tuple[str, List[Dict[str, Any]]]],
    target_validation_records: int,
) -> set[str]:
    sizes = [len(rows) for _key, rows in ordered_groups]
    suffix_remaining = [0] * (len(sizes) + 1)
    for i in range(len(sizes) - 1, -1, -1):
        suffix_remaining[i] = suffix_remaining[i + 1] + sizes[i]

    val_keys: set[str] = set()
    val_count = 0
    for i, (key, rows) in enumerate(ordered_groups):
        size = len(rows)
        remaining_after = suffix_remaining[i + 1]
        if val_count >= target_validation_records:
            continue

        take = False
        current_gap = abs(target_validation_records - val_count)
        take_gap = abs(target_validation_records - (val_count + size))
        if take_gap <= current_gap:
            take = True
        elif remaining_after < (target_validation_records - val_count):
            # Must take this group, otherwise we cannot get close enough anymore.
            take = True

        if take:
            val_keys.add(key)
            val_count += size

    return val_keys


def build_group_split(
    rows: List[Dict[str, Any]],
    *,
    group_key: str,
    target_validation_records: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in rows:
        if group_key not in rec:
            raise KeyError(f"Missing group key {group_key!r} in record: {rec.get('uid')!r}")
        groups[str(rec[group_key])].append(rec)

    ordered_groups = _stable_group_order(groups)
    val_keys = _assign_validation_groups(ordered_groups, target_validation_records)

    out_rows: List[Dict[str, Any]] = []
    train_records = 0
    val_records = 0
    train_groups = 0
    val_groups = 0

    for key, group_rows in ordered_groups:
        split = "validation" if key in val_keys else "train"
        if split == "validation":
            val_groups += 1
        else:
            train_groups += 1
        for rec in group_rows:
            rec2 = dict(rec)
            rec2["split"] = split
            out_rows.append(rec2)
            if split == "validation":
                val_records += 1
            else:
                train_records += 1

    # Preserve original order in output; only mutate split labels.
    split_by_group = {str(rec[group_key]): rec["split"] for rec in out_rows}
    out_rows = []
    for rec in rows:
        rec2 = dict(rec)
        rec2["split"] = split_by_group[str(rec[group_key])]
        out_rows.append(rec2)

    summary = {
        "group_key": group_key,
        "total_records": len(rows),
        "total_groups": len(groups),
        "target_validation_records": int(target_validation_records),
        "train_records": train_records,
        "validation_records": val_records,
        "train_groups": train_groups,
        "validation_groups": val_groups,
        "row_id_overlap_between_train_and_validation": 0,
    }
    return out_rows, summary


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a backup group-based split for a manifest JSONL.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--group-key", type=str, default="row_id")
    p.add_argument(
        "--target-validation-records",
        type=int,
        default=None,
        help="If omitted, reuse the original number of validation records from the input manifest.",
    )
    args = p.parse_args()

    rows = _read_jsonl(args.input)
    target_val = (
        int(args.target_validation_records)
        if args.target_validation_records is not None
        else _count_validation_records(rows)
    )

    out_rows, summary = build_group_split(
        rows,
        group_key=args.group_key,
        target_validation_records=target_val,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, out_rows)

    summary["input"] = str(args.input.resolve())
    summary["output"] = str(args.output.resolve())
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
