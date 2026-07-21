#!/usr/bin/env python3
"""Prepare the data/ layout expected by configs/experiments/*.yaml.

Supports three sources (pick what you have locally; do not hardcode public
Hugging Face namespaces in anonymous review materials):

1. Local Stage A / Stage B trees with manifests.
2. A single synthetic export where Stage B images carry ``styled=true`` and
   Stage A supervision is embedded in metadata (boxes/chars). In that case
   Stage B is linked as ``data/stage_b`` and Stage A images can be regenerated
   with ``paleo_ocr.synthetic_v_2_generator`` or supplied separately.
3. Real benchmark export with train/validation/extra splits.

Usage:
  python scripts/prepare_data_layout.py \\
    --real-manifest /path/to/real.jsonl \\
    --real-images /path/to/real_images \\
    --stage-a-root /path/to/stage_a \\
    --stage-b-root /path/to/stage_b \\
    --out data
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    ap.add_argument("--real-manifest", type=str, default="")
    ap.add_argument("--real-images", type=str, default="")
    ap.add_argument("--stage-a-root", type=str, default="", help="dir with all_manifest.jsonl")
    ap.add_argument("--stage-b-root", type=str, default="", help="dir with manifest.jsonl")
    ap.add_argument(
        "--synthetic-unified-root",
        type=str,
        default="",
        help="optional single export where styled images are Stage B",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.real_manifest:
        src = Path(args.real_manifest)
        rows = list(_iter_jsonl(src))
        _write_jsonl(out / "real" / "manifest.jsonl", rows)
        train = [r for r in rows if (r.get("split") or "") == "train"]
        val = [r for r in rows if (r.get("split") or "") in ("validation", "val")]
        if train:
            _write_jsonl(out / "real" / "manifest_train.jsonl", train)
        if val:
            _write_jsonl(out / "real" / "manifest_val.jsonl", val)
            with (out / "real" / "evaluation_row_ids.txt").open("w", encoding="utf-8") as f:
                for r in val:
                    rid = r.get("row_id") or (r.get("meta") or {}).get("row_id")
                    if rid is not None:
                        f.write(str(rid) + "\n")
        print(f"real: {len(rows)} total, train={len(train)}, val={len(val)}")

    if args.real_images:
        _link_or_copy(Path(args.real_images), out / "real" / "images", args.copy)

    if args.stage_a_root:
        root = Path(args.stage_a_root)
        man = root / "all_manifest.jsonl"
        if not man.exists():
            man = root / "manifest.jsonl"
        _link_or_copy(man, out / "stage_a" / "manifest.jsonl", copy=True)
        img = root / "images" if (root / "images").exists() else root
        _link_or_copy(img, out / "stage_a" / "images", args.copy)
        print(f"stage_a linked from {root}")

    if args.stage_b_root:
        root = Path(args.stage_b_root)
        man = root / "manifest.jsonl"
        _link_or_copy(man, out / "stage_b" / "manifest.jsonl", copy=True)
        img = root / "images" if (root / "images").exists() else root
        _link_or_copy(img, out / "stage_b" / "images", args.copy)
        print(f"stage_b linked from {root}")

    if args.synthetic_unified_root:
        # Treat unified styled export as Stage B; Stage A must be supplied
        # separately or regenerated. Document this clearly.
        root = Path(args.synthetic_unified_root)
        man = root / "manifest.jsonl"
        if not man.exists():
            # HF-style: may be parquet-only; user must export JSONL first
            raise SystemExit(
                "synthetic-unified-root needs manifest.jsonl. "
                "If your public export is parquet-only, convert to JSONL first. "
                "A single styled=true collection maps to data/stage_b; Stage A "
                "images are either a separate tree or regenerated by the Stage A generator."
            )
        _link_or_copy(man, out / "stage_b" / "manifest.jsonl", copy=True)
        img = root / "images" if (root / "images").exists() else root
        _link_or_copy(img, out / "stage_b" / "images", args.copy)
        note = out / "DATA_LAYOUT_NOTE.txt"
        note.write_text(
            "Stage B populated from a unified styled export.\n"
            "Stage A images are NOT automatically present in that export.\n"
            "Either provide --stage-a-root or regenerate Stage A with "
            "paleo_ocr.synthetic_v_2_generator and point data/stage_a at it.\n",
            encoding="utf-8",
        )
        print(f"stage_b from unified export; wrote {note}")

    print(f"done -> {out}")


if __name__ == "__main__":
    main()
