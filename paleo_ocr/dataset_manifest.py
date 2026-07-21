"""dataset_manifest
===================

Build a single "source of truth" manifest for the Paleo‑Hebrew seal images.

Why this module exists
----------------------
Your project has multiple pipelines (baseline OCR engines, synthetic data,
detector+classifier, translation, etc.).  Without a unified manifest, each
pipeline ends up re‑implementing slightly different dataset loading, path
handling (folder vs zip), text normalization, and train/val/test splits.

This module builds a per‑image manifest (JSONL or CSV) from:

* ``seals-fixed.csv`` (metadata; 1 row per seal)
* ``mapping.json`` (ground-truth strings + list of images per seal)
* images stored either in a directory or inside a ZIP archive

Important: the dataset does NOT include per-character bounding boxes.
Therefore the manifest is *image-level* (OCR line) only.  Fields for
future bbox annotations are included but default to null/empty.

Outputs
-------
Each JSONL record corresponds to ONE image:

* uid, row_id
* image reference (directory path or zip "inner" path)
* raw GT texts (Hebrew / English / Transliteration)
* a few normalized GT variants useful for CER/WER
* optional image metadata (width/height, sha256)
* optional seal metadata from the CSV (shape, border, etc.)

CLI usage
---------
Build manifest:

    python dataset_manifest.py build \
      --images-root /path/to/seals_images_jpeg.zip \
      --seals-csv /path/to/seals-fixed.csv \
      --out manifest.jsonl \
      --format jsonl \
      --split-ratios 0.8 0.1 0.1 \
      --seed 42 \
      --compute-image-metadata

Show stats:

    python dataset_manifest.py stats --manifest manifest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import posixpath
import random
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pandas is required for dataset_manifest.py (pip install pandas)."
    ) from e

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Pillow is required for dataset_manifest.py (pip install pillow)."
    ) from e


_WS_RE = re.compile(r"\s+")


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False)


def normalize_hebrew_variants(text: Optional[str]) -> Dict[str, Optional[str]]:
    """Create a few *lightweight* normalization variants.

    We intentionally keep normalization conservative.  The baseline evaluator
    can choose which field to use (raw vs no_space vs stripped).
    """

    if text is None:
        return {
            "raw": None,
            "stripped": None,
            "collapsed_ws": None,
            "no_space": None,
            "no_brackets": None,
        }

    raw = str(text)
    stripped = raw.strip()
    collapsed_ws = _WS_RE.sub(" ", stripped)
    no_space = re.sub(r"\s+", "", collapsed_ws)
    # Keep the content, remove the bracket markers.
    no_brackets = collapsed_ws.replace("[", "").replace("]", "")

    return {
        "raw": raw,
        "stripped": stripped,
        "collapsed_ws": collapsed_ws,
        "no_space": no_space,
        "no_brackets": no_brackets,
    }


def read_json_from_zip(zip_path: str, inner_path: str) -> Any:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner_path) as f:
            return json.load(f)


def find_mapping_in_zip(zip_path: str) -> str:
    """Find mapping.json inside the zip; prefer the shortest path."""
    with zipfile.ZipFile(zip_path) as zf:
        cands = [n for n in zf.namelist() if n.lower().endswith("mapping.json")]
    if not cands:
        raise FileNotFoundError(
            "mapping.json not found inside the zip. Pass --mapping explicitly."
        )
    cands.sort(key=lambda s: (len(s), s))
    return cands[0]


def open_image_any(images_root: str, image_ref: Dict[str, Any]) -> Image.Image:
    """Open an image from a directory or a zip container."""
    container = image_ref.get("container")
    if container and container.get("type") == "zip":
        zip_path = container["path"]
        inner = container["inner_path"]
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner) as f:
                img = Image.open(f)
                return img.convert("RGB")
    # directory
    path = image_ref["path"]
    img = Image.open(path)
    return img.convert("RGB")


def sha256_of_image_any(images_root: str, image_ref: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    container = image_ref.get("container")
    if container and container.get("type") == "zip":
        zip_path = container["path"]
        inner = container["inner_path"]
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        return h.hexdigest()
    # directory
    with open(image_ref["path"], "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_seals_csv(csv_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not csv_path:
        return None
    df = pd.read_csv(csv_path)
    if "ID" not in df.columns:
        raise ValueError("Expected column 'ID' in seals-fixed.csv")
    df = df.copy()
    df["ID"] = df["ID"].astype(str)
    df = df.set_index("ID", drop=False)
    return df


def load_mapping(images_root: str, mapping_path: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load mapping.json.

    Returns
    -------
    mapping : dict
        Keys are row_id strings.
    mapping_info : dict
        Information needed to resolve image paths.
    """
    is_zip = images_root.lower().endswith(".zip")
    if is_zip:
        if mapping_path is None:
            mapping_path = find_mapping_in_zip(images_root)
        mapping = read_json_from_zip(images_root, mapping_path)
        base_dir = posixpath.dirname(mapping_path)
        mapping_info = {
            "is_zip": True,
            "zip_path": images_root,
            "mapping_inner": mapping_path,
            "base_dir": base_dir,
        }
        return mapping, mapping_info

    # directory
    if mapping_path is None:
        # try standard location
        cand = Path(images_root) / "mapping.json"
        if cand.exists():
            mapping_path = str(cand)
        else:
            raise FileNotFoundError(
                "mapping.json not found next to images-root. Pass --mapping explicitly."
            )
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    mapping_info = {
        "is_zip": False,
        "dir_path": images_root,
        "mapping_path": mapping_path,
        "base_dir": images_root,
    }
    return mapping, mapping_info


def build_image_ref(rel_image_path: str, mapping_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build a uniform image reference for both zip and directory storage."""
    if mapping_info.get("is_zip"):
        base_dir = mapping_info.get("base_dir", "")
        inner = posixpath.join(base_dir, rel_image_path) if base_dir else rel_image_path
        return {
            "path": rel_image_path,  # as listed in mapping
            "container": {
                "type": "zip",
                "path": mapping_info["zip_path"],
                "inner_path": inner,
            },
        }

    return {
        "path": str(Path(mapping_info["dir_path"]) / rel_image_path),
        "container": None,
    }


def _coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and (x != x)):
            return None
        return float(x)
    except Exception:
        return None


def build_manifest_records(
    images_root: str,
    mapping: Dict[str, Any],
    mapping_info: Dict[str, Any],
    seals_df: Optional[pd.DataFrame] = None,
    compute_image_metadata: bool = False,
    compute_sha256: bool = False,
    include_csv_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """Build per-image manifest records."""

    records: List[Dict[str, Any]] = []

    # Optional existence check helpers
    zip_names: Optional[set] = None
    if mapping_info.get("is_zip"):
        try:
            with zipfile.ZipFile(mapping_info["zip_path"]) as zf:
                zip_names = set(zf.namelist())
        except Exception:
            zip_names = None

    for row_id, item in mapping.items():
        if not isinstance(item, dict):
            continue
        images = item.get("Images") or []
        if not isinstance(images, list):
            images = []

        # base GT fields from mapping
        heb = item.get("Hebrew")
        eng = item.get("English")
        trl = item.get("Transliteration")

        # optional metadata from CSV
        csv_meta: Dict[str, Any] = {}
        if include_csv_metadata and seals_df is not None and str(row_id) in seals_df.index:
            row = seals_df.loc[str(row_id)]
            # Keep a small, stable subset. You can extend as needed.
            keep_cols = [
                "Base shape",
                "Border",
                "Border type",
                "Divider",
                "Divider type",
                "Hoard/Group",
                "Iconic",
                "Iconographic motifs",
                "Attested as",
                "Gender marked",
                "Belonging mark (lamed)",
                "Filiation mark (bn/bt)",
                "Elohistic name",
            ]
            for c in keep_cols:
                if c in seals_df.columns:
                    v = row.get(c)
                    if isinstance(v, float) and (v != v):  # NaN
                        v = None
                    csv_meta[c] = v

        # normalization variants
        heb_variants = normalize_hebrew_variants(heb)
        eng_variants = {"raw": eng if eng is None else str(eng)}
        trl_variants = {"raw": trl if trl is None else str(trl)}

        for img_i, rel_img in enumerate(images):
            uid = f"{row_id}_{img_i:02d}" if len(images) > 1 else str(row_id)
            image_ref = build_image_ref(rel_img, mapping_info)

            # Existence check (helps detect mapping/corpus drift)
            exists = True
            container = image_ref.get("container")
            if container and container.get("type") == "zip":
                inner = container.get("inner_path")
                if zip_names is not None:
                    exists = inner in zip_names
            else:
                exists = os.path.exists(image_ref["path"])

            rec: Dict[str, Any] = {
                "uid": uid,
                "row_id": int(row_id) if str(row_id).isdigit() else str(row_id),
                "image": {
                    "rel_path": rel_img,
                    **image_ref,
                    "exists": bool(exists),
                },
                "gt": {
                    "hebrew": heb_variants,
                    "english": eng_variants,
                    "transliteration": trl_variants,
                    # placeholder for future per-char / per-line annotations
                    "bboxes": None,
                },
                "meta": {
                    "source": "seals-fixed",
                    "csv": csv_meta or None,
                    # unknown a priori; can be filled by an orientation module later
                    "orientation_deg": None,
                    "mirrored": None,
                },
                "split": None,  # assigned later
            }

            if (compute_image_metadata or compute_sha256) and exists:
                try:
                    if compute_image_metadata:
                        img = open_image_any(images_root, image_ref)
                        w, h = img.size
                        rec["image"]["width"] = int(w)
                        rec["image"]["height"] = int(h)
                    if compute_sha256:
                        rec["image"]["sha256"] = sha256_of_image_any(images_root, image_ref)
                except Exception as e:
                    rec["image"]["error"] = f"{type(e).__name__}: {e}"
            elif (compute_image_metadata or compute_sha256) and not exists:
                rec["image"]["error"] = "FileNotFound"

            records.append(rec)

    return records


def assign_splits_grouped(
    records: List[Dict[str, Any]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    group_field: str = "row_id",
) -> None:
    """Assign train/val/test splits while keeping groups together.

    Default grouping is by row_id (all photos of the same seal stay in the same
    split).  We aim for split ratios by *image count* using a greedy fill.
    """

    train_r, val_r, test_r = ratios
    if abs((train_r + val_r + test_r) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    # group -> indices
    groups: Dict[Any, List[int]] = {}
    for i, r in enumerate(records):
        key = r.get(group_field)
        groups.setdefault(key, []).append(i)

    group_items = [(k, idxs) for k, idxs in groups.items()]
    rng = random.Random(seed)
    rng.shuffle(group_items)

    total_imgs = len(records)
    targets = {
        "train": int(round(total_imgs * train_r)),
        "val": int(round(total_imgs * val_r)),
        "test": total_imgs,  # remainder
    }
    order = ["train", "val", "test"]
    cur = 0
    counts = {"train": 0, "val": 0, "test": 0}

    for _, idxs in group_items:
        split_name = order[cur]
        # move to next split if we reached the target (except for test)
        if split_name != "test" and counts[split_name] >= targets[split_name]:
            cur = min(cur + 1, 2)
            split_name = order[cur]
        for i in idxs:
            records[i]["split"] = split_name
        counts[split_name] += len(idxs)


def write_manifest_jsonl(records: Sequence[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(_json_dump(r) + "\n")


def write_manifest_csv(records: Sequence[Dict[str, Any]], out_path: str) -> None:
    """Flatten JSON into a CSV for quick inspection."""
    # minimal flat set
    fields = [
        "uid",
        "row_id",
        "split",
        "image_path",
        "image_rel_path",
        "image_container_type",
        "image_container_path",
        "image_inner_path",
        "width",
        "height",
        "sha256",
        "gt_hebrew_raw",
        "gt_hebrew_no_space",
        "gt_english_raw",
        "gt_transliteration_raw",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            img = r.get("image", {})
            cont = img.get("container") or {}
            gt = r.get("gt", {})
            heb = (gt.get("hebrew") or {})
            row = {
                "uid": r.get("uid"),
                "row_id": r.get("row_id"),
                "split": r.get("split"),
                "image_path": img.get("path"),
                "image_rel_path": img.get("rel_path"),
                "image_container_type": cont.get("type"),
                "image_container_path": cont.get("path"),
                "image_inner_path": cont.get("inner_path"),
                "width": img.get("width"),
                "height": img.get("height"),
                "sha256": img.get("sha256"),
                "gt_hebrew_raw": heb.get("raw"),
                "gt_hebrew_no_space": heb.get("no_space"),
                "gt_english_raw": (gt.get("english") or {}).get("raw"),
                "gt_transliteration_raw": (gt.get("transliteration") or {}).get("raw"),
            }
            w.writerow(row)


def read_manifest_jsonl(path: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def print_manifest_stats(records: Sequence[Dict[str, Any]]) -> None:
    total = len(records)
    by_split: Dict[str, int] = {}
    by_row: Dict[Any, int] = {}
    widths: List[int] = []
    heights: List[int] = []
    missing = 0

    for r in records:
        sp = r.get("split") or "(none)"
        by_split[sp] = by_split.get(sp, 0) + 1
        rid = r.get("row_id")
        by_row[rid] = by_row.get(rid, 0) + 1
        img = r.get("image", {})
        if img.get("exists") is False:
            missing += 1
        if isinstance(img.get("width"), int):
            widths.append(img["width"])
        if isinstance(img.get("height"), int):
            heights.append(img["height"])

    print(f"records: {total}")
    if missing:
        print(f"missing image files: {missing} ({missing / max(total,1):.1%})")
    print("splits:")
    for k in sorted(by_split.keys()):
        print(f"  {k}: {by_split[k]}")
    print(f"unique row_id groups: {len(by_row)}")
    if widths and heights:
        import statistics as st

        print(
            f"image size (w x h): mean={int(st.mean(widths))}x{int(st.mean(heights))} "
            f"min={min(widths)}x{min(heights)} max={max(widths)}x{max(heights)}"
        )


def _parse_split_ratios(vals: Sequence[str]) -> Tuple[float, float, float]:
    if len(vals) != 3:
        raise ValueError("--split-ratios expects 3 floats: train val test")
    r = tuple(float(x) for x in vals)
    s = sum(r)
    if s <= 0:
        raise ValueError("split ratios must be positive")
    return (r[0] / s, r[1] / s, r[2] / s)


def cmd_build(args: argparse.Namespace) -> None:
    mapping, mapping_info = load_mapping(args.images_root, args.mapping)
    seals_df = load_seals_csv(args.seals_csv)

    recs = build_manifest_records(
        images_root=args.images_root,
        mapping=mapping,
        mapping_info=mapping_info,
        seals_df=seals_df,
        compute_image_metadata=args.compute_image_metadata,
        compute_sha256=args.compute_sha256,
        include_csv_metadata=not args.no_csv_metadata,
    )

    # Handle missing images (mapping/corpus drift)
    missing = sum(1 for r in recs if r.get("image", {}).get("exists") is False)
    if missing and args.strict:
        raise RuntimeError(
            f"{missing} images referenced in mapping are missing from the corpus. "
            "Run without --strict or use --drop-missing to filter them out."
        )
    if args.drop_missing:
        recs = [r for r in recs if r.get("image", {}).get("exists") is not False]

    assign_splits_grouped(
        recs,
        ratios=args.split_ratios,
        seed=args.seed,
        group_field="row_id",
    )

    out = args.out
    if args.format == "jsonl":
        write_manifest_jsonl(recs, out)
    elif args.format == "csv":
        write_manifest_csv(recs, out)
    else:
        raise ValueError(f"Unknown format: {args.format}")

    print_manifest_stats(recs)
    print(f"saved: {out}")


def cmd_stats(args: argparse.Namespace) -> None:
    recs = read_manifest_jsonl(args.manifest)
    print_manifest_stats(recs)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build and inspect dataset manifest")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build a per-image manifest")
    b.add_argument("--images-root", required=True, help="Path to images dir or .zip archive")
    b.add_argument("--mapping", default=None, help="Path to mapping.json (or inner path if images-root is a zip)")
    b.add_argument("--seals-csv", default=None, help="Path to seals-fixed.csv for extra metadata")
    b.add_argument("--out", required=True, help="Output manifest path")
    b.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Output format")
    b.add_argument("--split-ratios", nargs=3, default=["0.8", "0.1", "0.1"], help="Train/val/test ratios")
    b.add_argument("--seed", type=int, default=42, help="Split random seed")
    b.add_argument("--compute-image-metadata", action="store_true", help="Compute width/height for each image")
    b.add_argument("--compute-sha256", action="store_true", help="Compute sha256 for each image")
    b.add_argument("--no-csv-metadata", action="store_true", help="Do not include extra metadata from the CSV")
    b.add_argument("--drop-missing", action="store_true", help="Drop records whose image file is missing")
    b.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any images referenced in mapping are missing from the corpus",
    )

    s = sub.add_parser("stats", help="Show quick stats for an existing manifest.jsonl")
    s.add_argument("--manifest", required=True, help="Path to manifest.jsonl")

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)
    if args.cmd == "build":
        args.split_ratios = _parse_split_ratios(args.split_ratios)
        cmd_build(args)
    elif args.cmd == "stats":
        cmd_stats(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
