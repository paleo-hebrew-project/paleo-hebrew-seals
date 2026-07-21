"""Aggregate Ultralytics detector training runs (results.csv + args.yaml) into one table.

Sanity-checks that validation data came from the same manifest(s) via yolo_dataset/export_meta.json
when present; otherwise falls back to data.yaml path + val image directory fingerprint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a comparison table from Ultralytics runs under a folder (e.g. runs/detect/.../runs/detect)"
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Directory containing one subfolder per run (each with args.yaml and usually results.csv)",
    )
    p.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Project root (contains paleo_ocr/). Auto-detected if omitted.",
    )
    p.add_argument("--out-csv", type=str, default="", help="Write aggregated table to this CSV path")
    p.add_argument("--quiet", action="store_true", help="Suppress stdout table (still write CSV if --out-csv)")
    return p.parse_args()


def _find_repo_root(start: Path) -> Path:
    for p in [start.resolve()] + list(start.resolve().parents):
        if (p / "paleo_ocr").is_dir() and (p / "paleo_ocr" / "__init__.py").exists():
            return p
    return start.resolve()


def _safe_yaml_load(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _resolve_data_yaml(data_field: str, save_dir: Path, repo_root: Path) -> Path:
    """Resolve Ultralytics args `data` path to an existing data.yaml."""
    p = Path(data_field)
    if p.is_absolute():
        return p.resolve()
    candidates = [
        repo_root / data_field,
        repo_root / "scripts" / data_field,
        save_dir / data_field,
        Path.cwd() / data_field,
    ]
    for c in candidates:
        try:
            cr = c.resolve()
            if cr.is_file():
                return cr
        except (OSError, RuntimeError):
            continue
    return (repo_root / "scripts" / data_field).resolve()


def _export_meta_path(data_yaml: Path) -> Path:
    return data_yaml.parent / "export_meta.json"


def _val_fingerprint_from_disk(data_yaml: Path) -> str:
    """When export_meta is missing: hash sorted rel paths under images/val."""
    import hashlib

    root = data_yaml.parent
    val_img = root / "images" / "val"
    if not val_img.is_dir():
        return ""
    rels: List[str] = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for f in sorted(val_img.rglob(pat)):
            try:
                rels.append(str(f.relative_to(val_img)))
            except ValueError:
                rels.append(f.name)
    h = hashlib.sha256("\n".join(rels).encode("utf-8")).hexdigest()[:16]
    return f"dir_sha256:{h}:n={len(rels)}"


def _read_results_last_row(results_csv: Path) -> Dict[str, str]:
    if not results_csv.is_file():
        return {}
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k: str(v) for k, v in rows[-1].items()}


def _collect_run(
    run_dir: Path,
    repo_root: Path,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (row dict, warnings for this run).
    """
    warnings: List[str] = []
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        return {}, [f"{run_dir.name}: missing args.yaml"]

    args = _safe_yaml_load(args_path)
    save_dir = Path(args.get("save_dir") or run_dir)
    data_field = str(args.get("data") or "")
    data_yaml = _resolve_data_yaml(data_field, save_dir, repo_root) if data_field else Path()
    if data_field and not data_yaml.is_file():
        warnings.append(f"{run_dir.name}: data.yaml not found (tried {data_yaml})")

    val_key = ""
    val_manifests: Tuple[str, ...] = ()
    train_manifests: Tuple[str, ...] = ()
    train_split_filter = ""
    val_split_filter = ""
    em_path = _export_meta_path(data_yaml) if data_yaml.is_file() else Path()
    if em_path.is_file():
        try:
            em = json.loads(em_path.read_text(encoding="utf-8"))
            vm = em.get("val_manifests") or []
            if isinstance(vm, list) and vm:
                val_manifests = tuple(sorted(str(Path(x).resolve()) for x in vm))
                val_key = "manifests:" + "|".join(val_manifests)
            tm = em.get("train_manifests") or []
            if isinstance(tm, list) and tm:
                train_manifests = tuple(sorted(str(Path(x).resolve()) for x in tm))
            train_split_filter = str(em.get("train_split_filter") or "")
            val_split_filter = str(em.get("val_split_filter") or "")
            if val_key and val_split_filter:
                val_key = f"{val_key}::val_split={val_split_filter}"
        except Exception as e:
            warnings.append(f"{run_dir.name}: export_meta.json read failed: {e}")
    if not val_key and data_yaml.is_file():
        val_key = _val_fingerprint_from_disk(data_yaml)
        if not val_key:
            warnings.append(f"{run_dir.name}: could not fingerprint val set (no export_meta, empty val/)")

    results_csv = run_dir / "results.csv"
    last = _read_results_last_row(results_csv)

    row: Dict[str, Any] = {
        "run_dir": str(run_dir.name),
        "model": str(args.get("model") or ""),
        "data_yaml": str(data_yaml) if data_yaml.is_file() else str(data_field),
        "val_manifests_key": val_key,
        "val_manifests": ";".join(val_manifests) if val_manifests else "",
        "train_manifests": ";".join(train_manifests) if train_manifests else "",
        "train_split_filter": train_split_filter,
        "val_split_filter": val_split_filter,
        "epochs_trained": str(args.get("epochs") or ""),
        "imgsz": str(args.get("imgsz") or ""),
        "batch": str(args.get("batch") or ""),
        "split": str(args.get("split") or ""),
    }
    # Last epoch metrics from results.csv (Ultralytics column names)
    for col in (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "epoch",
        "time",
    ):
        if col in last:
            row[col.replace("/", "_").replace("-", "_")] = last[col]

    if not last:
        warnings.append(f"{run_dir.name}: missing or empty results.csv")

    return row, warnings


def _sanity_val_keys(rows: Sequence[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Return (all_same, messages)."""
    keys = [r.get("val_manifests_key") or r.get("val_manifests") for r in rows]
    keys = [k for k in keys if k]
    if not keys:
        return False, ["No val_manifests_key / fingerprint could be computed for any run."]
    unique = sorted(set(keys))
    if len(unique) == 1:
        msgs = [
            f"OK: all {len(rows)} runs share the same val definition:\n  {unique[0][:500]}{'...' if len(unique[0]) > 500 else ''}"
        ]
        train_vals = [r.get("train_manifests") or "" for r in rows]
        ut = sorted(set(train_vals))
        if len(ut) > 1:
            msgs.append(
                f"\nNote: {len(ut)} distinct train_manifests sets (different synth/real regimes are OK); "
                "val_manifests above is the eval set used for metrics."
            )
        return True, msgs
    msgs = [
        f"WARNING: {len(unique)} distinct val definitions — comparisons are not apples-to-apples.",
        "Distinct keys:",
    ]
    for u in unique:
        who = [
            r["run_dir"]
            for r in rows
            if (r.get("val_manifests_key") or r.get("val_manifests")) == u
        ]
        msgs.append(f"  - ({len(who)} runs) {u[:300]}{'...' if len(u) > 300 else ''}")
        msgs.append(f"    runs: {', '.join(who[:12])}{' ...' if len(who) > 12 else ''}")
    return False, msgs


def main() -> None:
    args = parse_args()
    root = Path(args.root or "").expanduser()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        raise SystemExit(1)

    repo_root = Path(args.repo_root).expanduser() if args.repo_root else _find_repo_root(root)

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    all_rows: List[Dict[str, Any]] = []
    all_warnings: List[str] = []

    for d in subdirs:
        row, w = _collect_run(d, repo_root)
        if row:
            all_rows.append(row)
        all_warnings.extend(w)

    if not all_rows:
        print("No runs with args.yaml found.", file=sys.stderr)
        raise SystemExit(2)

    ok, sanity_msgs = _sanity_val_keys(all_rows)

    if not args.quiet:
        print("=== Val / test data sanity ===\n", file=sys.stderr)
        for m in sanity_msgs:
            print(m, file=sys.stderr)
        print(file=sys.stderr)
        if all_warnings:
            print("=== Per-run warnings ===\n", file=sys.stderr)
            for w in all_warnings:
                print(w, file=sys.stderr)
            print(file=sys.stderr)

    fieldnames: List[str] = []
    for r in all_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    if not args.quiet:
        w = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    out = (args.out_csv or "").strip()
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            wr.writeheader()
            for r in all_rows:
                wr.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"Wrote {len(all_rows)} rows to {out_path}", file=sys.stderr)

    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
