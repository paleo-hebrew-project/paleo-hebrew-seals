"""train_detect
===============

Training script for *glyph detection* (character-level bboxes) on Paleo-Hebrew seals.

Baseline: YOLOv8/YOLO11 via `ultralytics`.

This module expects you already produced YOLO-format labels from the bootstrap stage.
If you used `bootstrap_labels.py export-yolo`, you have:

  dataset_root/
    images/
      <uid>.jpg
    labels/
      <uid>.txt
    classes.txt
    index.jsonl

You also need a Ultralytics dataset YAML:

  data.yaml:
    path: /abs/path/to/dataset_root
    train: images/train
    val: images/val
    test: images/test
    names: [glyph]

This script can generate that YAML and folder split from a manifest.

Outputs:
- YOLO training runs under `runs/detect/...`
- A `predict_detect.py` script can then run inference and cache results.

Notes:
- Rotated boxes: Ultralytics has oriented bounding box models in some versions.
  We keep this as an optional future extension.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set PALEO_ULTRA_HASH_PARALLEL=0 to use stock Ultralytics (slow on NFS for huge datasets).
_ULTRA_GET_HASH_PATCHED = False


def _ultralytics_get_hash_parallel(paths: list[str]) -> str:
    """Match Ultralytics ``get_hash`` output but sum file sizes with parallel ``stat`` calls.

    Stock Ultralytics loops sequentially over *every* image and label path to verify
    ``*.cache``; on network filesystems this can take hours with no log lines after
    ``Fast image access``. Same hash algorithm as ``ultralytics.data.utils.get_hash``.
    """
    if not paths:
        h = hashlib.sha256(str(0).encode())
        h.update("".encode())
        return h.hexdigest()

    def _sz(p: str) -> int:
        try:
            return os.stat(p).st_size
        except OSError:
            return 0

    workers_env = os.environ.get("PALEO_ULTRA_HASH_WORKERS", "").strip()
    n = len(paths)
    if workers_env:
        workers = max(1, int(workers_env))
    else:
        cpu = os.cpu_count() or 8
        workers = min(64, max(8, cpu * 2))
    workers = min(workers, n)

    if workers <= 1:
        size = sum(_sz(p) for p in paths)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            chunk = max(1, n // (workers * 8) or 1)
            size = sum(ex.map(_sz, paths, chunksize=chunk))

    h = hashlib.sha256(str(size).encode())
    h.update("".join(paths).encode())
    return h.hexdigest()


def _maybe_patch_ultralytics_get_hash() -> None:
    """Speed up dataset cache verification; safe to call multiple times."""
    global _ULTRA_GET_HASH_PATCHED
    if _ULTRA_GET_HASH_PATCHED:
        return
    if os.environ.get("PALEO_ULTRA_HASH_PARALLEL", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return
    try:
        import ultralytics.data.dataset as ultra_ds  # type: ignore
        import ultralytics.data.utils as ultra_u  # type: ignore
    except Exception:
        return
    ultra_u.get_hash = _ultralytics_get_hash_parallel  # type: ignore[assignment]
    ultra_ds.get_hash = _ultralytics_get_hash_parallel  # type: ignore[assignment]
    _ULTRA_GET_HASH_PATCHED = True
    print(
        "[yolo-cache] Using parallel Ultralytics dataset hash (PALEO_ULTRA_HASH_WORKERS, "
        "or PALEO_ULTRA_HASH_PARALLEL=0 to disable)"
    )


def resolve_ultralytics_run_dir(project: str, name: str) -> Optional[Path]:
    """
    Locate an Ultralytics training run directory containing weights/best.pt.

    Ultralytics may save under ``project/name`` or, if ``cwd`` is already inside
    ``.../runs/detect``, under ``project/runs/detect/name`` (nested layout). It
    may also suffix the run folder (``name``, ``name2``, ``name3``) when the name
    already exists — we pick the newest ``best.pt`` among name and name+digits.
    """
    proj = Path(project).resolve()
    candidates: List[Path] = [proj, proj / "runs" / "detect"]

    def _matches_base(folder: str, base: str) -> bool:
        if folder == base:
            return True
        if folder.startswith(base):
            suf = folder[len(base) :]
            return suf.isdigit()
        return False

    best_path: Optional[Path] = None
    best_mtime = -1.0
    for root in candidates:
        if not root.is_dir():
            continue
        try:
            for p in root.iterdir():
                if not p.is_dir():
                    continue
                if not _matches_base(p.name, name):
                    continue
                w = p / "weights" / "best.pt"
                if not w.is_file():
                    continue
                try:
                    mt = w.stat().st_mtime
                except OSError:
                    continue
                if mt > best_mtime:
                    best_mtime = mt
                    best_path = p
        except OSError:
            continue
    return best_path.resolve() if best_path is not None else None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _symlink_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlinks:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def build_yolo_dataset_from_manifest(
    manifest_jsonl: Path,
    out_root: Path,
    image_key: str = "image",
    split_key: str = "split",
    bboxes_key: str = "gt.bboxes",
    use_symlinks: bool = True,
    class_name: str = "glyph",
) -> Path:
    """Create a YOLO dataset layout from a manifest that already has bboxes.

    Requires `record['gt']['bboxes']` to be a list of xyxy bboxes in pixel coordinates.

    Writes:
      out_root/
        images/{train,val,test}/<uid>.<ext>
        labels/{train,val,test}/<uid>.txt
        classes.txt
        data.yaml

    Returns:
      Path to the generated data.yaml
    """

    records = _read_jsonl(manifest_jsonl)
    out_root.mkdir(parents=True, exist_ok=True)

    # classes
    _write_text(out_root / "classes.txt", class_name + "\n")

    # helper to access nested key like "gt.bboxes"
    def get_nested(d: Dict[str, Any], dotted: str):
        cur: Any = d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    n_written = 0
    for r in records:
        uid = r.get("uid") or f"row{r.get('row_id','unknown')}"
        split = r.get(split_key, "train")
        img = r.get(image_key, {})
        rel = img.get("rel_path")
        abs_path = img.get("abs_path")
        if abs_path:
            src = Path(abs_path)
        else:
            # try joining to images_root if present
            images_root = r.get("images_root")
            src = Path(images_root) / rel if images_root and rel else None
        if src is None or not src.exists():
            continue

        bboxes = get_nested(r, bboxes_key)
        if not bboxes:
            continue

        # image dims for normalization
        w = img.get("width")
        h = img.get("height")
        if not (w and h):
            # if metadata missing, skip; you can regenerate metadata in dataset_manifest
            continue

        ext = src.suffix.lower()
        dst_img = out_root / "images" / split / f"{uid}{ext}"
        _symlink_or_copy(src, dst_img, use_symlinks)

        # YOLO label format: class cx cy bw bh (normalized 0..1)
        lines = []
        for bb in bboxes:
            x1, y1, x2, y2 = map(float, bb)
            cx = ((x1 + x2) * 0.5) / float(w)
            cy = ((y1 + y2) * 0.5) / float(h)
            bw = (x2 - x1) / float(w)
            bh = (y2 - y1) / float(h)
            # clamp
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        dst_lbl = out_root / "labels" / split / f"{uid}.txt"
        _write_text(dst_lbl, "\n".join(lines) + "\n")
        n_written += 1

    # Ultralytics data.yaml
    data_yaml = f"""# Auto-generated
path: {out_root.as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: {class_name}
"""
    data_yaml_path = out_root / "data.yaml"
    _write_text(data_yaml_path, data_yaml)

    print(f"[build] wrote {n_written} labeled images to {out_root}")
    return data_yaml_path


def _normalize_ultralytics_model_arg(model: str) -> str:
    """Use absolute paths for local checkpoint files so torch.load sees a stable path."""
    p = Path(model).expanduser()
    if p.is_file() and p.suffix.lower() in (".pt", ".pth", ".onnx", ".yaml", ".yml"):
        return str(p.resolve())
    return model


def _is_rtdetr_model(model: str) -> bool:
    name = Path(model).name.lower()
    return name.startswith("rtdetr")


def train_ultralytics(
    data_yaml: Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
    workers: int,
    lr0: Optional[float] = None,
    train_overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    """Train an Ultralytics detector. Returns the resolved run directory (contains weights/)."""
    try:
        from ultralytics import RTDETR, YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Ultralytics not installed. Install with: pip install ultralytics"
        ) from e

    _maybe_patch_ultralytics_get_hash()

    model = _normalize_ultralytics_model_arg(model)
    project_resolved = str(Path(project).resolve())

    if _is_rtdetr_model(model):
        yolo = RTDETR(model)
    else:
        yolo = YOLO(model)

    kwargs: Dict[str, Any] = dict(
        data=str(Path(data_yaml).resolve()),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
        project=project_resolved,
        name=name,
        workers=int(workers),
    )
    if lr0 is not None:
        kwargs["lr0"] = float(lr0)
    if train_overrides:
        # Keep explicit function args authoritative over overrides.
        for k, v in dict(train_overrides).items():
            if k in kwargs and k not in {"lr0"}:
                continue
            kwargs[k] = v

    yolo.train(**kwargs)
    trainer = getattr(yolo, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None) if trainer is not None else None
    if save_dir:
        return Path(str(save_dir)).resolve()
    found = resolve_ultralytics_run_dir(project_resolved, name)
    if found is not None:
        return found
    raise RuntimeError(
        f"Ultralytics finished but run directory not found (project={project_resolved!r}, name={name!r}). "
        f"Train from repo root or pass an absolute detector.project in YAML."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train glyph detector (YOLO).")

    p.add_argument("--manifest", type=str, default=None, help="manifest_with_bboxes.jsonl")
    p.add_argument("--yolo-ds", type=str, default=None, help="Existing YOLO dataset root (contains data.yaml)")
    p.add_argument("--out-root", type=str, default="yolo_glyph_ds", help="Where to build YOLO dataset from manifest")
    p.add_argument("--use-symlinks", action="store_true", help="Use symlinks instead of copying images")

    p.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model name or path")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", type=str, default="0", help="CUDA device string, e.g. '0' or '0,1'")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="paleo_glyph")
    p.add_argument("--lr0", type=float, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml: Optional[Path] = None

    if args.yolo_ds:
        ds_root = Path(args.yolo_ds)
        data_yaml = ds_root / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found under {ds_root}")
    else:
        if not args.manifest:
            raise ValueError("Provide --yolo-ds or --manifest")
        data_yaml = build_yolo_dataset_from_manifest(
            manifest_jsonl=Path(args.manifest),
            out_root=Path(args.out_root),
            use_symlinks=bool(args.use_symlinks),
        )

    train_ultralytics(
        data_yaml=data_yaml,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        lr0=args.lr0,
    )


if __name__ == "__main__":
    main()
