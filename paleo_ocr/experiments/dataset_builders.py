"""Manifest → YOLO layout and classifier ImageFolder crops (notebook-compatible schema)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from paleo_ocr.experiments.constants import FINAL_TO_BASE, HEB_SET, LETTER_TO_CLASS, class_names_heb22
from paleo_ocr.train_detect import _symlink_or_copy, _write_text


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def get_any_img_path(rec: dict, manifest_path: Path) -> Path:
    """Resolve image path like Paleo_OCR.ipynb (abs_path/path, then rel to manifest parent)."""
    img = rec.get("image", {}) or {}
    for k in ("abs_path", "path"):
        p = img.get(k)
        if p and Path(p).exists():
            return Path(p)
    rp = img.get("rel_path")
    if rp:
        for c in (Path(rp), manifest_path.parent / rp):
            if c.exists():
                return c
    raise FileNotFoundError(f"Image not found for uid={rec.get('uid')} img_keys={list(img.keys())}")


def normalize_hebrew_char(ch: str) -> Optional[str]:
    """Return normalized Hebrew letter in the 22-letter set, or None to skip."""
    if not ch:
        return None
    ch = ch.strip()
    if not ch:
        return None
    ch = FINAL_TO_BASE.get(ch, ch)
    if ch in HEB_SET:
        return ch
    return None


def safe_stem(s: str) -> str:
    s = str(s).replace(os.sep, "_")
    s = re.sub(r"[^0-9A-Za-z_\-\.]+", "_", s)
    return s[:180] if len(s) > 180 else s


def clamp_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def bg_from_corners(img: Any) -> Tuple[int, int, int]:
    from PIL import Image

    w, h = img.size
    pts = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    px = [img.getpixel(p) for p in pts]
    r = sum(p[0] for p in px) // 4
    g = sum(p[1] for p in px) // 4
    b = sum(p[2] for p in px) // 4
    return (r, g, b)


def crop_letter(
    img: Any,
    bbox_xyxy: Sequence[float],
    *,
    pad_pct: float = 0.12,
    make_square: bool = True,
    out_size: Optional[int] = None,
) -> Any:
    """Crop glyph; optional square pad and resize (matches notebook defaults)."""
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = int(round(max(bw, bh) * pad_pct))
    x1, y1, x2, y2 = clamp_xyxy(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)
    crop = img.crop((x1, y1, x2, y2)).convert("RGB")

    if make_square:
        cw, ch = crop.size
        s = max(cw, ch)
        bg = bg_from_corners(crop)
        canvas = Image.new("RGB", (s, s), bg)
        canvas.paste(crop, ((s - cw) // 2, (s - ch) // 2))
        crop = canvas

    if out_size is not None and out_size > 0:
        crop = crop.resize((out_size, out_size), Image.BICUBIC)
    return crop


def _to_yolo_line_singlecls(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> str:
    cx = ((x1 + x2) * 0.5) / float(w)
    cy = ((y1 + y2) * 0.5) / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _to_yolo_line_multiclass(
    cls_id: int, x1: float, y1: float, x2: float, y2: float, w: float, h: float
) -> str:
    cx = ((x1 + x2) * 0.5) / float(w)
    cy = ((y1 + y2) * 0.5) / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _uid_key(uid: str, manifest_stem: str) -> str:
    h = hashlib.md5(manifest_stem.encode("utf-8")).hexdigest()[:8]
    return f"{h}_{safe_stem(uid)}"


@dataclass
class YoloExportStats:
    written: int = 0
    skipped: int = 0


def _manifest_cache_signature(manifests: Sequence[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mp in manifests:
        rp = Path(mp).resolve()
        st = rp.stat()
        out.append(
            {
                "path": str(rp),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
            }
        )
    return out


def _dataset_export_complete(out_root: Path) -> bool:
    return (out_root / "data.yaml").is_file() and (out_root / "export_meta.json").is_file()


def _ultralytics_label_caches_ready(out_root: Path) -> bool:
    """Ultralytics YOLO writes labels/train.cache and labels/val.cache."""
    return (out_root / "labels" / "train.cache").is_file() and (out_root / "labels" / "val.cache").is_file()


def _yolo_dataset_ready(out_root: Path) -> bool:
    """Export done and Ultralytics label caches present (avoids parallel trainers racing on *.cache)."""
    return _dataset_export_complete(out_root) and _ultralytics_label_caches_ready(out_root)


def _prime_ultralytics_label_caches(data_yaml: Path) -> None:
    """Serially build labels/train.cache and labels/val.cache the same way training would, once."""
    try:
        import yaml
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.utils import DEFAULT_CFG
    except ImportError:
        return

    data_yaml = Path(data_yaml).resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    base = Path(raw["path"]).resolve()
    train_rel = raw["train"]
    val_rel = raw["val"]
    train_path = str(Path(train_rel) if Path(train_rel).is_absolute() else base / train_rel)
    val_path = str(Path(val_rel) if Path(val_rel).is_absolute() else base / val_rel)
    names = raw.get("names") or {0: "glyph"}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    data_dict = {"names": names, "channels": int(raw.get("channels") or 3)}
    hyp = DEFAULT_CFG
    n_train_str = ""
    try:
        train_dir = Path(train_path)
        if train_dir.is_dir():
            n_train_str = f" ({sum(1 for _ in train_dir.iterdir())} images)"
    except Exception:
        pass
    print(f"[yolo-cache] Ultralytics cache priming: train -> {train_path}{n_train_str}", flush=True)
    _t0 = time.time()
    YOLODataset(
        img_path=train_path,
        imgsz=640,
        cache=False,
        augment=False,
        hyp=hyp,
        prefix="",
        rect=False,
        batch_size=1,
        stride=32,
        single_cls=True,
        data=data_dict,
        task="detect",
    )
    print(f"[yolo-cache] train cache primed in {time.time() - _t0:.1f}s", flush=True)
    print(f"[yolo-cache] Ultralytics cache priming: val -> {val_path}", flush=True)
    _t0 = time.time()
    YOLODataset(
        img_path=val_path,
        imgsz=640,
        cache=False,
        augment=False,
        hyp=hyp,
        prefix="",
        rect=True,
        batch_size=1,
        stride=32,
        single_cls=True,
        data=data_dict,
        task="detect",
    )
    print(f"[yolo-cache] val cache primed in {time.time() - _t0:.1f}s", flush=True)


_LOCK_STALE_GRACE_SECONDS = 120.0
_LOCK_HEARTBEAT_INTERVAL = 5.0


def _pid_alive(pid: Optional[int]) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_lock_pid(lock_dir: Path) -> Optional[int]:
    try:
        return int((lock_dir / "pid").read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _lock_heartbeat_age(lock_dir: Path) -> Optional[float]:
    try:
        return time.time() - (lock_dir / "heartbeat").stat().st_mtime
    except (FileNotFoundError, OSError):
        return None


def _lock_is_stale(lock_dir: Path, grace: float = _LOCK_STALE_GRACE_SECONDS) -> bool:
    """A shared build lock is safe to steal when its heartbeat is older than ``grace``.

    A live builder refreshes ``lock_dir/heartbeat`` periodically. A freshly
    created lock that has not written a heartbeat yet is protected while it is
    younger than ``grace`` (initializing window). A lock whose owner died or
    hung on I/O stops heartbeating and becomes stealable after ``grace``.
    """
    hb_age = _lock_heartbeat_age(lock_dir)
    if hb_age is not None:
        return hb_age > grace
    try:
        lock_age = time.time() - lock_dir.stat().st_mtime
    except (FileNotFoundError, OSError):
        return True
    return lock_age > grace


def _start_lock_heartbeat(lock_dir: Path, interval: float = _LOCK_HEARTBEAT_INTERVAL) -> "threading.Thread":
    """Refresh ``lock_dir/heartbeat`` mtime from a daemon thread until the process exits."""
    import threading

    hb = lock_dir / "heartbeat"
    try:
        hb.write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass

    def _beat(stop: threading.Event) -> None:
        while not stop.wait(interval):
            try:
                os.utime(str(hb), None)
            except OSError:
                pass

    stop = threading.Event()
    t = threading.Thread(target=_beat, args=(stop,), daemon=True)
    t.start()
    t.stop_event = stop  # type: ignore[attr-defined]
    return t


def _yolo_dataset_cache_key(
    train_manifests: Sequence[Path],
    val_manifests: Sequence[Path],
    *,
    class_name: str,
    single_class: bool,
    cls_id_from_char: Optional[Callable[[str], int]],
    use_symlinks: bool,
    dedupe_uid_prefix: bool,
    train_split_filter: Optional[str],
    val_split_filter: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    cls_fn = None
    if cls_id_from_char is not None:
        cls_fn = f"{getattr(cls_id_from_char, '__module__', '')}.{getattr(cls_id_from_char, '__qualname__', getattr(cls_id_from_char, '__name__', 'callable'))}"
    payload: Dict[str, Any] = {
        "cache_schema_version": 1,
        "train_manifests": _manifest_cache_signature(train_manifests),
        "val_manifests": _manifest_cache_signature(val_manifests),
        "class_name": class_name,
        "single_class": bool(single_class),
        "cls_id_from_char": cls_fn,
        "use_symlinks": bool(use_symlinks),
        "dedupe_uid_prefix": bool(dedupe_uid_prefix),
        "train_split_filter": train_split_filter,
        "val_split_filter": val_split_filter,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20], payload


def build_yolo_dataset_from_manifests(
    train_manifests: Sequence[Path],
    val_manifests: Sequence[Path],
    out_root: Path,
    *,
    class_name: str = "glyph",
    single_class: bool = True,
    cls_id_from_char: Optional[Callable[[str], int]] = None,
    use_symlinks: bool = True,
    dedupe_uid_prefix: bool = True,
    train_split_filter: Optional[str] = None,
    val_split_filter: Optional[str] = None,
) -> Path:
    """
    Export YOLO layout: train from one or more manifests, val from one or more.
    Records use gt.bboxes (xyxy); multiclass uses gt.chars aligned with bboxes when not single_class.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    for sub in ("images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    _write_text(out_root / "classes.txt", class_name + "\n")

    stats = YoloExportStats()

    def handle_record(r: Dict[str, Any], manifest_path: Path, split: str) -> None:
        nonlocal stats
        uid = str(r.get("uid") or f"row{r.get('row_id', 'unknown')}")
        if dedupe_uid_prefix:
            uid = _uid_key(uid, manifest_path.stem)

        img_meta = r.get("image", {}) or {}
        try:
            src = get_any_img_path(r, manifest_path)
        except FileNotFoundError:
            stats.skipped += 1
            return

        w = img_meta.get("width")
        h = img_meta.get("height")
        if not w or not h:
            try:
                from PIL import Image

                with Image.open(src) as im:
                    w, h = im.size
            except Exception:
                stats.skipped += 1
                return

        gt = r.get("gt", {}) or {}
        bboxes = gt.get("bboxes") or []
        chars = gt.get("chars")

        if not bboxes:
            stats.skipped += 1
            return

        ext = src.suffix.lower() or ".jpg"
        dst_img = out_root / "images" / split / f"{uid}{ext}"
        _symlink_or_copy(src, dst_img, use_symlinks)

        lines: List[str] = []
        for i, bb in enumerate(bboxes):
            try:
                x1, y1, x2, y2 = map(float, bb)
            except Exception:
                continue
            if single_class:
                lines.append(_to_yolo_line_singlecls(x1, y1, x2, y2, float(w), float(h)))
            else:
                ch = None
                if isinstance(chars, list) and i < len(chars):
                    ch = normalize_hebrew_char(str(chars[i]))
                if ch is None or cls_id_from_char is None:
                    stats.skipped += 1
                    continue
                cid = cls_id_from_char(ch)
                lines.append(_to_yolo_line_multiclass(cid, x1, y1, x2, y2, float(w), float(h)))

        if not lines:
            stats.skipped += 1
            return

        dst_lbl = out_root / "labels" / split / f"{uid}.txt"
        _write_text(dst_lbl, "\n".join(lines) + "\n")
        stats.written += 1

    for mp in train_manifests:
        mp = Path(mp).resolve()
        records = _read_jsonl(mp)
        n = len(records)
        for i, r in enumerate(records, start=1):
            if train_split_filter and str(r.get("split", "")).lower() != str(train_split_filter).lower():
                continue
            handle_record(r, mp, "train")
            if n >= 10000 and (i % 10000 == 0 or i == n):
                print(
                    f"[yolo-cache] export train ({mp.name}): {i}/{n} "
                    f"({stats.written} written, {stats.skipped} skipped)",
                    flush=True,
                )

    for mp in val_manifests:
        mp = Path(mp).resolve()
        records = _read_jsonl(mp)
        n = len(records)
        for i, r in enumerate(records, start=1):
            if val_split_filter and str(r.get("split", "")).lower() != str(val_split_filter).lower():
                continue
            handle_record(r, mp, "val")
        print(
            f"[yolo-cache] export val ({mp.name}): {n} rows -> {stats.written} total written",
            flush=True,
        )

    names_yaml = f"  0: {class_name}\n" if single_class else "\n".join(
        f"  {i}: {class_name}" for i in range(22)
    )
    data_yaml = f"""# Auto-generated by paleo_ocr.experiments.dataset_builders
path: {out_root.as_posix()}
train: images/train
val: images/val
test: images/test
names:
{names_yaml}
"""
    data_yaml_path = out_root / "data.yaml"
    _write_text(data_yaml_path, data_yaml)
    meta = {
        "written": stats.written,
        "skipped": stats.skipped,
        "train_manifests": [str(p) for p in train_manifests],
        "val_manifests": [str(p) for p in val_manifests],
        "train_split_filter": train_split_filter,
        "val_split_filter": val_split_filter,
    }
    _write_text(out_root / "export_meta.json", json.dumps(meta, indent=2, ensure_ascii=False))
    return data_yaml_path


def build_or_reuse_yolo_dataset_from_manifests(
    train_manifests: Sequence[Path],
    val_manifests: Sequence[Path],
    out_root: Path,
    *,
    class_name: str = "glyph",
    single_class: bool = True,
    cls_id_from_char: Optional[Callable[[str], int]] = None,
    use_symlinks: bool = True,
    dedupe_uid_prefix: bool = True,
    train_split_filter: Optional[str] = None,
    val_split_filter: Optional[str] = None,
    shared_cache_root: Optional[Path] = None,
    lock_poll_seconds: float = 5.0,
) -> Path:
    if shared_cache_root is None:
        return build_yolo_dataset_from_manifests(
            train_manifests,
            val_manifests,
            out_root,
            class_name=class_name,
            single_class=single_class,
            cls_id_from_char=cls_id_from_char,
            use_symlinks=use_symlinks,
            dedupe_uid_prefix=dedupe_uid_prefix,
            train_split_filter=train_split_filter,
            val_split_filter=val_split_filter,
        )

    shared_cache_root = Path(shared_cache_root)
    shared_cache_root.mkdir(parents=True, exist_ok=True)
    cache_key, cache_payload = _yolo_dataset_cache_key(
        train_manifests,
        val_manifests,
        class_name=class_name,
        single_class=single_class,
        cls_id_from_char=cls_id_from_char,
        use_symlinks=use_symlinks,
        dedupe_uid_prefix=dedupe_uid_prefix,
        train_split_filter=train_split_filter,
        val_split_filter=val_split_filter,
    )
    cache_root = shared_cache_root / f"dataset_{cache_key}"
    lock_dir = shared_cache_root / f".dataset_{cache_key}.lock"

    if _yolo_dataset_ready(cache_root):
        print(f"[yolo-cache] reuse {cache_root}", flush=True)
        return cache_root / "data.yaml"

    announced_wait = False
    wait_start = time.time()
    last_announce = 0.0
    while True:
        if _yolo_dataset_ready(cache_root):
            print(f"[yolo-cache] reuse {cache_root}", flush=True)
            return cache_root / "data.yaml"
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if _lock_is_stale(lock_dir):
                # Orphaned lock: previous builder died or hung on I/O. Steal it.
                pid = _read_lock_pid(lock_dir)
                why = f"pid {pid} not alive" if not _pid_alive(pid) else "heartbeat stale"
                print(f"[yolo-cache] reclaiming stale shared lock {lock_dir} ({why})", flush=True)
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            now = time.time()
            if not announced_wait:
                print(f"[yolo-cache] wait for shared build {cache_root}", flush=True)
                announced_wait = True
                last_announce = now
            elif now - last_announce >= 60.0:
                elapsed = int(now - wait_start)
                hb_age = _lock_heartbeat_age(lock_dir)
                hb_str = f"; builder heartbeat {int(hb_age)}s ago" if hb_age is not None else ""
                print(
                    f"[yolo-cache] still waiting for shared build {cache_root} "
                    f"({elapsed}s elapsed{hb_str})",
                    flush=True,
                )
                last_announce = now
            time.sleep(max(0.2, float(lock_poll_seconds)))

    # We won the lock: record owner + start a heartbeat so waiters can detect hangs.
    try:
        (lock_dir / "pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    except OSError:
        pass
    heartbeat_thread = _start_lock_heartbeat(lock_dir)

    build_ok = False
    try:
        if _yolo_dataset_ready(cache_root):
            print(f"[yolo-cache] reuse {cache_root}", flush=True)
            return cache_root / "data.yaml"

        # Partial reuse: manifest export finished but a previous race left Ultralytics *.cache missing.
        if cache_root.exists() and _dataset_export_complete(cache_root) and not _ultralytics_label_caches_ready(
            cache_root
        ):
            dy = cache_root / "data.yaml"
            print(f"[yolo-cache] prime missing Ultralytics label caches under {cache_root}", flush=True)
            _prime_ultralytics_label_caches(dy)
            if not (cache_root / "shared_cache_meta.json").is_file():
                cache_meta = dict(cache_payload)
                cache_meta.update(
                    {
                        "dataset_root": str(cache_root),
                        "data_yaml": str(dy),
                        "built_at_unix": time.time(),
                        "builder_pid": os.getpid(),
                        "primed_ultralytics_caches_only": True,
                    }
                )
                _write_text(
                    cache_root / "shared_cache_meta.json",
                    json.dumps(cache_meta, indent=2, ensure_ascii=False),
                )
            build_ok = True
            print(f"[yolo-cache] ready {cache_root}", flush=True)
            return dy

        if cache_root.exists():
            shutil.rmtree(cache_root, ignore_errors=True)
        print(f"[yolo-cache] build {cache_root}", flush=True)
        _build_t0 = time.time()
        data_yaml = build_yolo_dataset_from_manifests(
            train_manifests,
            val_manifests,
            cache_root,
            class_name=class_name,
            single_class=single_class,
            cls_id_from_char=cls_id_from_char,
            use_symlinks=use_symlinks,
            dedupe_uid_prefix=dedupe_uid_prefix,
            train_split_filter=train_split_filter,
            val_split_filter=val_split_filter,
        )
        print(f"[yolo-cache] export done in {time.time() - _build_t0:.1f}s", flush=True)
        print(f"[yolo-cache] priming Ultralytics label caches (holds lock until train+val .cache exist)", flush=True)
        _prime_ultralytics_label_caches(data_yaml)
        cache_meta = dict(cache_payload)
        cache_meta.update(
            {
                "dataset_root": str(cache_root),
                "data_yaml": str(data_yaml),
                "built_at_unix": time.time(),
                "builder_pid": os.getpid(),
            }
        )
        _write_text(cache_root / "shared_cache_meta.json", json.dumps(cache_meta, indent=2, ensure_ascii=False))
        build_ok = True
        print(f"[yolo-cache] ready {cache_root}", flush=True)
        return data_yaml
    finally:
        try:
            heartbeat_thread.stop_event.set()
        except AttributeError:
            pass
        if not build_ok and cache_root.exists() and not _yolo_dataset_ready(cache_root):
            shutil.rmtree(cache_root, ignore_errors=True)
        shutil.rmtree(lock_dir, ignore_errors=True)


# Weighted YOLO train: pass same manifest multiple times in train_manifests
def effective_train_manifests_weighted(
    manifest_weights: Sequence[Tuple[Path, int]],
) -> List[Path]:
    """Repeat each manifest path `count` times so random YOLO shuffle approximates weights."""
    out: List[Path] = []
    for mp, count in manifest_weights:
        for _ in range(max(1, int(count))):
            out.append(Path(mp))
    return out


ClassifierManifestExportSpec = Union[Tuple[Path, str], Tuple[Path, str, Optional[str]]]


def export_classifier_imagefolder(
    manifest_split_list: Sequence[ClassifierManifestExportSpec],
    dst_root: Path,
    *,
    pad_pct: float = 0.12,
    make_square: bool = True,
    out_size: Optional[int] = None,
    max_workers: int = 8,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    manifest_split_list:
      - (manifest_path, split) where split is 'train' or 'val'
      - or (manifest_path, split, split_filter) to keep only rows with rec["split"] == split_filter
    Writes dst_root/{train,val}/<class_name>/*.png and classes.txt.
    """
    from PIL import Image

    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    class_names = class_names_heb22()
    for sp in ("train", "val"):
        for cls in class_names:
            (dst_root / sp / cls).mkdir(parents=True, exist_ok=True)

    (dst_root / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")
    (dst_root / "class_to_idx.json").write_text(
        json.dumps({c: i for i, c in enumerate(class_names)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta_rows: List[List[str]] = []
    lock_rows: List[str] = [
        "split",
        "class_folder",
        "letter",
        "uid",
        "idx",
        "src_image",
        "bbox_json",
        "out_path",
    ]

    def process_record(
        rec: dict,
        manifest_path: Path,
        forced_split: str,
        split_filter: Optional[str],
    ) -> List[List[str]]:
        if split_filter and str(rec.get("split", "")).lower() != str(split_filter).lower():
            return []
        uid = str(rec.get("uid") or "")
        gt = rec.get("gt", {}) or {}
        chars = gt.get("chars") or []
        bboxes = gt.get("bboxes") or []
        if not isinstance(chars, list) or not isinstance(bboxes, list) or len(chars) != len(bboxes) or not chars:
            return []
        try:
            img_path = get_any_img_path(rec, manifest_path)
        except FileNotFoundError:
            return [["__MISSING_IMAGE__"]]
        img = Image.open(img_path).convert("RGB")
        out_rows: List[List[str]] = []
        for i, (ch, bb) in enumerate(zip(chars, bboxes)):
            letter = normalize_hebrew_char(str(ch))
            if letter is None:
                continue
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, bb)
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            split = forced_split
            class_name = LETTER_TO_CLASS[letter]
            out_dir = dst_root / split / class_name
            out_name = f"{safe_stem(uid)}__{i:03d}__{ord(letter):04x}.png"
            out_path = out_dir / out_name
            if skip_existing and out_path.exists():
                pass
            else:
                crop = crop_letter(img, [x1, y1, x2, y2], pad_pct=pad_pct, make_square=make_square, out_size=out_size)
                crop.save(out_path, format="PNG", optimize=True)
            out_rows.append(
                [
                    split,
                    class_name,
                    letter,
                    uid,
                    str(i),
                    str(img_path),
                    json.dumps([x1, y1, x2, y2], ensure_ascii=False),
                    str(out_path),
                ]
            )
        return out_rows

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for item in manifest_split_list:
            if len(item) == 2:
                mp, forced_split = item
                split_filter = None
            elif len(item) == 3:
                mp, forced_split, split_filter = item
            else:
                raise ValueError(f"Invalid manifest export spec: {item!r}")
            mp = Path(mp)
            if not mp.exists():
                continue
            with mp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    futures.append(ex.submit(process_record, rec, mp, forced_split, split_filter))
        for fu in as_completed(futures):
            try:
                rows = fu.result()
                for row in rows:
                    if row and row[0] != "__MISSING_IMAGE__":
                        meta_rows.append(row)
            except Exception:
                continue

    meta_path = dst_root / "meta.csv"
    import csv

    with meta_path.open("w", encoding="utf-8", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(lock_rows)
        w.writerows(meta_rows)

    return {"dst_root": str(dst_root), "n_crops": len(meta_rows), "meta_csv": str(meta_path)}
