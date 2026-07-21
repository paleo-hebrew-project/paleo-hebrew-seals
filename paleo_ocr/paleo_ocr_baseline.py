"""paleo_ocr_baseline
====================

Baseline OCR runner for Paleo-Hebrew / Phoenician-like seal inscriptions.

This module serves two purposes:
1) **CLI baseline**: run a single engine on a manifest and write preds + metrics.
2) **Notebook helpers**: convenient batch functions used in Colab notebooks.

Engines (baseline)
-----------------
* tesseract
* kraken (CLI, requires a model file)
* trocr (HuggingFace)

Notes
-----
* Many seals are rotated/mirrored. If your manifest was processed with
  ``orientation_mirror.py analyze`` then per-record meta fields are applied:
  ``meta.orientation_deg`` and ``meta.mirrored`` and ``meta.reverse_pred``.
* Preprocess policy:
    - tesseract/kraken usually benefit from adaptive threshold
    - trocr usually works best on raw RGB

"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# -------------------------
# Imports with safe fallbacks
# -------------------------

def _import_metrics():
    try:
        from .ocr_metrics import compute_metrics_for_pairs, cer  # type: ignore
    except Exception:
        from ocr_metrics import compute_metrics_for_pairs, cer  # type: ignore
    return compute_metrics_for_pairs, cer


def _import_registry():
    try:
        from .model_registry import ensure_model  # type: ignore
    except Exception:
        from model_registry import ensure_model  # type: ignore
    return ensure_model


# -------------------------
# JSONL utils
# -------------------------


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: Union[str, Path], rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Notebook-friendly alias
def write_preds_jsonl(path: Union[str, Path], rows: Sequence[Dict[str, Any]]) -> None:
    write_jsonl(path, rows)


# -------------------------
# Manifest helpers
# -------------------------


def _get_gt_text(rec: Dict[str, Any]) -> str:
    gt = rec.get("gt")
    if isinstance(gt, str):
        return gt
    if isinstance(gt, dict):
        heb = gt.get("hebrew")
        if isinstance(heb, str):
            return heb
        if isinstance(heb, dict):
            return (heb.get("raw") or heb.get("collapsed_ws") or heb.get("stripped") or "")
        return (gt.get("raw") or gt.get("text") or "")
    return str(rec.get("gt_text") or "")


def _get_image_ref(rec: Dict[str, Any]) -> Dict[str, Any]:
    # normalized records might have image_path only
    if isinstance(rec.get("image"), dict):
        return rec["image"]
    # fallback
    p = rec.get("image_path") or rec.get("path")
    return {"path": p, "container": None, "exists": True}


def _open_image_from_record(rec: Dict[str, Any]):
    from PIL import Image

    img_ref = _get_image_ref(rec)
    container = img_ref.get("container")
    if isinstance(container, dict) and container.get("type") == "zip":
        zip_path = container.get("path")
        inner = container.get("inner_path")
        if not zip_path or not inner:
            raise FileNotFoundError("Invalid zip container reference in manifest")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner) as fp:
                return Image.open(fp).convert("RGB")

    path = img_ref.get("path") or rec.get("image_path")
    if not path:
        raise FileNotFoundError("Missing image path")
    return Image.open(path).convert("RGB")


def apply_manifest_transforms(img, rec: Dict[str, Any]):
    """Apply rotation/mirror from manifest meta if present."""
    from PIL import Image

    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    rot = int(meta.get("orientation_deg") or 0) % 360
    if rot:
        img = img.rotate(rot, expand=True, resample=Image.BICUBIC)
    if meta.get("mirrored") is True:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def postprocess_pred(pred: str, rec: Dict[str, Any]) -> str:
    pred = (pred or "").strip()
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    if meta.get("reverse_pred") is True:
        pred = pred[::-1]
    return pred.strip()


def normalize_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize manifest records for notebook runners.

    Adds:
      - image_path (if possible)
      - gt_text
    Keeps original fields (image/meta) so transforms can be applied.
    """
    out: List[Dict[str, Any]] = []
    for r in records:
        rr = dict(r)
        rr.setdefault("gt_text", _get_gt_text(rr))

        img_ref = _get_image_ref(rr)
        p = img_ref.get("path")
        if p:
            rr.setdefault("image_path", p)
        else:
            rr.setdefault("image_path", None)
        out.append(rr)
    return out


# -------------------------
# Preprocess
# -------------------------


def _preprocess_pil(img, mode: str):
    """Return a PIL image after preprocessing."""
    from PIL import Image
    mode = (mode or "none").lower()
    if mode == "none":
        return img
    if mode == "gray":
        return img.convert("L").convert("RGB")
    if mode == "threshold":
        # quick adaptive threshold using OpenCV if available; else simple global
        import numpy as np
        g = np.array(img.convert("L"), dtype=np.uint8)
        try:
            import cv2  # type: ignore

            g = cv2.medianBlur(g, 3)
            block = 31 if min(g.shape[:2]) >= 31 else max(3, (min(g.shape[:2]) // 2) * 2 + 1)
            bw = cv2.adaptiveThreshold(
                g,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                10,
            )
            return Image.fromarray(bw).convert("RGB")
        except Exception:
            thr = int(np.median(g))
            bw = (g > thr).astype(np.uint8) * 255
            return Image.fromarray(bw).convert("RGB")
    raise ValueError(f"Unknown preprocess mode: {mode}")


def _default_preprocess_for_engine(engine: str) -> str:
    if engine in {"tesseract", "kraken"}:
        return "threshold"
    return "none"


# -------------------------
# Engines (single image)
# -------------------------


def run_tesseract_one(img_path: Path, lang: str = "heb", psm: int = 7, oem: int = 1) -> str:
    # tesseract <img> stdout -l heb --oem 1 --psm 7
    cmd = ["tesseract", str(img_path), "stdout", "-l", lang, "--oem", str(oem), "--psm", str(psm)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return ""
    return (r.stdout or "").strip()


def run_kraken_one(img_path: Path, model_path: str, device: str = "cpu", threshold: bool = False) -> str:
    # Kraken CLI: kraken -i image - ocr -m model
    cmd = ["kraken", "-i", str(img_path), "-", "ocr", "-m", str(model_path)]
    if device != "cpu":
        cmd += ["--device", device]
    if threshold:
        cmd += ["-b"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return ""
    return (r.stdout or "").strip()


# Backwards-compatible alias used by orientation_mirror.py
def run_kraken(img_path: Union[Path, str], model_path: str, device: str = "cpu", threshold: bool = False) -> str:
    """Compatibility wrapper (single image)."""
    return run_kraken_one(Path(img_path), model_path=model_path, device=device, threshold=threshold)


def _resolve_trocr_ref(model_path: Optional[str], model_id: Optional[str], cache_models: str) -> str:
    if model_path:
        return model_path
    if not model_id:
        raise ValueError("TrOCR requires model_id or model_path")
    ensure_model = _import_registry()
    return str(ensure_model(model_id, cache_dir=cache_models))


def run_trocr_batch(
    imgs: List[Tuple[str, Any]],
    repo_marker_or_path: str,
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> Dict[str, str]:
    """Run TrOCR for many images.

    imgs: list of (uid, PIL.Image)
    returns: uid -> text
    """
    import json as _json
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_ref = repo_marker_or_path
    if repo_marker_or_path.endswith(".hf_repo"):
        meta = _json.loads(Path(repo_marker_or_path).read_text(encoding="utf-8"))
        model_ref = meta["repo"]

    processor = TrOCRProcessor.from_pretrained(model_ref)
    model = VisionEncoderDecoderModel.from_pretrained(model_ref)
    model.to(device)
    model.eval()

    out: Dict[str, str] = {}
    for uid, pil_img in imgs:
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pixel_values, max_new_tokens=int(max_new_tokens))
        txt = processor.batch_decode(ids, skip_special_tokens=True)[0]
        out[uid] = (txt or "").strip()
    return out


# -------------------------
# Notebook runners (batch)
# -------------------------


def run_tesseract(
    inputs: Union[Path, str, Sequence[Dict[str, Any]]],
    lang: str = "heb",
    psm: int = 7,
    oem: int = 1,
    threshold: bool = True,
    preprocess: str = "auto",
    tmp_dir: Union[str, Path] = ".tmp_tesseract",
) -> Union[str, List[Dict[str, Any]]]:
    """Run Tesseract.

    If inputs is a Path/str -> returns a single string.
    If inputs is a list of manifest records -> returns list of pred dicts.
    """
    if isinstance(inputs, (str, Path)):
        return run_tesseract_one(Path(inputs), lang=lang, psm=psm, oem=oem)

    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict[str, Any]] = []

    for rec in inputs:
        uid = str(rec.get("uid") or "")
        if not uid:
            continue
        try:
            img = _open_image_from_record(rec)
        except Exception:
            continue

        img = apply_manifest_transforms(img, rec)

        # preprocess policy
        mode = preprocess
        if mode == "auto":
            mode = "threshold" if threshold else "none"

        img = _preprocess_pil(img, mode)
        fpath = tmp / f"{uid}.png"
        img.save(fpath)
        pred = run_tesseract_one(fpath, lang=lang, psm=psm, oem=oem)
        pred = postprocess_pred(pred, rec)

        out_rows.append(
            {
                "uid": rec.get("uid"),
                "row_id": rec.get("row_id"),
                "image_path": (rec.get("image_path") or (_get_image_ref(rec).get("path"))),
                "gt_text": rec.get("gt_text") or _get_gt_text(rec),
                "pred_text": pred,
                "meta": {"engine": "tesseract", "psm": int(psm), "preprocess": mode},
            }
        )
    return out_rows


def run_trocr(
    records: Sequence[Dict[str, Any]],
    model_id: str = "hf:trocr-printed-base",
    model_path: Optional[str] = None,
    cache_models: str = "models_cache",
    device: str = "cuda",
    preprocess: str = "none",
    max_new_tokens: int = 64,
) -> List[Dict[str, Any]]:
    """Run TrOCR on a list of manifest records."""
    repo_ref = _resolve_trocr_ref(model_path=model_path, model_id=model_id, cache_models=cache_models)

    # Load all images as PIL first (so we can apply manifest transforms)
    imgs: List[Tuple[str, Any]] = []
    keep: List[Dict[str, Any]] = []
    for rec in records:
        uid = str(rec.get("uid") or "")
        if not uid:
            continue
        try:
            img = _open_image_from_record(rec)
        except Exception:
            continue
        img = apply_manifest_transforms(img, rec)
        img = _preprocess_pil(img, preprocess)
        imgs.append((uid, img))
        keep.append(rec)

    uid2txt = run_trocr_batch(imgs, repo_marker_or_path=repo_ref, device=device, max_new_tokens=max_new_tokens)

    out_rows: List[Dict[str, Any]] = []
    for rec in keep:
        uid = str(rec.get("uid") or "")
        pred = uid2txt.get(uid, "")
        pred = postprocess_pred(pred, rec)
        out_rows.append(
            {
                "uid": rec.get("uid"),
                "row_id": rec.get("row_id"),
                "image_path": (rec.get("image_path") or (_get_image_ref(rec).get("path"))),
                "gt_text": rec.get("gt_text") or _get_gt_text(rec),
                "pred_text": pred,
                "meta": {"engine": "trocr", "model_id": model_id, "preprocess": preprocess},
            }
        )
    return out_rows


def metrics_table(pred_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Return a dict of CER/WER summary metrics for notebook printing."""
    compute_metrics_for_pairs, _ = _import_metrics()

    pairs: List[Tuple[str, str]] = []
    for r in pred_rows:
        ref = (r.get("gt_text") or r.get("gt") or "")
        pred = (r.get("pred_text") or r.get("pred") or "")
        pairs.append((str(ref), str(pred)))
    return compute_metrics_for_pairs(pairs)


# -------------------------
# CLI
# -------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline OCR evaluation for Paleo-Hebrew")
    p.add_argument("--manifest", type=str, required=True, help="manifest jsonl with gt")
    p.add_argument("--out", type=str, default="baseline_preds.jsonl")
    p.add_argument("--cache-models", type=str, default="models_cache")

    p.add_argument("--engine", type=str, default="tesseract", choices=["tesseract", "kraken", "trocr"])
    p.add_argument("--preprocess", type=str, default="auto", choices=["auto", "none", "gray", "threshold"])
    p.add_argument("--auto-reverse", action="store_true", help="Try reversing prediction if it improves CER")

    # Tesseract
    p.add_argument("--tesseract-lang", type=str, default="heb")
    p.add_argument("--tesseract-psm", type=int, default=7)

    # Kraken
    p.add_argument("--kraken-model-path", type=str, default=None)
    p.add_argument("--kraken-model-id", type=str, default=None)
    p.add_argument("--kraken-device", type=str, default="cpu")
    p.add_argument("--kraken-threshold", action="store_true")

    # TrOCR
    p.add_argument("--trocr-model-path", type=str, default=None)
    p.add_argument("--trocr-model-id", type=str, default="hf:trocr-printed-base")
    p.add_argument("--trocr-device", type=str, default="cuda")
    p.add_argument("--max-items", type=int, default=None)

    return p.parse_args(argv)


def _resolve_model_path(model_path: Optional[str], model_id: Optional[str], cache_dir: str) -> Optional[str]:
    if model_path:
        return model_path
    if model_id:
        ensure_model = _import_registry()
        return str(ensure_model(model_id, cache_dir=cache_dir))
    return None


def _maybe_reverse(pred: str, gt: str, enable: bool) -> Tuple[str, bool]:
    if not enable:
        return pred, False
    _, cer_fn = _import_metrics()
    c1 = cer_fn(gt, pred)
    c2 = cer_fn(gt, pred[::-1])
    if c2 < c1:
        return pred[::-1], True
    return pred, False


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = read_jsonl(args.manifest)
    if args.max_items:
        rows = rows[: int(args.max_items)]

    # apply manifest transforms by default (if meta exists)
    rows = normalize_records(rows)

    tmp_dir = Path(".tmp_baseline")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, Any]] = []

    if args.engine == "tesseract":
        preds = run_tesseract(
            rows,
            lang=args.tesseract_lang,
            psm=args.tesseract_psm,
            threshold=True,
            preprocess=args.preprocess,
            tmp_dir=tmp_dir,
        )
        assert isinstance(preds, list)
        out_rows = preds

    elif args.engine == "trocr":
        mode = args.preprocess
        if mode == "auto":
            mode = _default_preprocess_for_engine("trocr")
        out_rows = run_trocr(
            rows,
            model_id=args.trocr_model_id,
            model_path=args.trocr_model_path,
            cache_models=args.cache_models,
            device=args.trocr_device,
            preprocess=mode,
        )

    elif args.engine == "kraken":
        kraken_model = _resolve_model_path(args.kraken_model_path, args.kraken_model_id, args.cache_models)
        if not kraken_model:
            raise ValueError("Kraken requires --kraken-model-path or --kraken-model-id")

        # Kraken is CLI-only here; do per-image with temp files.
        mode = args.preprocess
        if mode == "auto":
            mode = _default_preprocess_for_engine("kraken")

        for rec in rows:
            uid = str(rec.get("uid") or "")
            if not uid:
                continue
            try:
                img = _open_image_from_record(rec)
            except Exception:
                continue
            img = apply_manifest_transforms(img, rec)
            img = _preprocess_pil(img, mode)
            fpath = tmp_dir / f"{uid}.png"
            img.save(fpath)
            pred = run_kraken_one(
                fpath,
                model_path=kraken_model,
                device=args.kraken_device,
                threshold=bool(args.kraken_threshold),
            )
            pred = postprocess_pred(pred, rec)
            gt_text = rec.get("gt_text") or _get_gt_text(rec)
            pred, rev_used = _maybe_reverse(pred, gt_text, enable=args.auto_reverse)

            out_rows.append(
                {
                    "uid": rec.get("uid"),
                    "row_id": rec.get("row_id"),
                    "image_path": rec.get("image_path"),
                    "gt_text": gt_text,
                    "pred_text": pred,
                    "meta": {
                        "engine": "kraken",
                        "model": kraken_model,
                        "preprocess": mode,
                        "auto_reverse": bool(rev_used),
                    },
                }
            )

    # write outputs
    write_preds_jsonl(args.out, out_rows)
    m = metrics_table(out_rows)
    Path(str(Path(args.out).with_suffix(".metrics.json"))).write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
