"""orientation_mirror
=====================

Estimate orientation and (optionally) mirror/reverse settings for the
Paleo‑Hebrew seal images and write the results back into the dataset
manifest.

Context
-------
In seal corpora, images often come with arbitrary rotations; additionally,
some impressions may be left-right mirrored depending on how the seal was
captured. If these effects are not normalized, CER/WER can explode and
detector/recognizer training can become unstable.

This module operates *without* per-character bounding boxes. It works at
image level and updates the per-image manifest records with:

* meta.orientation_deg: {0, 90, 180, 270} (rotation to apply to make text
  roughly horizontal)
* meta.layout_hint: "horizontal" | "vertical" (based on projection profiles)
* meta.sort_primary: "x" | "y" (useful later when assembling glyphs)
* meta.n_lines_est: estimated number of text lines (rough heuristic)
* meta.mirrored: bool|None (left-right mirror likely?)
* meta.reading_dir: "rtl" (default for Paleo-Hebrew)
* meta.reverse_pred: bool (reverse predicted strings before scoring/metrics)

Notes on mirror detection
-------------------------
Purely image-based heuristics cannot reliably detect left-right mirroring
for short inscriptions. Therefore mirror/reverse detection is implemented
as an *optional* OCR-assisted stage:

* mode=heuristic: estimate only rotation/layout from the image.
* mode=ocr: try to decide mirrored/reverse using an OCR engine (tesseract)
  by maximizing a plausibility score.
* mode=oracle: if GT exists in the manifest, choose mirrored/reverse by
  minimizing CER between OCR output and GT (still relies on OCR but uses GT).
* mode=auto (default): heuristic for rotation, then oracle/ocr if possible.

CLI
---
Analyze a manifest and write an updated manifest:

    python orientation_mirror.py analyze \
      --manifest manifest.jsonl \
      --out manifest_oriented.jsonl \
      --mode auto \
      --engine tesseract \
      --tesseract-lang heb \
      --max-items 2000 \
      --stats-out orient_stats.json

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("OpenCV (cv2) is required for orientation_mirror.py") from e

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise RuntimeError("Pillow is required for orientation_mirror.py") from e

try:
    # Reuse shared metrics if available (recommended).
    from ocr_metrics import character_error_rate  # type: ignore
except Exception:  # pragma: no cover
    # Minimal fallback to keep this module usable standalone.
    def character_error_rate(pred: str, ref: str, ignore_spaces: bool = False) -> float:
        """Compute character error rate (CER) using Levenshtein distance."""
        if ignore_spaces:
            pred = re.sub(r"\s+", "", pred)
            ref = re.sub(r"\s+", "", ref)

        # Classic DP edit distance (O(n*m)); fine for short seal strings.
        n, m = len(pred), len(ref)
        if m == 0:
            return 0.0 if n == 0 else 1.0

        prev = list(range(m + 1))
        for i in range(1, n + 1):
            cur = [i] + [0] * m
            pc = pred[i - 1]
            for j in range(1, m + 1):
                cost = 0 if pc == ref[j - 1] else 1
                cur[j] = min(
                    prev[j] + 1,      # deletion
                    cur[j - 1] + 1,   # insertion
                    prev[j - 1] + cost,  # substitution
                )
            prev = cur
        dist = prev[m]
        return float(dist) / float(max(1, m))


HEB_CHARS_RE = re.compile(r"[\u0590-\u05FF]")
WS_RE = re.compile(r"\s+")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _open_image_from_record(rec: Dict[str, Any]) -> Image.Image:
    img_ref = rec.get("image") or {}
    container = img_ref.get("container")
    if container and container.get("type") == "zip":
        zip_path = container.get("path")
        inner = container.get("inner_path")
        if not zip_path or not inner:
            raise FileNotFoundError("Invalid zip container reference in manifest")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner) as fp:
                img = Image.open(fp)
                return img.convert("RGB")

    path = img_ref.get("path")
    if not path:
        raise FileNotFoundError("Missing image.path in manifest")
    img = Image.open(path)
    return img.convert("RGB")


def _to_gray_np(img: Image.Image, max_side: int = 512) -> np.ndarray:
    """Convert to grayscale numpy and optionally downscale for speed."""
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    gray = np.array(img.convert("L"), dtype=np.uint8)
    return gray


def _adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    """Return binary image with ink==1, background==0."""
    # Normalize contrast a bit
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.medianBlur(g, 3)

    # Adaptive threshold; invert so ink=1
    # Block size must be odd.
    block = 31 if min(g.shape[:2]) >= 31 else max(3, (min(g.shape[:2]) // 2) * 2 + 1)
    bw = cv2.adaptiveThreshold(
        g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        10,
    )
    bw = (bw > 0).astype(np.uint8)

    # Small morphological clean-up
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return bw


def _projection_scores(bw: np.ndarray) -> Tuple[float, float, float, int]:
    """Compute projection-based orientation hints.

    Returns
    -------
    hv_score : float
        Positive => more horizontal (row variation larger), negative => more vertical.
    abs_score : float
        Absolute magnitude of hv_score.
    density : float
        Fraction of ink pixels.
    n_lines_est : int
        Estimated number of text lines (roughly peaks in row projection).
    """
    h, w = bw.shape[:2]
    ink = bw.astype(np.float32)
    density = float(ink.mean())

    row = ink.sum(axis=1)
    col = ink.sum(axis=0)
    row_mean = float(row.mean()) + 1e-6
    col_mean = float(col.mean()) + 1e-6
    row_var = float(row.std() / row_mean)
    col_var = float(col.std() / col_mean)
    hv_score = row_var - col_var

    # Smooth row profile and count prominent peaks
    k = max(3, (h // 50) * 2 + 1)  # odd kernel
    row_s = cv2.GaussianBlur(row.reshape(-1, 1), (1, k), 0).reshape(-1)
    thr = row_s.max() * 0.35
    peaks = 0
    for i in range(1, len(row_s) - 1):
        if row_s[i] > thr and row_s[i] >= row_s[i - 1] and row_s[i] >= row_s[i + 1]:
            peaks += 1
    n_lines_est = int(max(1, min(peaks, 10))) if peaks > 0 else 1

    return float(hv_score), float(abs(hv_score)), density, n_lines_est


def _connected_component_quality(bw: np.ndarray) -> float:
    """Heuristic quality measure: rewards coherent components, penalizes speckle."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return -5.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    if areas.size == 0:
        return -5.0
    small_frac = float((areas < 6).mean())
    med = float(np.median(areas))
    # Mildly prefer moderate component counts
    n = float(areas.size)
    score = math.log1p(n) + 0.5 * math.log1p(med) - 3.0 * small_frac
    return float(score)


def _apply_rotation_pil(img: Image.Image, rot_deg: int) -> Image.Image:
    rot_deg = int(rot_deg) % 360
    if rot_deg == 0:
        return img
    # PIL rotates counter-clockwise for positive angles.
    return img.rotate(rot_deg, expand=True, resample=Image.BICUBIC)


def _apply_mirror_pil(img: Image.Image, mirrored: bool) -> Image.Image:
    if not mirrored:
        return img
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def _normalize_text_basic(s: str) -> str:
    s = s.strip()
    s = WS_RE.sub(" ", s)
    return s


def _hebrew_ratio(s: str) -> float:
    if not s:
        return 0.0
    heb = len(HEB_CHARS_RE.findall(s))
    return float(heb) / float(max(1, len(s)))


class OCREngine:
    name: str

    def predict(self, img: Image.Image) -> str:
        raise NotImplementedError


class TesseractEngine(OCREngine):
    def __init__(self, lang: str = "heb", psm: int = 7) -> None:
        self.name = "tesseract"
        self.lang = lang
        self.psm = int(psm)
        import pytesseract  # local import

        self._pt = pytesseract

    def predict(self, img: Image.Image) -> str:
        cfg = f"--psm {self.psm}"
        try:
            txt = self._pt.image_to_string(img, lang=self.lang, config=cfg)
        except Exception:
            # If the requested lang isn't available, retry without lang.
            txt = self._pt.image_to_string(img, config=cfg)
        return _normalize_text_basic(txt)


class KrakenEngine(OCREngine):
    """Kraken OCR engine wrapper.

    We keep this lightweight by reusing the toolkit's `run_kraken` helper
    (CLI wrapper). This avoids depending on Kraken's internal Python APIs.
    """

    def __init__(self, model_path: str, device: str = "cpu", threshold: bool = True) -> None:
        self.name = "kraken"
        self.model_path = str(model_path)
        self.device = str(device)
        self.threshold = bool(threshold)

        try:
            from paleo_ocr_baseline import run_kraken  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Kraken engine requested but paleo_ocr_baseline.run_kraken is not available. "
                "Ensure you run this script from the toolkit folder (python paleo_ocr/orientation_mirror.py ...)"
            ) from e

        self._run_kraken = run_kraken
        self._tmp_dir = Path(tempfile.gettempdir()) / "paleo_ocr_orient_kraken"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def predict(self, img: Image.Image) -> str:
        tmp_path = self._tmp_dir / f"kr_{uuid.uuid4().hex}.png"
        img.save(tmp_path)
        try:
            txt = self._run_kraken(tmp_path, model_path=self.model_path, device=self.device, threshold=self.threshold)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)  # py3.8+ on Linux in Colab is ok
            except Exception:
                pass
        return _normalize_text_basic(txt)


def _make_engine(
    name: str,
    tesseract_lang: str,
    tesseract_psm: int,
    kraken_model_path: Optional[str] = None,
    kraken_model_id: Optional[str] = None,
    cache_models: str = "models_cache",
    kraken_device: str = "cpu",
    kraken_threshold: bool = True,
) -> Optional[OCREngine]:
    name = (name or "").lower().strip()
    if name in ("tesseract", "tes"):
        try:
            return TesseractEngine(lang=tesseract_lang, psm=tesseract_psm)
        except Exception:
            return None
    if name in ("kraken", "krk"):
        path = kraken_model_path
        if (not path) and kraken_model_id:
            try:
                from model_registry import ensure_model  # type: ignore

                path = str(ensure_model(kraken_model_id, cache_dir=cache_models))
            except Exception as e:
                print(f"[orientation_mirror] WARN: failed to resolve kraken model id {kraken_model_id}: {e}")
                path = None

        if not path:
            print(
                "[orientation_mirror] WARN: engine=kraken requested but no model provided. "
                "Use --kraken-model-path or --kraken-model-id (configured in model_registry.py)."
            )
            return None

        try:
            return KrakenEngine(model_path=path, device=kraken_device, threshold=kraken_threshold)
        except Exception as e:
            print(f"[orientation_mirror] WARN: failed to init Kraken engine: {e}")
            return None
    return None


@dataclass
class RotationResult:
    rot_deg: int
    layout_hint: str
    sort_primary: str
    n_lines_est: int
    score_abs: float
    density: float
    cc_score: float
    confidence: float


def estimate_rotation_layout(img: Image.Image, max_side: int = 512) -> RotationResult:
    """Estimate coarse rotation and layout hints using image-only heuristics."""
    candidates = [0, 90, 180, 270]
    scored: List[Tuple[int, float, float, float, int]] = []

    for rot in candidates:
        g = _to_gray_np(_apply_rotation_pil(img, rot), max_side=max_side)
        bw = _adaptive_binarize(g)
        hv, abs_score, density, n_lines_est = _projection_scores(bw)
        cc = _connected_component_quality(bw)

        # Penalize extreme densities (almost blank or almost full)
        density_pen = 0.0
        if density < 0.01:
            density_pen = -3.0
        elif density > 0.35:
            density_pen = -2.0

        total = abs_score + 0.15 * cc + density_pen
        scored.append((rot, total, hv, density, n_lines_est))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_rot, best_total, best_hv, best_density, best_lines = scored[0]
    second = scored[1][1] if len(scored) > 1 else best_total
    # Confidence: normalized margin
    margin = max(0.0, best_total - second)
    conf = float(margin / (abs(best_total) + 1e-6))

    layout_hint = "horizontal" if best_hv >= 0 else "vertical"
    sort_primary = "x" if layout_hint == "horizontal" else "y"

    # Recompute cc_score for the chosen rotation for reporting.
    g = _to_gray_np(_apply_rotation_pil(img, best_rot), max_side=max_side)
    bw = _adaptive_binarize(g)
    cc_score = _connected_component_quality(bw)
    _, abs_score, _, _ = _projection_scores(bw)

    return RotationResult(
        rot_deg=int(best_rot),
        layout_hint=layout_hint,
        sort_primary=sort_primary,
        n_lines_est=int(best_lines),
        score_abs=float(abs_score),
        density=float(best_density),
        cc_score=float(cc_score),
        confidence=float(conf),
    )


def decide_mirror_reverse(
    img: Image.Image,
    rot_deg: int,
    engine: Optional[OCREngine],
    mode: str,
    gt_text: Optional[str],
    prefer_rtl: bool = True,
    max_side: int = 768,
) -> Tuple[Optional[bool], bool, Dict[str, Any]]:
    """Try to decide mirrored and reverse_pred.

    Returns
    -------
    mirrored : bool|None
        None if no strong evidence.
    reverse_pred : bool
        Whether to reverse predicted strings before metrics.
    debug : dict
        Scores and predictions for inspection.
    """
    mode = (mode or "auto").lower().strip()
    debug: Dict[str, Any] = {"mode": mode, "candidates": []}

    # Defaults: Paleo-Hebrew is RTL; many OCR engines output LTR order.
    reverse_default = True if prefer_rtl else False

    if engine is None or mode == "heuristic":
        return None, reverse_default, debug

    # Prepare candidates
    candidates: List[Tuple[bool, bool]] = [(False, False), (False, True), (True, False), (True, True)]
    best: Optional[Tuple[bool, bool, float]] = None

    # Use GT if oracle and GT exists
    use_gt = (mode in ("oracle", "auto")) and (gt_text is not None and str(gt_text).strip() != "")
    gt_norm = None
    if use_gt:
        gt_norm = WS_RE.sub("", str(gt_text))  # no spaces

    base = _apply_rotation_pil(img, rot_deg)
    # Downscale to speed OCR scoring
    w, h = base.size
    if max(w, h) > max_side:
        s = max_side / float(max(w, h))
        base = base.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)

    for mirrored, rev in candidates:
        cand_img = _apply_mirror_pil(base, mirrored)
        pred = ""
        try:
            pred = engine.predict(cand_img)
        except Exception as e:
            debug["error"] = f"{type(e).__name__}: {e}"
            continue

        pred_use = pred[::-1] if rev else pred
        pred_use = _normalize_text_basic(pred_use)
        pred_nospace = WS_RE.sub("", pred_use)
        heb_ratio = _hebrew_ratio(pred_use)

        if use_gt and gt_norm is not None:
            cer = character_error_rate(pred_nospace, gt_norm)
            score = -cer + 0.15 * heb_ratio
        else:
            # Weak unsupervised plausibility
            score = heb_ratio + 0.01 * min(40, len(pred_nospace))

        debug["candidates"].append(
            {
                "mirrored": mirrored,
                "reverse_pred": rev,
                "pred": pred,
                "pred_used": pred_use,
                "hebrew_ratio": heb_ratio,
                "score": score,
            }
        )

        if best is None or score > best[2]:
            best = (mirrored, rev, score)

    if best is None:
        return None, reverse_default, debug

    # Decide if mirror is confident: compare best mirrored vs best non-mirrored
    best_m = max((c for c in debug["candidates"] if c["mirrored"]), key=lambda x: x["score"], default=None)
    best_nm = max((c for c in debug["candidates"] if not c["mirrored"]), key=lambda x: x["score"], default=None)
    mirrored_out: Optional[bool] = None
    if best_m and best_nm:
        diff = float(best_m["score"] - best_nm["score"])
        if abs(diff) >= 0.05:
            mirrored_out = bool(diff > 0)
        else:
            mirrored_out = None

    reverse_out = bool(best[1])
    debug["chosen"] = {"mirrored": mirrored_out, "reverse_pred": reverse_out, "score": float(best[2])}
    return mirrored_out, reverse_out, debug


def _get_gt_hebrew(rec: Dict[str, Any]) -> Optional[str]:
    gt = (rec.get("gt") or {}).get("hebrew")
    if isinstance(gt, dict):
        # prefer 'raw'
        return gt.get("raw") or gt.get("collapsed_ws") or gt.get("stripped")
    if isinstance(gt, str):
        return gt
    return None


def analyze_manifest(
    records: List[Dict[str, Any]],
    mode: str = "auto",
    engine_name: str = "tesseract",
    tesseract_lang: str = "heb",
    tesseract_psm: int = 7,
    kraken_model_path: Optional[str] = None,
    kraken_model_id: Optional[str] = None,
    cache_models: str = "models_cache",
    kraken_device: str = "cpu",
    kraken_threshold: bool = True,
    max_side_rot: int = 512,
    max_side_ocr: int = 768,
    max_items: Optional[int] = None,
    seed: int = 42,
    prefer_rtl: bool = True,
    write_debug: bool = False,
    debug_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    if max_items is not None and max_items < len(idxs):
        rng.shuffle(idxs)
        idxs = idxs[: int(max_items)]

    engine = _make_engine(
        engine_name,
        tesseract_lang=tesseract_lang,
        tesseract_psm=tesseract_psm,
        kraken_model_path=kraken_model_path,
        kraken_model_id=kraken_model_id,
        cache_models=cache_models,
        kraken_device=kraken_device,
        kraken_threshold=kraken_threshold,
    )

    if write_debug:
        if not debug_dir:
            debug_dir = "orient_debug"
        os.makedirs(debug_dir, exist_ok=True)

    # Stats accumulation
    rot_counts: Dict[int, int] = {0: 0, 90: 0, 180: 0, 270: 0}
    mirrored_counts = {"true": 0, "false": 0, "none": 0}
    reverse_counts = {"true": 0, "false": 0}
    confs: List[float] = []
    densities: List[float] = []
    line_est: List[int] = []

    updated = records
    for j, i in enumerate(idxs):
        rec = updated[i]
        if not ((rec.get("image") or {}).get("exists", True)):
            continue
        try:
            img = _open_image_from_record(rec)
        except Exception:
            continue

        rot_res = estimate_rotation_layout(img, max_side=max_side_rot)
        rot_counts[rot_res.rot_deg] = rot_counts.get(rot_res.rot_deg, 0) + 1
        confs.append(rot_res.confidence)
        densities.append(rot_res.density)
        line_est.append(rot_res.n_lines_est)

        gt_hebrew = _get_gt_hebrew(rec)
        mirrored, reverse_pred, debug = decide_mirror_reverse(
            img,
            rot_deg=rot_res.rot_deg,
            engine=engine,
            mode=mode,
            gt_text=gt_hebrew,
            prefer_rtl=prefer_rtl,
            max_side=max_side_ocr,
        )

        if mirrored is None:
            mirrored_counts["none"] += 1
        elif mirrored:
            mirrored_counts["true"] += 1
        else:
            mirrored_counts["false"] += 1
        reverse_counts["true" if reverse_pred else "false"] += 1

        meta = rec.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            rec["meta"] = meta

        meta["orientation_deg"] = int(rot_res.rot_deg)
        meta["layout_hint"] = rot_res.layout_hint
        meta["sort_primary"] = rot_res.sort_primary
        meta["n_lines_est"] = int(rot_res.n_lines_est)
        meta["orientation_confidence"] = float(rot_res.confidence)
        meta["binarize_density"] = float(rot_res.density)
        meta["cc_score"] = float(rot_res.cc_score)

        meta["reading_dir"] = "rtl" if prefer_rtl else "ltr"
        meta["reverse_pred"] = bool(reverse_pred)
        meta["mirrored"] = mirrored

        if write_debug and debug_dir:
            uid = str(rec.get("uid", i))
            try:
                dbg_path = os.path.join(debug_dir, f"{uid}.json")
                with open(dbg_path, "w", encoding="utf-8") as f:
                    json.dump(debug, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    stats: Dict[str, Any] = {
        "n_records": len(records),
        "n_analyzed": len(idxs),
        "rotation_counts": rot_counts,
        "mirrored_counts": mirrored_counts,
        "reverse_counts": reverse_counts,
        "confidence": {
            "mean": float(statistics.mean(confs)) if confs else 0.0,
            "median": float(statistics.median(confs)) if confs else 0.0,
        },
        "density": {
            "mean": float(statistics.mean(densities)) if densities else 0.0,
            "median": float(statistics.median(densities)) if densities else 0.0,
        },
        "n_lines_est": {
            "mean": float(statistics.mean(line_est)) if line_est else 0.0,
            "median": float(statistics.median(line_est)) if line_est else 0.0,
        },
        "engine_available": bool(engine is not None),
        "engine_name": engine_name,
        "mode": mode,
    }
    return updated, stats


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate orientation/mirror and update dataset manifest")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Analyze manifest and write updated manifest")
    a.add_argument("--manifest", required=True, help="Input manifest.jsonl")
    a.add_argument("--out", required=True, help="Output manifest.jsonl (updated)")
    a.add_argument("--mode", default="auto", choices=["auto", "heuristic", "ocr", "oracle"],
                   help="How to decide mirror/reverse. Rotation is always heuristic.")
    a.add_argument("--engine", default="tesseract", help="OCR engine for mirror/reverse scoring (default: tesseract)")
    a.add_argument("--tesseract-lang", default="heb", help="Tesseract language (default: heb)")
    a.add_argument("--tesseract-psm", type=int, default=7, help="Tesseract PSM (default: 7)")

    # Kraken options (used when --engine kraken)
    a.add_argument("--kraken-model-path", default=None, help="Path to Kraken model file (.mlmodel/.pyrnn.gz)")
    a.add_argument(
        "--kraken-model-id",
        default=None,
        help="Model id to resolve via model_registry.ensure_model (e.g. kraken:hebrew-printed:v1)",
    )
    a.add_argument(
        "--cache-models",
        default="models_cache",
        help="Cache directory for model registry downloads (default: models_cache)",
    )
    a.add_argument("--kraken-device", default="cpu", help="Kraken device string (cpu, cuda, cuda:0, etc.)")
    a.add_argument(
        "--no-kraken-threshold",
        dest="kraken_threshold",
        action="store_false",
        default=True,
        help="Disable thresholding before Kraken OCR (default: enabled)",
    )
    a.add_argument("--max-items", type=int, default=None, help="Analyze only a subset for quick experiments")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--prefer-rtl", action="store_true", default=True,
                   help="Assume script is RTL; stored as reading_dir and influences reverse default.")
    a.add_argument("--no-prefer-rtl", dest="prefer_rtl", action="store_false")
    a.add_argument("--stats-out", default=None, help="Write stats JSON to this path")
    a.add_argument("--write-debug", action="store_true", help="Write per-image debug JSONs")
    a.add_argument("--debug-dir", default=None, help="Directory for debug JSONs")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.cmd == "analyze":
        recs = _read_jsonl(args.manifest)
        updated, stats = analyze_manifest(
            records=recs,
            mode=args.mode,
            engine_name=args.engine,
            tesseract_lang=args.tesseract_lang,
            tesseract_psm=args.tesseract_psm,
            kraken_model_path=args.kraken_model_path,
            kraken_model_id=args.kraken_model_id,
            cache_models=args.cache_models,
            kraken_device=args.kraken_device,
            kraken_threshold=bool(args.kraken_threshold),
            max_items=args.max_items,
            seed=args.seed,
            prefer_rtl=bool(args.prefer_rtl),
            write_debug=bool(args.write_debug),
            debug_dir=args.debug_dir,
        )
        _write_jsonl(updated, args.out)
        if args.stats_out:
            with open(args.stats_out, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
