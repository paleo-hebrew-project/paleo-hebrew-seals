"""model_registry
===============

Practical model registry / downloader layer.

Problem
-------
You need a stable way to:
- reference OCR/detection/classification models by identifier
- download/cache them (Zenodo, HuggingFace Hub, direct URLs)
- pin versions/checksums
- provide a single interface to baseline scripts and training pipelines

This module provides:
- ModelSpec dataclass
- registry dict with built-in specs
- functions:
    resolve_model(id) -> local path
    ensure_model(id, cache_dir) -> download if missing

Supported sources
-----------------
- direct URL (http/https)
- HuggingFace Hub (repo_id + filename)
- Zenodo record/file (record_id + filename)

Notes
-----
- We do not hardcode any niche paleo-hebrew model URLs.
  Instead we provide a *place to put them* and helper functions.
- For Kraken:
    - Kraken models are typically .mlmodel or .pyrnn.gz depending on version.
    - You can pin a specific Zenodo record that hosts a kraken model.
    - In your baseline, you can call ensure_model("kraken:hebrew-printed:v1")
      and get a path.

Integration points
------------------
- paleo_ocr_baseline.py can accept --model-id for each engine and call ensure_model
- training scripts can accept --init-from model id

"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModelSpec:
    id: str
    kind: str  # ocr, det, cls, lm, other
    source: str  # url, hf, zenodo, local

    # URL
    url: Optional[str] = None

    # HF
    hf_repo: Optional[str] = None
    hf_filename: Optional[str] = None
    hf_revision: Optional[str] = None

    # Zenodo
    zenodo_record: Optional[str] = None
    zenodo_filename: Optional[str] = None

    # Local
    local_path: Optional[str] = None

    # Pinning
    sha256: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


# -------------------------
# Registry
# -------------------------

REGISTRY: Dict[str, ModelSpec] = {
    # Examples (placeholders). Add real ones as you find them.

    # Kraken: you should replace with a real Zenodo record once chosen.
    "kraken:hebrew-printed:v1": ModelSpec(
        id="kraken:hebrew-printed:v1",
        kind="ocr",
        source="zenodo",
        zenodo_record="0000000",  # TODO
        zenodo_filename="hebrew-printed.mlmodel",  # TODO
        sha256=None,
        extra={"engine": "kraken", "script": "hebrew"},
    ),

    # TrOCR models are usually from HF; example only.
    "hf:trocr-printed-base": ModelSpec(
        id="hf:trocr-printed-base",
        kind="ocr",
        source="hf",
        hf_repo="microsoft/trocr-base-printed",
        hf_filename=None,
        hf_revision=None,
        extra={"engine": "trocr"},
    ),

    # YOLO detector checkpoint you trained locally
    "local:yolo:paleo_glyph": ModelSpec(
        id="local:yolo:paleo_glyph",
        kind="det",
        source="local",
        local_path="runs/detect/paleo_glyph/weights/best.pt",
        extra={"engine": "ultralytics"},
    ),
}


# -------------------------
# Download helpers
# -------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_url(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dst)


def zenodo_download_url(record_id: str, filename: str) -> str:
    # Zenodo files are accessible via /records/<id>/files/<filename>
    # Works for public records.
    return f"https://zenodo.org/records/{record_id}/files/{filename}"


def ensure_hf(repo: str, filename: Optional[str], revision: Optional[str], cache_dir: Path) -> Path:
    """Download from HF Hub into cache.

    If filename is None, we rely on transformers/diffusers to download.
    Here we only support file downloads via huggingface_hub.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError("Need huggingface_hub. pip install huggingface_hub") from e

    if filename is None:
        # Store a marker: caller should treat repo id as model reference
        marker = cache_dir / (repo.replace("/", "__") + ".hf_repo")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"repo": repo, "revision": revision}), encoding="utf-8")
        return marker

    path = hf_hub_download(repo_id=repo, filename=filename, revision=revision, cache_dir=str(cache_dir))
    return Path(path)


# -------------------------
# Public API
# -------------------------


def resolve_spec(model_id: str) -> ModelSpec:
    if model_id not in REGISTRY:
        raise KeyError(f"Unknown model id: {model_id}. Add it to REGISTRY or provide a local path.")
    return REGISTRY[model_id]


def ensure_model(model_id: str, cache_dir: str = "models_cache") -> Path:
    """Ensure model is available locally and return its path.

    cache_dir layout:
      cache_dir/<sanitized_model_id>/...
    """

    spec = resolve_spec(model_id)
    cache = Path(cache_dir)
    sub = cache / re.sub(r"[^a-zA-Z0-9._-]+", "_", spec.id)
    sub.mkdir(parents=True, exist_ok=True)

    if spec.source == "local":
        if not spec.local_path:
            raise ValueError(f"local model spec missing local_path: {model_id}")
        p = Path(spec.local_path)
        if not p.exists():
            raise FileNotFoundError(f"Local model path does not exist: {p}")
        return p.resolve()

    if spec.source == "url":
        if not spec.url:
            raise ValueError(f"url model spec missing url: {model_id}")
        fname = Path(spec.url).name
        dst = sub / fname
        if not dst.exists():
            download_url(spec.url, dst)
        if spec.sha256:
            got = sha256_file(dst)
            if got.lower() != spec.sha256.lower():
                raise RuntimeError(f"SHA256 mismatch for {model_id}: got {got} expected {spec.sha256}")
        return dst

    if spec.source == "zenodo":
        if not spec.zenodo_record or not spec.zenodo_filename:
            raise ValueError(f"zenodo spec missing record/filename: {model_id}")
        url = zenodo_download_url(spec.zenodo_record, spec.zenodo_filename)
        dst = sub / spec.zenodo_filename
        if not dst.exists():
            download_url(url, dst)
        if spec.sha256:
            got = sha256_file(dst)
            if got.lower() != spec.sha256.lower():
                raise RuntimeError(f"SHA256 mismatch for {model_id}: got {got} expected {spec.sha256}")
        return dst

    if spec.source == "hf":
        if not spec.hf_repo:
            raise ValueError(f"hf spec missing repo: {model_id}")
        p = ensure_hf(spec.hf_repo, spec.hf_filename, spec.hf_revision, cache)
        return p

    raise ValueError(f"Unknown source type: {spec.source}")


def register(spec: ModelSpec) -> None:
    """Register a new model spec at runtime."""
    REGISTRY[spec.id] = spec


def list_models(kind: Optional[str] = None) -> Dict[str, ModelSpec]:
    if kind is None:
        return dict(REGISTRY)
    return {k: v for k, v in REGISTRY.items() if v.kind == kind}


if __name__ == "__main__":
    # Simple CLI: list or ensure
    import argparse

    p = argparse.ArgumentParser(description="Model registry helper")
    p.add_argument("--list", action="store_true")
    p.add_argument("--kind", type=str, default=None)
    p.add_argument("--ensure", type=str, default=None)
    p.add_argument("--cache", type=str, default="models_cache")
    args = p.parse_args()

    if args.list:
        ms = list_models(args.kind)
        for k, v in ms.items():
            print(k, "->", v.source)

    if args.ensure:
        path = ensure_model(args.ensure, cache_dir=args.cache)
        print(str(path))
