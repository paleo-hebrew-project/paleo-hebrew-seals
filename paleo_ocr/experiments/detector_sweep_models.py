"""Detector backbone lists for parallel sweeps (Ultralytics YOLO / RT-DETR).

These tuples mirror ``scripts/run_parallel_detector_sweep.sh`` and the Phase 1 table in
``MODEL_MATRIX.md``. Use with ``python -m paleo_ocr.experiments.run_experiment --config …
--detector-model <name>``.

If Ultralytics in your environment does not ship a checkpoint (e.g. some ``yolo26*`` builds),
override the sweep in the shell via ``DETECTOR_MODELS`` or run a subset from Python.
"""

from __future__ import annotations

from typing import Literal, Tuple

DetectorSweepProfile = Literal["paper", "full"]

# Subset for quick runs / compact tables.
DETECTOR_MODELS_PAPER: Tuple[str, ...] = (
    "yolov8n.pt",
    "yolo11m.pt",
    "rtdetr-l.pt",
)

# Full backbone grid (YOLOv8 / YOLO11 / YOLO26 / RT-DETR).
DETECTOR_MODELS_FULL: Tuple[str, ...] = (
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
    "rtdetr-l.pt",
    "rtdetr-x.pt",
)

# Optional extra standalone YAMLs (e.g. Stage A–only control), relative to repo root.
DETECTOR_EXTRA_DEFAULT_CONFIG_RELPATHS: Tuple[str, ...] = (
    "configs/experiments/detector_stage_a_only_valreal.yaml",
)


def detector_models_for_profile(profile: DetectorSweepProfile) -> Tuple[str, ...]:
    if profile == "paper":
        return DETECTOR_MODELS_PAPER
    if profile == "full":
        return DETECTOR_MODELS_FULL
    raise ValueError(f"Unknown detector sweep profile: {profile!r} (use paper or full)")
