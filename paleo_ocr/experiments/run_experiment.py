"""Dispatch detector or classifier experiment from a YAML config."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

from paleo_ocr.experiments.classifier_train import run_classifier_experiment
from paleo_ocr.experiments.detector_train import run_detector_experiment
from paleo_ocr.experiments.schema import ExperimentConfig, load_experiment_config


def apply_detector_phase1_patience(cfg: ExperimentConfig, raw: str) -> None:
    """Set ``phases[1].train_overrides['patience']`` for sequential detector configs (finetune / phase 1)."""
    t = (raw or "").strip()
    if not t:
        return
    low = t.lower()
    if low in ("yaml", "default", "file"):
        return
    if low in ("off", "none", "disable", "false", "no"):
        patience = 0
    else:
        patience = int(t)
        if patience < 0:
            raise ValueError(f"--detector-phase1-patience must be >= 0, got {patience}")
    _merge_detector_phase1_train_override(cfg, "patience", patience)


def apply_detector_phase1_close_mosaic(cfg: ExperimentConfig, raw: str) -> None:
    """Set ``phases[1].train_overrides['close_mosaic']`` (Ultralytics: last N epochs without mosaic; 0 = never close)."""
    t = (raw or "").strip()
    if not t:
        return
    low = t.lower()
    if low in ("yaml", "default", "file"):
        return
    if low in ("off", "none", "disable", "false", "no"):
        n = 0
    else:
        n = int(t)
        if n < 0:
            raise ValueError(f"--detector-phase1-close-mosaic must be >= 0, got {n}")
    _merge_detector_phase1_train_override(cfg, "close_mosaic", n)


def _merge_detector_phase1_train_override(cfg: ExperimentConfig, key: str, value: Any) -> None:
    phases = cfg.phases
    if len(phases) < 2:
        raise ValueError(
            f"--detector-phase1-* overrides require at least two `phases` entries (phase 1 = finetune); "
            f"cannot set {key!r}."
        )
    ph1 = phases[1]
    merged = dict(ph1.train_overrides or {})
    merged[key] = value
    ph1.train_overrides = merged


def apply_classifier_phase1_epochs(cfg: ExperimentConfig, raw: str) -> None:
    """Set ``phases[1].epochs`` for sequential classifier configs (real finetune / phase 1)."""
    t = (raw or "").strip()
    if not t:
        return
    if t.lower() in ("yaml", "default", "file"):
        return
    epochs = int(t)
    if epochs <= 0:
        raise ValueError(f"--classifier-phase1-epochs must be > 0, got {epochs}")
    if len(cfg.phases) < 2:
        raise ValueError(
            "--classifier-phase1-epochs requires at least two `phases` entries "
            "(phase 1 = real finetune)."
        )
    cfg.phases[1].epochs = epochs
    cfg.classifier.epochs = sum(int(p.epochs) for p in cfg.phases)


def _model_slug(name: str) -> str:
    """Filesystem-safe fragment for experiment_name / run_name."""
    s = name.strip().replace("/", "_").replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_.+-]+", "_", s)
    return s[:120] if len(s) > 120 else s


def _classifier_phase1_epoch_slug(raw: str | None) -> str | None:
    t = (raw or "").strip()
    if not t or t.lower() in ("yaml", "default", "file"):
        return None
    epochs = int(t)
    if epochs <= 0:
        raise ValueError(f"--classifier-phase1-epochs must be > 0, got {epochs}")
    return f"real{epochs}e"


def apply_cli_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    """
    Optional backbone / run name overrides for sweep scripts.

    If --experiment-name is set, it wins. Otherwise, when a model override is
    given, experiment_name becomes ``{base}__{model_slug}``.
    """
    base_name = cfg.experiment_name
    explicit = bool(args.experiment_name)

    if args.experiment_name:
        cfg.experiment_name = args.experiment_name

    if cfg.task == "detector" and args.detector_model:
        cfg.detector.model = args.detector_model
        if not explicit:
            cfg.experiment_name = f"{base_name}__{_model_slug(args.detector_model)}"
        cfg.detector.run_name = cfg.experiment_name

    if cfg.task == "classifier" and args.classifier_model:
        cfg.classifier.model = args.classifier_model
        if not explicit:
            cfg.experiment_name = f"{base_name}__{_model_slug(args.classifier_model)}"

    if cfg.task == "classifier" and not explicit:
        phase1_slug = _classifier_phase1_epoch_slug(args.classifier_phase1_epochs)
        if phase1_slug:
            cfg.experiment_name = f"{cfg.experiment_name}__{phase1_slug}"

    if cfg.task == "detector" and args.detector_shared_cache_root:
        cfg.detector.shared_dataset_cache = True
        cfg.detector.shared_dataset_cache_root = args.detector_shared_cache_root

    if cfg.task == "detector" and args.detector_batch is not None:
        cfg.detector.batch = int(args.detector_batch)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paleo OCR experiment from YAML")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--regime", type=str, default=None, help="Classifier regime override")
    p.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment_name (output folder); else derived when using model overrides",
    )
    p.add_argument(
        "--detector-model",
        type=str,
        default=None,
        help="Override detector checkpoint name for Ultralytics (e.g. yolo11m.pt, rtdetr-l.pt)",
    )
    p.add_argument(
        "--classifier-model",
        type=str,
        default=None,
        help="Override timm classifier model name (e.g. convnext_base, efficientnet_b0)",
    )
    p.add_argument(
        "--classifier-start-phase",
        type=int,
        default=0,
        help="Sequential classifier only: skip completed phases [0..N-1] and train from phase N (0-based).",
    )
    p.add_argument(
        "--classifier-init-weights",
        type=str,
        default=None,
        help="Sequential classifier only: explicit checkpoint when resuming (overrides auto-discovery of previous phase).",
    )
    p.add_argument(
        "--classifier-resume-weights",
        type=str,
        choices=("auto", "best", "last"),
        default="auto",
        help="Sequential classifier only: when auto-resolving phase-N init weights, choose best, last, or auto.",
    )
    p.add_argument(
        "--classifier-end-phase",
        type=int,
        default=None,
        help="Sequential classifier only: exclusive upper bound on phase index (run start_phase .. end_phase-1). "
        "Example: 0 and end_phase 1 = only first phase (synth pretrain).",
    )
    p.add_argument(
        "--classifier-phase1-epochs",
        type=str,
        default=None,
        help="Sequential classifier: override phases[1].epochs (real finetune length). "
        "Use 20/60/120 for the paper regimes; omit or use 'yaml' to keep the config value.",
    )
    p.add_argument(
        "--detector-start-phase",
        type=int,
        default=0,
        help="Sequential detector only: skip completed phases [0..N-1] and train from phase N (0-based).",
    )
    p.add_argument(
        "--detector-init-weights",
        type=str,
        default=None,
        help="Sequential detector only: explicit best.pt when resuming (overrides auto-discovery of previous phase).",
    )
    p.add_argument(
        "--detector-end-phase",
        type=int,
        default=None,
        help="Sequential detector only: exclusive upper bound on phase index (run start_phase .. end_phase-1). "
        "Example: 0 and end_phase 1 = only first phase (synth pretrain).",
    )
    p.add_argument(
        "--detector-phase1-patience",
        type=str,
        default=None,
        help="Sequential detector: override phases[1].train_overrides.patience (Ultralytics early stopping). "
        "Integer = epochs without improvement; 0 or off/none/disable = turn off. "
        "Omit or use 'yaml' to keep the value from the config file.",
    )
    p.add_argument(
        "--detector-phase1-close-mosaic",
        type=str,
        default=None,
        help="Sequential detector: override phases[1].train_overrides.close_mosaic. "
        "Ultralytics disables mosaic for the last N epochs; 0 or off/none/disable = never (no 'Closing dataloader mosaic'). "
        "Omit or use 'yaml' for the config file value.",
    )
    p.add_argument(
        "--detector-shared-cache-root",
        type=str,
        default=None,
        help="Override detector.shared_dataset_cache_root (where the shared YOLO export lives). "
        "Point this at fast LOCAL storage (e.g. /tmp/paleo_yolo_cache) when the project lives on a slow FUSE/NFS share; "
        "symlinks still target the source images, but cache metadata ops become ~16x faster.",
    )
    p.add_argument(
        "--detector-batch",
        type=int,
        default=None,
        help="Override detector.batch (Ultralytics train batch size). Lower this for large models (e.g. yolov8x/yolo26x) "
        "that OOM at the YAML default on a single GPU.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    apply_cli_overrides(cfg, args)
    if cfg.task == "classifier" and args.classifier_phase1_epochs is not None:
        apply_classifier_phase1_epochs(cfg, args.classifier_phase1_epochs)
    if cfg.task == "detector" and args.detector_phase1_patience is not None:
        apply_detector_phase1_patience(cfg, args.detector_phase1_patience)
    if cfg.task == "detector" and args.detector_phase1_close_mosaic is not None:
        apply_detector_phase1_close_mosaic(cfg, args.detector_phase1_close_mosaic)
    cfg_path = Path(args.config).resolve()
    out = Path(cfg.output_root) / cfg.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out / "config_used.yaml")

    if cfg.task == "detector":
        run_detector_experiment(
            cfg,
            start_phase=int(args.detector_start_phase or 0),
            end_phase=args.detector_end_phase,
            init_weights=args.detector_init_weights,
        )
    elif cfg.task == "classifier":
        run_classifier_experiment(
            cfg,
            regime_override=args.regime,
            start_phase=int(args.classifier_start_phase or 0),
            end_phase=args.classifier_end_phase,
            init_weights=args.classifier_init_weights,
            resume_weights=args.classifier_resume_weights,
        )
    else:
        raise SystemExit(f"Unknown task {cfg.task}")


if __name__ == "__main__":
    main()
