"""YAML/CLI detector training: YOLO export + Ultralytics with multi-source regimes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from paleo_ocr.experiments.dataset_builders import (
    build_or_reuse_yolo_dataset_from_manifests,
    effective_train_manifests_weighted,
)
from paleo_ocr.experiments.run_utils import set_global_seed, write_run_metadata
from paleo_ocr.experiments.schema import ExperimentConfig, load_experiment_config
from paleo_ocr.train_detect import resolve_ultralytics_run_dir, train_ultralytics


def _last_ultralytics_metrics(run_dir: Path) -> Dict[str, Any]:
    """Read the last row from Ultralytics results.csv and normalize key metrics."""
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        return {}

    try:
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}

    if not rows:
        return {}

    last = rows[-1]

    def _to_float(key: str) -> Optional[float]:
        raw = last.get(key)
        if raw is None or str(raw).strip() == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    return {
        "mAP50": _to_float("metrics/mAP50(B)"),
        "mAP50_95": _to_float("metrics/mAP50-95(B)"),
        "precision": _to_float("metrics/precision(B)"),
        "recall": _to_float("metrics/recall(B)"),
        "epoch": _to_float("epoch"),
        "time": _to_float("time"),
    }


def _write_detector_metrics(out_dir: Path, run_dir: Path) -> Dict[str, Any]:
    """Write metrics_detector.json from an Ultralytics run directory."""
    metrics = _last_ultralytics_metrics(run_dir)
    payload: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "weights": str(run_dir / "weights" / "best.pt"),
    }
    payload.update({k: v for k, v in metrics.items() if v is not None})
    (out_dir / "metrics_detector.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def _resolve_source_path(cfg: ExperimentConfig, source_id: str) -> Path:
    for s in cfg.data.sources:
        if s.id == source_id:
            return Path(s.path)
    raise KeyError(f"Unknown source id {source_id!r}. Available: {[x.id for x in cfg.data.sources]}")


def _shared_dataset_cache_root(cfg: ExperimentConfig) -> Optional[Path]:
    if not cfg.detector.shared_dataset_cache:
        return None
    if cfg.detector.shared_dataset_cache_root:
        return Path(cfg.detector.shared_dataset_cache_root).resolve()
    return (Path(cfg.output_root) / "_shared_yolo_cache").resolve()


def _link_dataset_root(link_path: Path, dataset_root: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        return
    try:
        link_path.symlink_to(dataset_root.resolve(), target_is_directory=True)
    except OSError:
        # Best-effort convenience link only; training uses the real data.yaml path directly.
        return


def _write_dataset_refs(yolo_root: Path, refs: Dict[str, str]) -> None:
    yolo_root.mkdir(parents=True, exist_ok=True)
    (yolo_root / "dataset_refs.json").write_text(
        json.dumps(refs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_train_val_manifests(cfg: ExperimentConfig) -> Tuple[List[Path], List[Path]]:
    """
    Returns (train_manifests, val_manifests) based on mixing.mode and detector.regime.
    """
    d = cfg.data
    m = cfg.mixing
    regime = cfg.detector.regime

    val_path = Path(d.val_manifest)
    if not val_path.exists():
        raise FileNotFoundError(f"val_manifest not found: {val_path}")

    if m.mode == "weighted" and m.weights:
        weighted: List[Tuple[Path, int]] = []
        for sid, w in m.weights.items():
            weighted.append((_resolve_source_path(cfg, sid), max(1, int(round(float(w) * 10)))))
        train = effective_train_manifests_weighted(weighted)
        return train, [val_path]

    train: List[Path] = []

    if regime == "real_only":
        train.append(_resolve_source_path(cfg, d.real_source_id))
    elif regime == "synth_stage_a_only":
        train.append(_resolve_source_path(cfg, d.synth_stage_a_id))
    elif regime == "synth_stage_b_only":
        train.append(_resolve_source_path(cfg, d.synth_stage_b_id))
    elif regime in ("synth_all", "stage_a_plus_b"):
        train.append(_resolve_source_path(cfg, d.synth_stage_a_id))
        train.append(_resolve_source_path(cfg, d.synth_stage_b_id))
    elif regime == "synth_train_real_val":
        # notebook default: train = stage B synth, val = real manifest
        train.append(_resolve_source_path(cfg, d.synth_stage_b_id))
    elif regime == "custom":
        for sid in m.weights.keys():
            train.append(_resolve_source_path(cfg, sid))
    else:
        raise ValueError(f"Unknown detector.regime: {regime}")

    return train, [val_path]


def run_detector_experiment(
    cfg: ExperimentConfig,
    *,
    start_phase: int = 0,
    end_phase: Optional[int] = None,
    init_weights: Optional[str] = None,
) -> Dict[str, Any]:
    set_global_seed(cfg.seed)
    out = Path(cfg.output_root) / cfg.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    yolo_root = out / "yolo_dataset"
    shared_cache_root = _shared_dataset_cache_root(cfg)
    dataset_refs: Dict[str, str] = {}
    from dataclasses import asdict

    write_run_metadata(out, config_dict=asdict(cfg), seed=cfg.seed)

    project = str(Path(cfg.detector.project).resolve())
    name = cfg.detector.run_name
    train_overrides = dict(cfg.detector.train_overrides or {})

    val_path = Path(cfg.data.val_manifest)
    if not val_path.exists():
        raise FileNotFoundError(f"val_manifest not found: {val_path}")
    val_m = [val_path]

    if cfg.mixing.mode == "sequential" and cfg.phases:
        n_phases = len(cfg.phases)
        end_exclusive = n_phases if end_phase is None else int(end_phase)
        if end_exclusive < 0 or end_exclusive > n_phases:
            raise ValueError(
                f"end_phase (exclusive) must be in [0, {n_phases}], got {end_exclusive}"
            )
        if start_phase < 0 or start_phase >= n_phases:
            raise ValueError(
                f"start_phase must be in [0, {n_phases}) for this config, got {start_phase}"
            )
        if start_phase >= end_exclusive:
            raise ValueError(
                f"start_phase ({start_phase}) must be < end_phase exclusive ({end_exclusive})"
            )

        phase_logs: List[Dict[str, Any]] = []
        if start_phase > 0:
            log_path = out / "detector_sequential_log.json"
            if log_path.is_file():
                try:
                    prev_data = json.loads(log_path.read_text(encoding="utf-8"))
                    prev_phases = prev_data.get("phases") or []
                    if len(prev_phases) >= start_phase:
                        phase_logs = list(prev_phases[:start_phase])
                except Exception:
                    pass

        weights_path: Optional[str] = None
        if start_phase > 0:
            if init_weights:
                iw = Path(init_weights).expanduser().resolve()
                if not iw.is_file():
                    raise FileNotFoundError(f"--detector-init-weights not found: {iw}")
                weights_path = str(iw)
            else:
                prev_idx = start_phase - 1
                prev_run_name = f"{name}_ph{prev_idx}"
                found = resolve_ultralytics_run_dir(project, prev_run_name)
                if found is None:
                    raise FileNotFoundError(
                        f"Cannot resume at phase {start_phase}: no Ultralytics run with weights found "
                        f"for previous phase (expected run name like {prev_run_name!r} under project {project!r}). "
                        f"Pass --detector-init-weights /path/to/best.pt from that phase."
                    )
                bw = found / "weights" / "best.pt"
                if not bw.is_file():
                    raise FileNotFoundError(f"Missing {bw}")
                weights_path = str(bw)

        for i, ph in enumerate(cfg.phases):
            if i < start_phase:
                continue
            if i >= end_exclusive:
                break
            tr = [_resolve_source_path(cfg, sid) for sid in ph.source_ids]
            local_phase_root = yolo_root / f"phase_{i}"
            data_yaml = build_or_reuse_yolo_dataset_from_manifests(
                tr,
                val_m,
                local_phase_root,
                single_class=cfg.detector.single_class,
                use_symlinks=cfg.detector.use_symlinks,
                train_split_filter=cfg.detector.train_split_filter,
                val_split_filter=cfg.detector.val_split_filter,
                shared_cache_root=shared_cache_root,
            )
            dataset_root = data_yaml.parent.resolve()
            dataset_refs[f"phase_{i}"] = str(dataset_root)
            _link_dataset_root(local_phase_root, dataset_root)
            init = cfg.detector.model if weights_path is None else weights_path
            if i > 0 and isinstance(init, str):
                ip = Path(init).expanduser()
                if ip.suffix.lower() in (".pt", ".pth") and not ip.is_file():
                    found_prev = resolve_ultralytics_run_dir(
                        project, f"{name}_ph{i - 1}"
                    )
                    if found_prev is not None:
                        bw = found_prev / "weights" / "best.pt"
                        if bw.is_file():
                            init = str(bw.resolve())
                ip = Path(init).expanduser()
                if ip.suffix.lower() in (".pt", ".pth") and not ip.is_file():
                    raise FileNotFoundError(
                        f"Sequential phase {i}: checkpoint not found at {init!r}. "
                        f"Expected a previous-phase run named like {name}_ph{i - 1!r} under project {project!r}, "
                        f"or pass --detector-init-weights."
                    )
            merged_overrides = dict(train_overrides)
            if getattr(ph, "train_overrides", None):
                merged_overrides.update(ph.train_overrides)
            run_name = f"{name}_ph{i}"
            run_dir = train_ultralytics(
                data_yaml=data_yaml,
                model=init,
                epochs=ph.epochs,
                imgsz=cfg.detector.imgsz,
                batch=cfg.detector.batch,
                device=cfg.detector.device,
                project=project,
                name=run_name,
                workers=cfg.detector.workers,
                lr0=ph.lr or cfg.detector.lr0,
                train_overrides=merged_overrides,
            )
            best_weights = run_dir / "weights" / "best.pt"
            weights_path = str(best_weights.resolve())
            phase_logs.append(
                {
                    "phase_idx": i,
                    "phase_name": ph.name,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "source_ids": list(ph.source_ids),
                    "train_manifests": [str(p) for p in tr],
                    "dataset_root": str(dataset_root),
                    "data_yaml": str(data_yaml),
                    "init_model_or_weights": str(init),
                    "best_weights": str(best_weights),
                    "epochs": int(ph.epochs),
                    "lr0": ph.lr if ph.lr is not None else cfg.detector.lr0,
                    "train_overrides": merged_overrides,
                    "metrics_last": _last_ultralytics_metrics(run_dir),
                }
            )

        seq_payload = {
            "mode": "sequential",
            "project": project,
            "base_run_name": name,
            "shared_dataset_cache_root": str(shared_cache_root) if shared_cache_root else None,
            "dataset_refs": dataset_refs,
            "last_weights": weights_path,
            "phases": phase_logs,
            "resume": {
                "start_phase": start_phase,
                "end_phase": end_phase,
                "init_weights": str(Path(init_weights).resolve()) if init_weights else None,
            },
        }
        (out / "detector_sequential_log.json").write_text(
            json.dumps(seq_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_dataset_refs(yolo_root, dataset_refs)
        if phase_logs:
            _write_detector_metrics(out, Path(phase_logs[-1]["run_dir"]))
        return {
            "yolo_dataset": str(yolo_root),
            "shared_dataset_cache_root": str(shared_cache_root) if shared_cache_root else None,
            "last_weights": weights_path,
            "sequential_log": str(out / "detector_sequential_log.json"),
        }

    if cfg.mixing.mode == "alternating":
        # Epoch-level alternation: multiple short train_ultralytics runs
        real_tr = _resolve_source_path(cfg, cfg.data.real_source_id)
        synth_tr = _resolve_source_path(cfg, cfg.data.synth_stage_b_id)
        weights_path: Optional[str] = None
        cycle = 0
        total_cycles = max(1, cfg.detector.epochs // max(1, cfg.detector.alternating_real_epochs + cfg.detector.alternating_synth_epochs))
        for _ in range(total_cycles):
            for tag, manifest, ep in (
                ("real", real_tr, cfg.detector.alternating_real_epochs),
                ("synth", synth_tr, cfg.detector.alternating_synth_epochs),
            ):
                if ep <= 0:
                    continue
                local_alt_root = yolo_root / f"alt_{tag}"
                data_yaml = build_or_reuse_yolo_dataset_from_manifests(
                    [manifest],
                    val_m,
                    local_alt_root,
                    single_class=cfg.detector.single_class,
                    use_symlinks=cfg.detector.use_symlinks,
                    train_split_filter=cfg.detector.train_split_filter,
                    val_split_filter=cfg.detector.val_split_filter,
                    shared_cache_root=shared_cache_root,
                )
                dataset_refs[f"alt_{tag}"] = str(data_yaml.parent.resolve())
                _link_dataset_root(local_alt_root, data_yaml.parent.resolve())
                init = cfg.detector.model if weights_path is None else weights_path
                run_dir = train_ultralytics(
                    data_yaml=data_yaml,
                    model=init,
                    epochs=ep,
                    imgsz=cfg.detector.imgsz,
                    batch=cfg.detector.batch,
                    device=cfg.detector.device,
                    project=project,
                    name=f"{name}_alt_{cycle}_{tag}",
                    workers=cfg.detector.workers,
                    lr0=cfg.detector.lr0,
                    train_overrides=train_overrides,
                )
                weights_path = str(run_dir / "weights" / "best.pt")
                cycle += 1
        _write_dataset_refs(yolo_root, dataset_refs)
        if weights_path:
            _write_detector_metrics(out, Path(weights_path).parent)
        return {
            "yolo_dataset": str(yolo_root),
            "shared_dataset_cache_root": str(shared_cache_root) if shared_cache_root else None,
            "last_weights": weights_path,
        }

    train_m, val_m = resolve_train_val_manifests(cfg)
    data_yaml = build_or_reuse_yolo_dataset_from_manifests(
        train_m,
        val_m,
        yolo_root,
        single_class=cfg.detector.single_class,
        use_symlinks=cfg.detector.use_symlinks,
        train_split_filter=cfg.detector.train_split_filter,
        val_split_filter=cfg.detector.val_split_filter,
        shared_cache_root=shared_cache_root,
    )
    dataset_root = data_yaml.parent.resolve()
    dataset_refs["main"] = str(dataset_root)
    _link_dataset_root(yolo_root / "main", dataset_root)
    _write_dataset_refs(yolo_root, dataset_refs)

    run_dir = train_ultralytics(
        data_yaml=data_yaml,
        model=cfg.detector.model,
        epochs=cfg.detector.epochs,
        imgsz=cfg.detector.imgsz,
        batch=cfg.detector.batch,
        device=cfg.detector.device,
        project=project,
        name=name,
        workers=cfg.detector.workers,
        lr0=cfg.detector.lr0,
        train_overrides=train_overrides,
    )
    _write_detector_metrics(out, run_dir)
    return {
        "yolo_dataset": str(yolo_root),
        "data_yaml": str(data_yaml),
        "shared_dataset_cache_root": str(shared_cache_root) if shared_cache_root else None,
        "weights": str(run_dir / "weights" / "best.pt"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detector experiment (YOLO)")
    p.add_argument("--config", type=str, required=True, help="YAML experiment config")
    p.add_argument(
        "--detector-start-phase",
        type=int,
        default=0,
        help="Sequential mode only: skip phases 0..N-1 and train from phase N (requires prior weights on disk or --detector-init-weights).",
    )
    p.add_argument(
        "--detector-init-weights",
        type=str,
        default=None,
        help="Sequential mode only: explicit best.pt when resuming with --detector-start-phase>0.",
    )
    p.add_argument(
        "--detector-end-phase",
        type=int,
        default=None,
        help="Sequential mode only: exclusive stop index (run phases start_phase .. end_phase-1). "
        "Example: --detector-start-phase 0 --detector-end-phase 1 runs only phase 0.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    if cfg.task != "detector":
        raise SystemExit("config task must be detector")
    out = run_detector_experiment(
        cfg,
        start_phase=int(args.detector_start_phase or 0),
        end_phase=args.detector_end_phase,
        init_weights=args.detector_init_weights,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
