"""Classifier experiment driver: crop export + timm training (single / weighted / alternating / curriculum / sequential)."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paleo_ocr.experiments.classify_core import (
    ClassifyTrainConfig,
    _build_transforms,
    build_train_loader_single,
    build_train_loader_weighted,
    train_alternating_epochs,
    train_curriculum_weighted,
    train_one_phase,
)
from paleo_ocr.experiments.dataset_builders import export_classifier_imagefolder
from paleo_ocr.experiments.run_utils import set_global_seed, write_run_metadata
from paleo_ocr.experiments.schema import ExperimentConfig, load_experiment_config
from torchvision import datasets
from torch.utils.data import DataLoader


def _phase_dir_slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (name or "phase").strip())
    return s[:80] if len(s) > 80 else s


def _classify_train_config(
    cfg: ExperimentConfig,
    *,
    data_dir: Path,
    out_dir: Path,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    freeze_backbone_epochs: Optional[int] = None,
    warm_start_ckpt: Optional[Path] = None,
) -> ClassifyTrainConfig:
    c = cfg.classifier
    return ClassifyTrainConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        model=c.model,
        epochs=int(epochs if epochs is not None else c.epochs),
        batch=c.batch,
        imgsz=c.imgsz,
        lr=float(lr if lr is not None else c.lr),
        wd=c.wd,
        num_workers=int(getattr(c, "num_workers", 8)),
        device="cuda",
        amp=c.amp,
        best_metric=c.best_metric,
        topk=c.topk,
        seed=cfg.seed,
        tensorboard=getattr(c, "tensorboard", False),
        transform_style=getattr(c, "transform_style", "legacy"),
        label_smoothing=float(getattr(c, "label_smoothing", 0.0)),
        warmup_epochs=float(getattr(c, "warmup_epochs", 0.0)),
        cosine_schedule=bool(getattr(c, "cosine_schedule", False)),
        freeze_backbone_epochs=int(
            freeze_backbone_epochs if freeze_backbone_epochs is not None else getattr(c, "freeze_backbone_epochs", 0)
        ),
        prefetch_factor=int(getattr(c, "prefetch_factor", 4)),
        drop_last=bool(getattr(c, "drop_last", True)),
        warm_start_ckpt=warm_start_ckpt,
    )


def _resolve_source(cfg: ExperimentConfig, sid: str) -> Path:
    for s in cfg.data.sources:
        if s.id == sid:
            return Path(s.path)
    raise KeyError(sid)


def _classifier_train_manifest_spec(cfg: ExperimentConfig, sid: str) -> Tuple[Path, str, Optional[str]]:
    split_filter = None
    if sid == cfg.data.real_source_id:
        split_filter = getattr(cfg.classifier, "train_split_filter", None)
    return (_resolve_source(cfg, sid), "train", split_filter)


def _classifier_val_manifest_spec(cfg: ExperimentConfig) -> Tuple[Path, str, Optional[str]]:
    return (
        Path(cfg.data.val_manifest),
        "val",
        getattr(cfg.classifier, "val_split_filter", None),
    )


def _assert_classifier_real_val_split_safety(cfg: ExperimentConfig, *, uses_real_train: bool) -> None:
    """
    Prevent silent train/val leakage when the same real manifest is reused for both
    train and val and classifier split filters are not configured.
    """
    if not uses_real_train:
        return
    real_path = _resolve_source(cfg, cfg.data.real_source_id).resolve()
    val_path = Path(cfg.data.val_manifest).resolve()
    if real_path != val_path:
        return

    train_filter = str(getattr(cfg.classifier, "train_split_filter", "") or "").strip()
    val_filter = str(getattr(cfg.classifier, "val_split_filter", "") or "").strip()
    if not train_filter or not val_filter:
        raise ValueError(
            "Classifier config uses the same manifest for real train and val, but "
            "classifier.train_split_filter / classifier.val_split_filter are not set. "
            "This would leak validation rows into train. Set, for example, "
            "train_split_filter: train and val_split_filter: validation, or use separate manifests."
        )
    if train_filter.lower() == val_filter.lower():
        raise ValueError(
            "classifier.train_split_filter and classifier.val_split_filter must differ when "
            "real train and val come from the same manifest."
        )


def _resolve_classifier_resume_ckpt(
    out: Path,
    cfg: ExperimentConfig,
    prev_idx: int,
    *,
    resume_weights: str = "auto",
) -> Optional[Path]:
    """
    Resolve previous-phase classifier weights.

    Preferred layout is the new sequential one:
      classifier_run/phase_XX_<name>/{best,last}.pt

    Fallback for older synth-only sweeps:
      classifier_run/{best,last}.pt
    """
    prev_name = cfg.phases[prev_idx].name
    if resume_weights not in {"auto", "best", "last"}:
        raise ValueError(f"Unknown resume_weights={resume_weights!r}; use 'auto', 'best', or 'last'")

    phase_dir = out / "classifier_run" / f"phase_{prev_idx:02d}_{_phase_dir_slug(prev_name)}"
    candidates = _classifier_resume_candidates(phase_dir, out / "classifier_run", resume_weights=resume_weights)
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def _classifier_resume_candidates(
    phase_dir: Path,
    legacy_run_dir: Path,
    *,
    resume_weights: str,
) -> List[Path]:
    if resume_weights not in {"auto", "best", "last"}:
        raise ValueError(f"Unknown resume_weights={resume_weights!r}; use 'auto', 'best', or 'last'")
    if resume_weights == "best":
        return [
            phase_dir / "best.pt",
            legacy_run_dir / "best.pt",
        ]
    if resume_weights == "last":
        return [
            phase_dir / "last.pt",
            legacy_run_dir / "last.pt",
        ]
    return [
        phase_dir / "best.pt",
        phase_dir / "last.pt",
        legacy_run_dir / "best.pt",
        legacy_run_dir / "last.pt",
    ]


def _classifier_next_phase_ckpt(phase_out: Path, *, resume_weights: str) -> Optional[Path]:
    for p in _classifier_resume_candidates(phase_out, phase_out, resume_weights=resume_weights):
        if p.is_file():
            return p.resolve()
    return None


def _manifest_pairs_for_regime(
    cfg: ExperimentConfig,
    regime_override: Optional[str] = None,
) -> Tuple[List[Tuple[Path, str, Optional[str]]], List[Tuple[Path, str, Optional[str]]]]:
    """Returns (train_pairs, val_pairs) as (manifest, split) lists."""
    d = cfg.data
    m = cfg.mixing
    ro = (regime_override or "").strip() or None

    train_pairs: List[Tuple[Path, str, Optional[str]]] = []
    if m.mode == "weighted" and m.weights:
        for sid in m.weights.keys():
            train_pairs.append(_classifier_train_manifest_spec(cfg, sid))
    elif ro == "real_only":
        train_pairs.append(_classifier_train_manifest_spec(cfg, d.real_source_id))
    elif ro in ("synth_a_only",):
        train_pairs.append(_classifier_train_manifest_spec(cfg, d.synth_stage_a_id))
    elif ro in ("synth_b_only", "synth_only"):
        train_pairs.append(_classifier_train_manifest_spec(cfg, d.synth_stage_b_id))
    elif ro in ("stage_a_plus_b", "synth_all"):
        train_pairs.append(_classifier_train_manifest_spec(cfg, d.synth_stage_a_id))
        train_pairs.append(_classifier_train_manifest_spec(cfg, d.synth_stage_b_id))
    else:
        train_pairs.append(_classifier_train_manifest_spec(cfg, d.synth_stage_b_id))

    val_pairs = [_classifier_val_manifest_spec(cfg)]
    return train_pairs, val_pairs


def _run_classifier_sequential(
    cfg: ExperimentConfig,
    out: Path,
    data_root: Path,
    *,
    start_phase: int = 0,
    end_phase: Optional[int] = None,
    init_weights: Optional[str] = None,
    resume_weights: str = "auto",
) -> Dict[str, Any]:
    """Train sequential classifier phases, with optional resume/start-phase support."""
    n_phases = len(cfg.phases)
    end_exclusive = n_phases if end_phase is None else int(end_phase)
    if end_exclusive < 0 or end_exclusive > n_phases:
        raise ValueError(f"end_phase (exclusive) must be in [0, {n_phases}], got {end_exclusive}")
    if start_phase < 0 or start_phase >= n_phases:
        raise ValueError(f"start_phase must be in [0, {n_phases}) for this config, got {start_phase}")
    if start_phase >= end_exclusive:
        raise ValueError(f"start_phase ({start_phase}) must be < end_phase exclusive ({end_exclusive})")

    classifier_run_root = out / "classifier_run"
    log_path = out / "classifier_sequential_log.json"
    phase_logs: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    prev_ckpt: Optional[Path] = None
    cls_cfg = cfg.classifier
    nw = int(getattr(cls_cfg, "num_workers", 8))
    phase_slice = cfg.phases[start_phase:end_exclusive]
    uses_real_train = any(cfg.data.real_source_id in ph.source_ids for ph in phase_slice)
    _assert_classifier_real_val_split_safety(cfg, uses_real_train=uses_real_train)

    if start_phase > 0 and log_path.is_file():
        try:
            prev_data = json.loads(log_path.read_text(encoding="utf-8"))
            prev_phases = prev_data.get("phases") or []
            if len(prev_phases) >= start_phase:
                phase_logs = list(prev_phases[:start_phase])
        except Exception:
            pass

    if start_phase > 0:
        if init_weights:
            prev_ckpt = Path(init_weights).expanduser().resolve()
            if not prev_ckpt.is_file():
                raise FileNotFoundError(f"--classifier-init-weights not found: {prev_ckpt}")
        else:
            prev_idx = start_phase - 1
            prev_ckpt = _resolve_classifier_resume_ckpt(
                out,
                cfg,
                prev_idx,
                resume_weights=resume_weights,
            )
            if prev_ckpt is None:
                raise FileNotFoundError(
                    f"Cannot resume at phase {start_phase}: missing previous phase checkpoint for "
                    f"{cfg.experiment_name!r}. Looked for resume_weights={resume_weights!r} in the "
                    "new sequential path and legacy classifier_run directory. Run the earlier phase "
                    "first, choose a different --classifier-resume-weights mode, or pass "
                    "--classifier-init-weights /path/to/checkpoint.pt."
                )

    for i, ph in enumerate(cfg.phases):
        if i < start_phase:
            continue
        if i >= end_exclusive:
            break
        phase_root = data_root / f"seq_{i:02d}_{_phase_dir_slug(ph.name)}"
        train_pairs = [_classifier_train_manifest_spec(cfg, sid) for sid in ph.source_ids]
        val_pairs = [_classifier_val_manifest_spec(cfg)]
        export_classifier_imagefolder(
            train_pairs + val_pairs,
            phase_root,
            pad_pct=0.12,
            make_square=True,
            out_size=None,
            max_workers=8,
        )
        phase_out = classifier_run_root / f"phase_{i:02d}_{_phase_dir_slug(ph.name)}"
        lr = float(ph.lr) if ph.lr is not None else cls_cfg.lr
        freeze_e = int(getattr(cls_cfg, "freeze_backbone_epochs", 0)) if i == 0 else 0
        tc = _classify_train_config(
            cfg,
            data_dir=phase_root,
            out_dir=phase_out,
            epochs=int(ph.epochs),
            lr=lr,
            freeze_backbone_epochs=freeze_e,
            warm_start_ckpt=prev_ckpt,
        )
        tfm_train, tfm_val = _build_transforms(tc.imgsz, tc.transform_style)
        val_ds = datasets.ImageFolder(str(phase_root / "val"), transform=tfm_val)
        val_loader = DataLoader(
            val_ds,
            batch_size=cls_cfg.batch,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
        )
        classes = val_ds.classes
        train_loader, classes_tr = build_train_loader_single(
            phase_root,
            tfm_train,
            cls_cfg.batch,
            nw,
            cfg.seed,
            drop_last=tc.drop_last,
            prefetch_factor=tc.prefetch_factor,
        )
        if classes_tr != classes:
            raise RuntimeError("train/val class mismatch in sequential phase")
        summary = train_one_phase(tc, train_loader=train_loader, val_loader=val_loader, classes=classes)
        best_ckpt = phase_out / "best.pt"
        last_ckpt = phase_out / "last.pt"
        prev_ckpt = _classifier_next_phase_ckpt(phase_out, resume_weights=resume_weights)
        if prev_ckpt is None:
            raise FileNotFoundError(
                f"Phase {i} completed but no checkpoint matched resume_weights={resume_weights!r} in {phase_out}."
            )
        phase_payload = {
            "phase_idx": i,
            "phase_name": ph.name,
            "source_ids": list(ph.source_ids),
            "train_manifests": [str(_resolve_source(cfg, sid)) for sid in ph.source_ids],
            "dataset_root": str(phase_root),
            "phase_out_dir": str(phase_out),
            "init_weights": str(tc.warm_start_ckpt) if tc.warm_start_ckpt else None,
            "best_weights": str(best_ckpt.resolve()) if best_ckpt.is_file() else None,
            "last_weights": str(last_ckpt.resolve()) if last_ckpt.is_file() else None,
            "resume_weights": resume_weights,
            "next_phase_init_weights": str(prev_ckpt),
            "epochs": int(ph.epochs),
            "lr": lr,
            **summary,
        }
        phase_logs.append(phase_payload)
        summaries.append(phase_payload)

    seq_payload = {
        "mode": "sequential",
        "last_best_ckpt": str((classifier_run_root / f"phase_{(end_exclusive - 1):02d}_{_phase_dir_slug(cfg.phases[end_exclusive - 1].name)}" / "best.pt").resolve())
        if end_exclusive > 0 and (classifier_run_root / f"phase_{(end_exclusive - 1):02d}_{_phase_dir_slug(cfg.phases[end_exclusive - 1].name)}" / "best.pt").is_file()
        else None,
        "last_resume_ckpt": str(prev_ckpt.resolve()) if prev_ckpt and prev_ckpt.is_file() else None,
        "phases": phase_logs,
        "resume": {
            "start_phase": start_phase,
            "end_phase": end_phase,
            "init_weights": str(Path(init_weights).expanduser().resolve()) if init_weights else None,
            "resume_weights": resume_weights,
        },
    }
    log_path.write_text(json.dumps(seq_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "sequential": True,
        "phase_summaries": summaries,
        "last_best_ckpt": str(prev_ckpt.resolve()) if prev_ckpt and prev_ckpt.is_file() else None,
        "classifier_sequential_log": str(log_path),
    }


def run_classifier_experiment(
    cfg: ExperimentConfig,
    regime_override: Optional[str] = None,
    start_phase: int = 0,
    end_phase: Optional[int] = None,
    init_weights: Optional[str] = None,
    resume_weights: str = "auto",
) -> Dict[str, Any]:
    set_global_seed(cfg.seed)
    out = Path(cfg.output_root) / cfg.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out, config_dict=asdict(cfg), seed=cfg.seed)

    cls_cfg = cfg.classifier
    data_root = out / "cls_imagefolder"
    m = cfg.mixing

    if m.mode == "sequential":
        if not cfg.phases:
            raise ValueError("mixing.mode: sequential requires a non-empty top-level `phases` list in YAML")
        return _run_classifier_sequential(
            cfg,
            out,
            data_root,
            start_phase=start_phase,
            end_phase=end_phase,
            init_weights=init_weights,
            resume_weights=resume_weights,
        )

    train_pairs, val_pairs = _manifest_pairs_for_regime(cfg, regime_override)
    uses_real_train = any(Path(mp).resolve() == _resolve_source(cfg, cfg.data.real_source_id).resolve() for mp, _sp, _flt in train_pairs)
    _assert_classifier_real_val_split_safety(cfg, uses_real_train=uses_real_train)
    export_classifier_imagefolder(
        train_pairs + val_pairs,
        data_root,
        pad_pct=0.12,
        make_square=True,
        out_size=None,
        max_workers=8,
    )

    tc = _classify_train_config(cfg, data_dir=data_root, out_dir=out / "classifier_run")
    nw = tc.num_workers

    tfm_train, tfm_val = _build_transforms(tc.imgsz, tc.transform_style)
    val_ds = datasets.ImageFolder(str(data_root / "val"), transform=tfm_val)
    val_loader = DataLoader(val_ds, batch_size=cls_cfg.batch, shuffle=False, num_workers=nw, pin_memory=True)
    classes = val_ds.classes

    if m.mode == "alternating":
        _assert_classifier_real_val_split_safety(cfg, uses_real_train=True)
        real_dir = data_root / "train_real_only_tmp"
        synth_dir = data_root / "train_synth_only_tmp"
        # Re-export two train folders: real-only and synth-only
        export_classifier_imagefolder(
            [_classifier_train_manifest_spec(cfg, cfg.data.real_source_id)],
            real_dir,
        )
        export_classifier_imagefolder(
            [_classifier_train_manifest_spec(cfg, cfg.data.synth_stage_b_id)],
            synth_dir,
        )
        tc.out_dir = out / "classifier_run"
        summary = train_alternating_epochs(
            tc,
            real_train_dir=real_dir / "train",
            synth_train_dir=synth_dir / "train",
            val_dir=data_root / "val",
            k_synth_batches=max(1, m.k_synth_batches),
        )
        return summary

    if m.mode == "curriculum":
        _assert_classifier_real_val_split_safety(cfg, uses_real_train=True)
        real_dir = data_root / "train_real_only_tmp"
        synth_dir = data_root / "train_synth_only_tmp"
        export_classifier_imagefolder(
            [_classifier_train_manifest_spec(cfg, cfg.data.real_source_id)],
            real_dir,
        )
        export_classifier_imagefolder(
            [_classifier_train_manifest_spec(cfg, cfg.data.synth_stage_b_id)],
            synth_dir,
        )
        tc.out_dir = out / "classifier_run"
        return train_curriculum_weighted(
            tc,
            real_train_dir=real_dir / "train",
            synth_train_dir=synth_dir / "train",
            val_loader=val_loader,
            classes=classes,
            synth_frac_start=m.synth_frac_start,
            synth_frac_end=m.synth_frac_end,
        )

    if m.mode == "weighted" and m.weights:
        _assert_classifier_real_val_split_safety(cfg, uses_real_train=(cfg.data.real_source_id in m.weights))
        roots: List[Tuple[Path, str, float]] = []
        for sid, wt in m.weights.items():
            mp = _resolve_source(cfg, sid)
            # use per-manifest subexport
            sub = data_root / f"weighted_train_{sid}"
            export_classifier_imagefolder([_classifier_train_manifest_spec(cfg, sid)], sub)
            roots.append((sub / "train", sid, float(wt)))
        train_loader, classes_tr = build_train_loader_weighted(
            roots,
            tfm_train,
            cls_cfg.batch,
            nw,
            cfg.seed,
        )
        if set(classes_tr) != set(val_ds.classes):
            raise RuntimeError("class mismatch between sources")
        summary = train_one_phase(tc, train_loader=train_loader, val_loader=val_loader, classes=val_ds.classes)
        return summary

    train_loader, classes_tr = build_train_loader_single(
        data_root,
        tfm_train,
        cls_cfg.batch,
        nw,
        cfg.seed,
        drop_last=tc.drop_last,
        prefetch_factor=tc.prefetch_factor,
    )
    if classes_tr != classes:
        raise RuntimeError("train/val class mismatch")
    summary = train_one_phase(tc, train_loader=train_loader, val_loader=val_loader, classes=classes)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classifier experiment")
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--regime",
        type=str,
        default=None,
        help="Optional override: real_only, synth_a_only, synth_b_only, stage_a_plus_b",
    )
    p.add_argument(
        "--start-phase",
        type=int,
        default=0,
        help="Sequential classifier only: skip completed phases [0..N-1] and train from phase N (0-based).",
    )
    p.add_argument(
        "--init-weights",
        type=str,
        default=None,
        help="Sequential classifier only: explicit checkpoint when resuming (overrides auto-discovery of previous phase).",
    )
    p.add_argument(
        "--resume-weights",
        type=str,
        choices=("auto", "best", "last"),
        default="auto",
        help="Sequential classifier only: when auto-resolving phase-N init weights, choose best, last, or auto.",
    )
    p.add_argument(
        "--end-phase",
        type=int,
        default=None,
        help="Sequential classifier only: exclusive upper bound on phase index (run start_phase .. end_phase-1).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    if cfg.task != "classifier":
        raise SystemExit("config task must be classifier")
    out = run_classifier_experiment(
        cfg,
        regime_override=args.regime,
        start_phase=int(args.start_phase or 0),
        end_phase=args.end_phase,
        init_weights=args.init_weights,
        resume_weights=args.resume_weights,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
