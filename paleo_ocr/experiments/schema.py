"""YAML configuration schema for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class SourceConfig:
    """Logical data source id → manifest path."""

    id: str
    path: str


@dataclass
class DataConfig:
    sources: List[SourceConfig] = field(default_factory=list)
    val_manifest: str = ""
    # Maps regime to which source ids feed training (paths resolved via sources)
    real_source_id: str = "real"
    synth_stage_a_id: str = "synth_stage_a"
    synth_stage_b_id: str = "synth_stage_b"


@dataclass
class TrainPhaseConfig:
    name: str = "phase"
    source_ids: List[str] = field(default_factory=list)
    epochs: int = 10
    lr: Optional[float] = None
    # Merged over detector.train_overrides for this phase only (sequential mode).
    train_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MixingConfig:
    # single: one train source. sequential: classifier uses top-level `phases` (train order, warm-start ckpt);
    # detector uses phases + Ultralytics sequential logic separately.
    mode: str = "single"  # single | sequential | weighted | alternating | curriculum
    # weighted: per-source weights (relative)
    weights: Dict[str, float] = field(default_factory=dict)
    # alternating: after each `real` batch, take K synth batches (classifier); detector uses epoch_alternation
    k_synth_batches: int = 1
    # curriculum: linear decay from synth_frac_start to synth_frac_end over total_epochs
    synth_frac_start: float = 0.9
    synth_frac_end: float = 0.1
    curriculum_epochs: int = 0


@dataclass
class DetectorConfig:
    model: str = "yolov8n.pt"
    epochs: int = 100
    imgsz: int = 1024
    batch: int = 16
    device: str = "0"
    workers: int = 8
    project: str = "runs/detect"
    run_name: str = "exp"
    lr0: Optional[float] = None
    use_symlinks: bool = True
    # Reuse one shared YOLO export across experiments with identical manifests/filters.
    shared_dataset_cache: bool = True
    # Optional override for the shared YOLO export cache root.
    shared_dataset_cache_root: Optional[str] = None
    single_class: bool = True
    # Optional split filters for manifests that carry rec["split"] labels
    # (e.g. train / validation / test / extra). If None, use all rows.
    train_split_filter: Optional[str] = None
    val_split_filter: Optional[str] = None
    # Forwarded directly to ultralytics.YOLO.train(**kwargs), e.g.
    # {"optimizer":"AdamW","patience":20,"mosaic":0.4,...}
    train_overrides: Dict[str, Any] = field(default_factory=dict)
    # YOLO train regime
    regime: str = "synth_train_real_val"  # see detector_train.resolve_manifests
    alternating_real_epochs: int = 0
    alternating_synth_epochs: int = 0


@dataclass
class ClassifierConfig:
    model: str = "convnext_base"
    epochs: int = 30
    imgsz: int = 128
    batch: int = 128
    lr: float = 3e-4
    wd: float = 1e-4
    amp: bool = False
    best_metric: str = "val_acc1"
    topk: int = 5
    # If True, log scalars to classifier_run/tensorboard/ (TensorBoard).
    tensorboard: bool = False
    # Transforms: "legacy" ([-1,1] normalize, light jitter) | "notebook" (ImageNet norm + aug as Paleo_OCR.ipynb cell 1).
    transform_style: str = "legacy"
    label_smoothing: float = 0.0
    # LR schedule: cosine decay after linear warmup (both in optimizer steps), matching the notebook.
    warmup_epochs: float = 0.0
    cosine_schedule: bool = False
    # Train only classifier head for this many epochs (timm: params whose name contains "head").
    freeze_backbone_epochs: int = 0
    # Optional split filters for manifests that carry rec["split"] labels.
    # Applied to real-train rows (train_split_filter) and val_manifest rows (val_split_filter).
    train_split_filter: Optional[str] = None
    val_split_filter: Optional[str] = None
    num_workers: int = 8
    prefetch_factor: int = 4
    drop_last: bool = True


@dataclass
class ExperimentConfig:
    experiment_name: str = "paleo_exp"
    seed: int = 42
    task: str = "detector"  # detector | classifier
    output_root: str = "runs/paleo_experiments"
    data: DataConfig = field(default_factory=DataConfig)
    mixing: MixingConfig = field(default_factory=MixingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    phases: List[TrainPhaseConfig] = field(default_factory=list)
    notes: str = ""


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError("Need PyYAML: pip install pyyaml") from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(path: Union[str, Path]) -> ExperimentConfig:
    path = Path(path)
    raw = _load_yaml(path)
    return dict_to_config(raw)


def _only_fields(d: Dict[str, Any], cls) -> Dict[str, Any]:
    keys = getattr(cls, "__dataclass_fields__", {})
    return {k: v for k, v in d.items() if k in keys}


def dict_to_config(d: Dict[str, Any]) -> ExperimentConfig:
    data = d.get("data") or {}
    sources = [SourceConfig(**_only_fields(s, SourceConfig)) for s in data.get("sources", [])]
    mixing = MixingConfig(**_only_fields(d.get("mixing") or {}, MixingConfig))
    det = DetectorConfig(**_only_fields(d.get("detector") or {}, DetectorConfig))
    cls_ = ClassifierConfig(**_only_fields(d.get("classifier") or {}, ClassifierConfig))
    phases = [TrainPhaseConfig(**_only_fields(p, TrainPhaseConfig)) for p in d.get("phases", [])]
    dc = DataConfig(
        sources=sources,
        val_manifest=data.get("val_manifest", ""),
        real_source_id=data.get("real_source_id", "real"),
        synth_stage_a_id=data.get("synth_stage_a_id", "synth_stage_a"),
        synth_stage_b_id=data.get("synth_stage_b_id", "synth_stage_b"),
    )
    return ExperimentConfig(
        experiment_name=d.get("experiment_name", "paleo_exp"),
        seed=int(d.get("seed", 42)),
        task=d.get("task", "detector"),
        output_root=d.get("output_root", "runs/paleo_experiments"),
        data=dc,
        mixing=mixing,
        detector=det,
        classifier=cls_,
        phases=phases,
        notes=d.get("notes", ""),
    )


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Nested dict for JSON serialization (dataclasses → dict).
    """
    from dataclasses import asdict

    return asdict(cfg)
