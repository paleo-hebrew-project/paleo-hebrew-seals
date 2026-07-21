"""Classifier training core (timm + ImageFolder), including mixed-source regimes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler

from paleo_ocr.experiments.dataloaders import curriculum_synth_fraction, weighted_sampler_for_concat
from paleo_ocr.train_classify import (
    _topk_correct,
    _update_confusion,
    compute_multiclass_metrics_from_conf,
)


@dataclass
class ClassifyTrainConfig:
    data_dir: Path  # contains train/ and val/ ImageFolder
    out_dir: Path
    model: str = "convnext_base"
    epochs: int = 30
    batch: int = 128
    imgsz: int = 128
    lr: float = 3e-4
    wd: float = 1e-4
    num_workers: int = 8
    device: str = "cuda"
    amp: bool = False
    best_metric: str = "val_acc1"
    topk: int = 5
    print_worst_k: int = 0
    print_confusions_k: int = 0
    seed: int = 42
    # If True, writes TensorBoard scalars under out_dir/tensorboard/ (pip install tensorboard).
    tensorboard: bool = False
    # "legacy" | "notebook" — see _build_transforms
    transform_style: str = "legacy"
    label_smoothing: float = 0.0
    warmup_epochs: float = 0.0
    cosine_schedule: bool = False
    freeze_backbone_epochs: int = 0
    prefetch_factor: int = 4
    drop_last: bool = True
    # Optional: load weights before training (sequential phases / finetune).
    warm_start_ckpt: Optional[Path] = None


def _pick_best_metric(epoch_metrics: Dict[str, Any], best_metric: str, illegible_idx: Optional[int]) -> float:
    bm = best_metric
    if bm == "val_acc1":
        return float(epoch_metrics["val_acc1"])
    if bm == "val_acc5":
        if "val_acc5" in epoch_metrics:
            return float(epoch_metrics["val_acc5"])
        for k in epoch_metrics:
            if k.startswith("val_acc") and k != "val_acc1":
                return float(epoch_metrics[k])
        return 0.0
    if bm == "macro_f1":
        return float(epoch_metrics["metrics_all"]["macro_f1"])
    if bm == "balanced_acc":
        return float(epoch_metrics["metrics_all"]["balanced_acc"])
    if bm == "macro_f1_no_illegible" and illegible_idx is not None:
        return float(epoch_metrics["metrics_no_illegible"]["macro_f1"])
    if bm == "balanced_acc_no_illegible" and illegible_idx is not None:
        return float(epoch_metrics["metrics_no_illegible"]["balanced_acc"])
    return float(epoch_metrics["val_acc1"])


def _build_transforms(imgsz: int, style: str = "legacy"):
    """style=legacy: symmetric [-1,1] norm (previous project default). style=notebook: ImageNet norm + aug from Paleo_OCR.ipynb."""
    from torchvision import transforms

    if style == "legacy":
        tfm_train = transforms.Compose(
            [
                transforms.Resize((imgsz, imgsz)),
                transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        tfm_val = transforms.Compose(
            [
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return tfm_train, tfm_val

    if style == "notebook":
        tfm_train = transforms.Compose(
            [
                transforms.Resize((imgsz, imgsz)),
                transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)], p=0.4),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=3),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        tfm_val = transforms.Compose(
            [
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return tfm_train, tfm_val

    raise ValueError(f"Unknown transform style: {style!r} (use 'legacy' or 'notebook')")


def _timm_classifier_param_ids(model: nn.Module) -> set[int]:
    """
    Resolve trainable classifier params across timm families.

    ConvNeXt / ViT / Swin typically expose a `head`, ResNet uses `fc`,
    EfficientNet commonly uses `classifier`, etc.
    """

    def ids_from_module(obj: Any) -> set[int]:
        if isinstance(obj, nn.Module):
            return {id(p) for p in obj.parameters(recurse=True)}
        if isinstance(obj, (list, tuple)):
            out: set[int] = set()
            for item in obj:
                if isinstance(item, nn.Module):
                    out.update(id(p) for p in item.parameters(recurse=True))
            return out
        return set()

    def ids_from_prefixes(prefixes: List[str]) -> set[int]:
        clean = [p for p in prefixes if p]
        out: set[int] = set()
        for name, p in model.named_parameters():
            for prefix in clean:
                if (
                    name == prefix
                    or name.startswith(prefix + ".")
                    or name.endswith("." + prefix)
                    or f".{prefix}." in name
                ):
                    out.add(id(p))
                    break
        return out

    get_classifier = getattr(model, "get_classifier", None)
    if callable(get_classifier):
        try:
            classifier_obj = get_classifier()
        except TypeError:
            classifier_obj = None
        out = ids_from_module(classifier_obj)
        if out:
            return out

        try:
            classifier_name = get_classifier(name_only=True)
        except TypeError:
            classifier_name = classifier_obj if isinstance(classifier_obj, str) else None
        if isinstance(classifier_name, str) and classifier_name:
            out = ids_from_prefixes([classifier_name])
            if out:
                return out

    # Fallback for common timm naming conventions.
    return ids_from_prefixes(["head", "fc", "classifier", "classif"])


def _timm_set_head_only_trainable(model: nn.Module, head_only: bool) -> None:
    """Freeze backbone and keep only classifier params trainable across timm families."""
    if not head_only:
        for p in model.parameters():
            p.requires_grad = True
        return

    classifier_ids = _timm_classifier_param_ids(model)
    if not classifier_ids:
        raise ValueError(
            f"Could not identify classifier parameters for model type {type(model).__name__}. "
            "Set freeze_backbone_epochs=0 for this model/config."
        )

    n_trainable = 0
    for _n, p in model.named_parameters():
        p.requires_grad = id(p) in classifier_ids
        if p.requires_grad:
            n_trainable += 1

    if n_trainable == 0:
        raise ValueError(
            f"Classifier parameter detection returned an empty set for model type {type(model).__name__}. "
            "Set freeze_backbone_epochs=0 for this model/config."
        )


def _build_ce_loss(label_smoothing: float) -> nn.Module:
    ls = float(label_smoothing)
    if ls > 0.0:
        return nn.CrossEntropyLoss(label_smoothing=ls)
    return nn.CrossEntropyLoss()


def _make_cosine_warmup_lambda(
    *,
    steps_per_epoch: int,
    total_epochs: int,
    warmup_epochs: float,
) -> Callable[[int], float]:
    total_steps = max(1, steps_per_epoch * max(1, total_epochs))
    warmup_steps = max(1, int(steps_per_epoch * float(warmup_epochs)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def train_one_phase(
    cfg: ClassifyTrainConfig,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: List[str],
    per_batch_hook: Optional[Callable[[Dict[str, int]], None]] = None,
) -> Dict[str, Any]:
    """
    Run training for cfg.epochs using provided loaders.
    per_batch_hook: called with source tag counts if caller adds tags (optional).
    """
    try:
        import timm
    except Exception as e:
        raise RuntimeError("Need timm. pip install timm") from e

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "labels.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    use_pretrained = cfg.warm_start_ckpt is None
    model = timm.create_model(cfg.model, pretrained=use_pretrained, num_classes=len(classes))
    model.to(device)

    if cfg.warm_start_ckpt is not None:
        ckpt_path = Path(cfg.warm_start_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"warm_start_ckpt not found: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu")
        state = ck.get("model", ck) if isinstance(ck, dict) else ck
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint at {ckpt_path} does not contain a state dict")
        miss, unexp = model.load_state_dict(state, strict=False)
        if miss or unexp:
            print(f"[classify] warm_start missing={len(miss)} unexpected={len(unexp)} (strict=False)")

    loss_fn = _build_ce_loss(cfg.label_smoothing)
    freeze_e = int(cfg.freeze_backbone_epochs)
    if freeze_e < 0:
        raise ValueError("freeze_backbone_epochs must be >= 0")
    if freeze_e > 0 and freeze_e >= cfg.epochs:
        raise ValueError(f"freeze_backbone_epochs ({freeze_e}) must be < epochs ({cfg.epochs})")

    steps_per_epoch = len(train_loader)
    cosine = bool(cfg.cosine_schedule)
    warmup_e = float(cfg.warmup_epochs)

    def make_opt_sched(head_only: bool) -> Tuple[torch.optim.Optimizer, Any, Any]:
        if head_only:
            _timm_set_head_only_trainable(model, head_only=True)
            params = [p for p in model.parameters() if p.requires_grad]
        else:
            _timm_set_head_only_trainable(model, head_only=False)
            params = list(model.parameters())
        o = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
        if cosine:
            lam = _make_cosine_warmup_lambda(
                steps_per_epoch=steps_per_epoch,
                total_epochs=max(1, cfg.epochs),
                warmup_epochs=warmup_e,
            )
            s = torch.optim.lr_scheduler.LambdaLR(o, lam)
        else:
            s = None
        sc = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp))
        return o, s, sc

    if freeze_e > 0:
        opt, scheduler, scaler = make_opt_sched(head_only=True)
    else:
        opt, scheduler, scaler = make_opt_sched(head_only=False)

    illegible_idx = classes.index("illegible") if "illegible" in classes else None

    best_metric_val = float("-inf")
    best_epoch = -1
    hist_path = cfg.out_dir / "metrics_history.jsonl"
    if hist_path.exists():
        hist_path.unlink()

    tb_writer: Any = None
    if getattr(cfg, "tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = cfg.out_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
        except ImportError:
            print("[classify] tensorboard requested but not installed; pip install tensorboard")

    last_metrics: Dict[str, Any] = {}

    # global_step only for logging if needed; scheduler has internal step for LambdaLR
    for epoch in range(cfg.epochs):
        if freeze_e > 0 and epoch == freeze_e:
            opt, scheduler, scaler = make_opt_sched(head_only=False)

        model.train()
        tr_total = 0
        tr_correct = 0
        tr_loss_sum = 0.0
        source_counts: Dict[str, int] = {}

        for batch in train_loader:
            if len(batch) == 3:
                x, y, tags = batch
            else:
                x, y = batch
                tags = None

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            bs = y.numel()
            tr_total += bs
            tr_loss_sum += float(loss.item()) * bs
            tr_correct += (logits.argmax(1) == y).sum().item()

            if tags is not None:
                for t in tags:
                    source_counts[t] = source_counts.get(t, 0) + 1
            if per_batch_hook and tags is not None:
                per_batch_hook(source_counts)

        train_acc = tr_correct / max(1, tr_total)
        train_loss = tr_loss_sum / max(1, tr_total)

        model.eval()
        val_total = 0
        val_correct1 = 0
        val_correctk = 0
        val_loss_sum = 0.0
        conf = torch.zeros((len(classes), len(classes)), dtype=torch.int64)

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                bs = y.numel()
                val_total += bs
                val_loss_sum += float(loss.item()) * bs
                pred = logits.argmax(1)
                val_correct1 += (pred == y).sum().item()
                val_correctk += _topk_correct(logits, y, k=cfg.topk)
                yt = y.detach().cpu().to(torch.int64)
                yp = pred.detach().cpu().to(torch.int64)
                _update_confusion(conf, yt, yp)

        val_acc1 = val_correct1 / max(1, val_total)
        val_acck = val_correctk / max(1, val_total)
        val_loss = val_loss_sum / max(1, val_total)

        m_all = compute_multiclass_metrics_from_conf(conf, classes, ignore_idx=None)
        m_no_ill = (
            compute_multiclass_metrics_from_conf(conf, classes, ignore_idx=illegible_idx)
            if illegible_idx is not None
            else None
        )

        epoch_metrics: Dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc1": float(val_acc1),
            f"val_acc{int(cfg.topk)}": float(val_acck),
            "metrics_all": m_all,
            "metrics_no_illegible": m_no_ill,
            "best_metric_name": cfg.best_metric,
            "train_source_counts": source_counts if source_counts else None,
        }

        torch.save({"model": model.state_dict(), "classes": classes, "epoch": epoch}, cfg.out_dir / "last.pt")

        metric_val = _pick_best_metric(epoch_metrics, cfg.best_metric, illegible_idx)
        if metric_val > best_metric_val:
            best_metric_val = metric_val
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "classes": classes, "epoch": epoch}, cfg.out_dir / "best.pt")
            (cfg.out_dir / "metrics_best.json").write_text(
                json.dumps(epoch_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (cfg.out_dir / "confusion_best.json").write_text(json.dumps(conf.tolist(), ensure_ascii=False), encoding="utf-8")

        (cfg.out_dir / "metrics_last.json").write_text(
            json.dumps(epoch_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (cfg.out_dir / "confusion_last.json").write_text(json.dumps(conf.tolist(), ensure_ascii=False), encoding="utf-8")

        with hist_path.open("a", encoding="utf-8") as hf:
            hf.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", train_loss, epoch)
            tb_writer.add_scalar("train/acc", train_acc, epoch)
            tb_writer.add_scalar("val/loss", val_loss, epoch)
            tb_writer.add_scalar("val/acc1", val_acc1, epoch)
            tb_writer.add_scalar("val/macro_f1", float(m_all["macro_f1"]), epoch)

        last_metrics = epoch_metrics
        print(
            f"[epoch {epoch+1}/{cfg.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_acc1={val_acc1:.4f} | macroF1={m_all['macro_f1']:.4f} | best={best_metric_val:.4f} (ep={best_epoch+1})"
        )

        if int(cfg.print_worst_k) > 0:
            per = m_all["per_class"]
            per_nz = [r for r in per if int(r["support"]) > 0]
            per_nz.sort(key=lambda r: float(r["f1"]))
            worst = per_nz[: int(cfg.print_worst_k)]
            if worst:
                msg = ", ".join([f"{r['class']} f1={r['f1']:.3f} sup={r['support']}" for r in worst])
                print(f"  worst_f1: {msg}")
        if int(cfg.print_confusions_k) > 0:
            confs = (m_all.get("top_confusions") or [])[: int(cfg.print_confusions_k)]
            if confs:
                msg = ", ".join([f"{c['true']}→{c['pred']}({c['count']})" for c in confs])
                print(f"  top_confusions: {msg}")

    if tb_writer is not None:
        tb_writer.close()

    summary = {
        "best_metric": cfg.best_metric,
        "best_value": best_metric_val,
        "best_epoch": best_epoch,
        "last_metrics": last_metrics,
        "out_dir": str(cfg.out_dir),
    }
    (cfg.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


class TaggedTensorDataset(Dataset):
    """Wrap a dataset that returns (x,y) to also return a string tag."""

    def __init__(self, ds: Dataset, tag: str):
        self.ds = ds
        self.tag = tag

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        x, y = self.ds[idx]
        return x, y, self.tag


def build_train_loader_single(
    data_dir: Path,
    tfm_train: Any,
    batch: int,
    num_workers: int,
    seed: int,
    *,
    drop_last: bool = True,
    prefetch_factor: int = 4,
) -> Tuple[DataLoader, List[str]]:
    from torchvision import datasets

    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=tfm_train)
    g = torch.Generator()
    g.manual_seed(seed)
    kwargs: Dict[str, Any] = {
        "batch_size": batch,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "generator": g,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    loader = DataLoader(train_ds, **kwargs)
    return loader, train_ds.classes


def build_train_loader_weighted(
    train_roots: List[Tuple[Path, str, float]],
    tfm_train: Any,
    batch: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, List[str]]:
    """
    train_roots: (folder containing class subdirs, source_tag, weight)
    """
    from torchvision import datasets

    ds_list: List[Dataset] = []
    tags: List[str] = []
    weights: List[float] = []
    classes_ref: Optional[List[str]] = None

    for root, tag, wt in train_roots:
        ds0 = datasets.ImageFolder(str(root), transform=tfm_train)
        if classes_ref is None:
            classes_ref = ds0.classes
        elif ds0.classes != classes_ref:
            raise ValueError(f"class mismatch {root}: {ds0.classes} vs {classes_ref}")
        ds_list.append(TaggedTensorDataset(ds0, tag))
        tags.append(tag)
        weights.append(float(wt))

    concat = ConcatDataset(ds_list)
    num_samples = max(len(concat), batch * 2)
    sampler = weighted_sampler_for_concat([d.ds for d in ds_list], weights, num_samples=num_samples, seed=seed)

    def collate(batch):
        xs, ys, ts = zip(*batch)
        import torch

        return torch.stack(xs, 0), torch.tensor(ys), list(ts)

    loader = DataLoader(
        concat,
        batch_size=batch,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    assert classes_ref is not None
    return loader, classes_ref


def train_alternating_epochs(
    cfg: ClassifyTrainConfig,
    *,
    real_train_dir: Path,
    synth_train_dir: Path,
    val_dir: Path,
    k_synth_batches: int = 1,
) -> Dict[str, Any]:
    """One 'epoch' = one pass over real loader + k_synth_batches passes over synth per real batch (approximate)."""
    from torchvision import datasets

    tfm_train, tfm_val = _build_transforms(cfg.imgsz, getattr(cfg, "transform_style", "legacy"))
    train_real = datasets.ImageFolder(str(real_train_dir), transform=tfm_train)
    train_synth = datasets.ImageFolder(str(synth_train_dir), transform=tfm_train)
    val_ds = datasets.ImageFolder(str(val_dir), transform=tfm_val)
    if train_real.classes != train_synth.classes:
        raise ValueError("class folders must match between real and synth train dirs")
    classes = train_real.classes

    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    import timm

    model = timm.create_model(cfg.model, pretrained=True, num_classes=len(classes))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    loss_fn = _build_ce_loss(getattr(cfg, "label_smoothing", 0.0))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp))

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    d_kw: Dict[str, Any] = {}
    if cfg.num_workers > 0:
        d_kw["persistent_workers"] = True
        d_kw["prefetch_factor"] = max(2, int(getattr(cfg, "prefetch_factor", 4)))
    loader_real = DataLoader(
        train_real,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        generator=g,
        **d_kw,
    )
    loader_synth = DataLoader(
        train_synth,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
        **d_kw,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"real_samples": 0, "synth_samples": 0, "optimizer_steps": 0}

    for epoch in range(cfg.epochs):
        model.train()
        it_s = iter(loader_synth)
        for x, y in loader_real:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            stats["real_samples"] += y.numel()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            stats["optimizer_steps"] += 1

            for _ in range(k_synth_batches):
                try:
                    xs, ys = next(it_s)
                except StopIteration:
                    it_s = iter(loader_synth)
                    xs, ys = next(it_s)
                xs = xs.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True)
                stats["synth_samples"] += ys.numel()
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                    logits = model(xs)
                    loss = loss_fn(logits, ys)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                stats["optimizer_steps"] += 1

        # val
        model.eval()
        conf = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                pred = logits.argmax(1)
                yt = y.detach().cpu().to(torch.int64)
                yp = pred.detach().cpu().to(torch.int64)
                _update_confusion(conf, yt, yp)

        m_all = compute_multiclass_metrics_from_conf(conf, classes, ignore_idx=None)
        print(f"[alt epoch {epoch+1}] macroF1={m_all['macro_f1']:.4f} real={stats['real_samples']} synth={stats['synth_samples']} steps={stats['optimizer_steps']}")

    torch.save({"model": model.state_dict(), "classes": classes, "epoch": cfg.epochs - 1}, cfg.out_dir / "last.pt")
    (cfg.out_dir / "alternating_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return {"alternating_stats": stats, "out_dir": str(cfg.out_dir)}


def train_curriculum_weighted(
    base: ClassifyTrainConfig,
    *,
    real_train_dir: Path,
    synth_train_dir: Path,
    val_loader: DataLoader,
    classes: List[str],
    synth_frac_start: float,
    synth_frac_end: float,
) -> Dict[str, Any]:
    """Each epoch: weighted mix between real and synth train folders; linear schedule."""
    tfm_train, _ = _build_transforms(base.imgsz, getattr(base, "transform_style", "legacy"))
    hist_path = base.out_dir / "metrics_history.jsonl"
    base.out_dir.mkdir(parents=True, exist_ok=True)
    if hist_path.exists():
        hist_path.unlink()

    device = torch.device(base.device if torch.cuda.is_available() else "cpu")
    import timm

    model = timm.create_model(base.model, pretrained=True, num_classes=len(classes))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=base.lr, weight_decay=base.wd)
    loss_fn = _build_ce_loss(getattr(base, "label_smoothing", 0.0))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(base.amp))

    illegible_idx = classes.index("illegible") if "illegible" in classes else None
    best_metric_val = float("-inf")
    best_epoch = -1

    for epoch in range(base.epochs):
        frac = curriculum_synth_fraction(epoch, base.epochs, synth_frac_start, synth_frac_end)
        w_real = max(1e-6, 1.0 - frac)
        w_synth = max(1e-6, frac)
        train_loader, _ = build_train_loader_weighted(
            [
                (real_train_dir, "real", w_real),
                (synth_train_dir, "synth", w_synth),
            ],
            tfm_train,
            base.batch,
            base.num_workers,
            base.seed + epoch,
        )

        model.train()
        for batch in train_loader:
            if len(batch) == 3:
                x, y, _tags = batch
            else:
                x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(base.amp)):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        conf = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                pred = logits.argmax(1)
                yt = y.detach().cpu().to(torch.int64)
                yp = pred.detach().cpu().to(torch.int64)
                _update_confusion(conf, yt, yp)

        m_all = compute_multiclass_metrics_from_conf(conf, classes, ignore_idx=None)
        m_no_ill = (
            compute_multiclass_metrics_from_conf(conf, classes, ignore_idx=illegible_idx)
            if illegible_idx is not None
            else None
        )
        epoch_metrics: Dict[str, Any] = {
            "epoch": epoch,
            "curriculum_synth_frac": float(frac),
            "val_acc1": float(m_all["acc"]),
            "metrics_all": m_all,
            "metrics_no_illegible": m_no_ill,
        }
        metric_val = _pick_best_metric(epoch_metrics, base.best_metric, illegible_idx)
        if metric_val > best_metric_val:
            best_metric_val = metric_val
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "classes": classes, "epoch": epoch}, base.out_dir / "best.pt")

        torch.save({"model": model.state_dict(), "classes": classes, "epoch": epoch}, base.out_dir / "last.pt")
        with hist_path.open("a", encoding="utf-8") as hf:
            hf.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
        print(f"[curriculum {epoch+1}/{base.epochs}] synth_frac={frac:.3f} macroF1={m_all['macro_f1']:.4f}")

    (base.out_dir / "train_summary.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_metric": best_metric_val}, indent=2), encoding="utf-8"
    )
    return {"out_dir": str(base.out_dir), "best_metric": best_metric_val, "best_epoch": best_epoch}
