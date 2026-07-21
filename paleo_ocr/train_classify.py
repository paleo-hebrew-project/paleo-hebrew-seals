"""train_classify
================

Training script for *glyph classification* given cropped glyph images.

We assume the pipeline:
- detector finds bboxes
- crops are extracted (another small utility can do that)
- classifier predicts one of N glyph classes + an extra class "illegible"

Backbone: timm models (ConvNeXt / Swin) with PyTorch.

Dataset format
--------------
Simplest: ImageFolder style:

  cls_ds/
    train/
      א/
        img1.png
        ...
      ב/
      ...
      illegible/
    val/
      א/
      ...

Outputs
-------
- checkpoint .pt
- labels.json (class names)
- metrics_history.jsonl (per-epoch metrics)
- metrics_last.json, confusion_last.json (last epoch snapshot)
- metrics_best.json,  confusion_best.json  (best snapshot)

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# -------------------------
# Metrics helpers (no sklearn)
# -------------------------
def _topk_correct(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    k = min(int(k), int(logits.size(1)))
    topk = logits.topk(k, dim=1).indices  # [B, k]
    return (topk == y.unsqueeze(1)).any(dim=1).sum().item()


def _update_confusion(conf: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    """
    conf[t, p] += 1
    y_true, y_pred on CPU, int64
    """
    C = conf.size(0)
    idx = y_true * C + y_pred
    binc = torch.bincount(idx, minlength=C * C).reshape(C, C)
    conf += binc.to(conf.dtype)


def compute_multiclass_metrics_from_conf(
    conf: torch.Tensor,
    class_names: List[str],
    ignore_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns:
      acc, macro_p/r/f1, weighted_f1, balanced_acc, per_class[]
      + top_confusions (excluding diag)
    """
    eps = 1e-12
    C = conf.size(0)

    conf_f = conf.to(torch.float64)
    tp = conf_f.diag()
    support = conf_f.sum(dim=1)   # true counts
    pred_cnt = conf_f.sum(dim=0)  # predicted counts

    precision = tp / (pred_cnt + eps)
    recall = tp / (support + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    mask = support > 0
    if ignore_idx is not None and 0 <= ignore_idx < C:
        mask[ignore_idx] = False

    macro_p = precision[mask].mean().item() if mask.any() else float("nan")
    macro_r = recall[mask].mean().item() if mask.any() else float("nan")
    macro_f1 = f1[mask].mean().item() if mask.any() else float("nan")

    w = support.clone()
    if ignore_idx is not None and 0 <= ignore_idx < C:
        w[ignore_idx] = 0
    wsum = w.sum().item()
    weighted_f1 = (f1 * w).sum().item() / (wsum + eps)

    total = conf_f.sum().item()
    correct = tp.sum().item()
    acc = correct / (total + eps)

    balanced_acc = recall[mask].mean().item() if mask.any() else float("nan")

    per_class = []
    for i in range(C):
        per_class.append({
            "class": class_names[i] if i < len(class_names) else str(i),
            "class_idx": int(i),
            "support": int(support[i].item()),
            "precision": float(precision[i].item()),
            "recall": float(recall[i].item()),
            "f1": float(f1[i].item()),
        })

    # Top confusions (excluding diagonal)
    top_confusions: List[Dict[str, Any]] = []
    conf_no_diag = conf.clone()
    conf_no_diag.fill_diagonal_(0)
    flat = conf_no_diag.flatten()
    k = min(20, flat.numel())
    if k > 0:
        vals, idxs = torch.topk(flat, k=k)
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            if v <= 0:
                break
            t = idx // C
            p = idx % C
            top_confusions.append({
                "true": class_names[t] if t < len(class_names) else str(t),
                "pred": class_names[p] if p < len(class_names) else str(p),
                "count": int(v),
                "true_idx": int(t),
                "pred_idx": int(p),
            })

    return {
        "acc": float(acc),
        "macro_p": float(macro_p),
        "macro_r": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_acc": float(balanced_acc),
        "per_class": per_class,
        "top_confusions": top_confusions,
    }


# -------------------------
# Args
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train glyph classifier (timm ConvNeXt/Swin).")
    p.add_argument("--data", type=str, required=True, help="Root of ImageFolder dataset")
    p.add_argument("--model", type=str, default="convnext_base", help="timm model name")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--imgsz", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str, default="runs/classify")
    p.add_argument("--name", type=str, default="paleo_glyph_cls")
    p.add_argument("--amp", action="store_true")

    # new: metric/verbosity controls
    p.add_argument("--best-metric", type=str, default="val_acc1",
                   choices=["val_acc1", "val_acc5", "macro_f1", "balanced_acc",
                            "macro_f1_no_illegible", "balanced_acc_no_illegible"],
                   help="Which metric to use for selecting best.pt")
    p.add_argument("--topk", type=int, default=5, help="Compute val_acc@topk (default 5)")
    p.add_argument("--print-worst-k", type=int, default=0,
                   help="Print worst-K classes by F1 each epoch (0 disables)")
    p.add_argument("--print-confusions-k", type=int, default=0,
                   help="Print top-K confusions each epoch (0 disables)")
    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()

    try:
        from torchvision import datasets
    except Exception as e:
        raise RuntimeError("Need torchvision. pip install torchvision") from e

    from paleo_ocr.experiments.classify_core import (
        ClassifyTrainConfig,
        _build_transforms,
        build_train_loader_single,
        train_one_phase,
    )

    out_dir = Path(args.out_dir) / args.name
    data_root = Path(args.data)
    tfm_train, tfm_val = _build_transforms(args.imgsz)
    train_loader, classes = build_train_loader_single(
        data_root, tfm_train, args.batch, args.num_workers, seed=42
    )
    val_ds = datasets.ImageFolder(str(data_root / "val"), transform=tfm_val)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cfg = ClassifyTrainConfig(
        data_dir=data_root,
        out_dir=out_dir,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr=args.lr,
        wd=args.wd,
        num_workers=args.num_workers,
        device=args.device,
        amp=bool(args.amp),
        best_metric=args.best_metric,
        topk=args.topk,
        print_worst_k=int(args.print_worst_k),
        print_confusions_k=int(args.print_confusions_k),
    )
    train_one_phase(cfg, train_loader=train_loader, val_loader=val_loader, classes=classes)
    print(f"[train_classify] done. out={out_dir}")


if __name__ == "__main__":
    main()
