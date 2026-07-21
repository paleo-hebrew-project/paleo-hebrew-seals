"""Evaluate a timm classifier checkpoint on a val ImageFolder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from paleo_ocr.train_classify import _update_confusion, compute_multiclass_metrics_from_conf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate classifier checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="best.pt with model state_dict + classes")
    p.add_argument("--model", type=str, required=True, help="timm model name (must match training)")
    p.add_argument("--data", type=str, required=True, help="ImageFolder root with val/")
    p.add_argument("--imgsz", type=int, default=128)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-json", type=str, default="metrics_classifier.json")
    p.add_argument("--out-csv", type=str, default="metrics_classifier.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import timm

    try:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(args.checkpoint, map_location="cpu")
    classes: list = ck["classes"]
    state = ck["model"]

    tfm = transforms.Compose(
        [
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    val_ds = datasets.ImageFolder(str(Path(args.data) / "val"), transform=tfm)
    loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model, pretrained=False, num_classes=len(classes))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    conf = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    t0 = time.perf_counter()
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(1)
            yt = y.detach().cpu().to(torch.int64)
            yp = pred.detach().cpu().to(torch.int64)
            _update_confusion(conf, yt, yp)
            n += y.numel()
    t1 = time.perf_counter()

    m_all = compute_multiclass_metrics_from_conf(conf, classes, ignore_idx=None)
    metrics: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "n_val_samples": int(n),
        "wall_time_s": float(t1 - t0),
        "seconds_per_crop": float((t1 - t0) / max(1, n)),
        "crops_per_sec": float(n / max(1e-9, (t1 - t0))),
        **{k: v for k, v in m_all.items() if k not in ("per_class", "top_confusions")},
        "per_class": m_all["per_class"],
    }
    try:
        metrics["num_parameters"] = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        pass
    if torch.cuda.is_available():
        metrics["cuda_peak_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated())

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np

        np.save(str(out_p.with_name(out_p.stem + "_confusion.npy")), conf.cpu().numpy())
    except Exception:
        pass
    out_p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.out_csv).write_text("\n".join(f"{k},{v}" for k, v in sorted(metrics.items()) if not isinstance(v, list)), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
