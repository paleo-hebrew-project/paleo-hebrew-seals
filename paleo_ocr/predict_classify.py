"""predict_classify
=================

Batch inference for glyph classification.

Two common modes:
1) Folder mode: classify all images under a folder (glyph crops)
2) JSONL mode: classify crops listed in a JSONL file

Output: predictions.jsonl
{
  "id": "...",
  "path": "...",
  "pred": {
      "label": "א",
      "conf": 0.83,
      "topk": [["א",0.83],["ב",0.04],...],
      "probs": [0.10, 0.83, ...],   # optional (full distribution)
      "logits": [...],             # optional
      "classes": ["א","ב",...]
  }
}

Caching:
- optional per-item cache files

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch predict glyph classes.")
    p.add_argument("--ckpt", type=str, required=True, help="best.pt from train_classify")
    p.add_argument("--model", type=str, default="convnext_base", help="timm model name (must match training)")
    p.add_argument("--folder", type=str, default=None, help="Folder with crop images")
    p.add_argument("--jsonl", type=str, default=None, help="JSONL with {'id','path'} rows")
    p.add_argument("--out", type=str, default="classify_preds.jsonl")
    p.add_argument("--imgsz", type=int, default=128)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--cache-dir", type=str, default=None)

    # New: control how much to write
    p.add_argument(
        "--output",
        type=str,
        default="topk",
        choices=["topk", "full"],
        help="Write only topk (default) or full probability distribution.",
    )
    p.add_argument(
        "--save-logits",
        action="store_true",
        help="Additionally save raw logits (can be large).",
    )
    p.add_argument(
        "--include-classes",
        action="store_true",
        help="Include 'classes' list in every row pred (verbose but self-contained).",
    )

    return p.parse_args()


def _iter_images(folder: Path) -> List[Dict[str, str]]:
    rows = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
            continue
        rows.append({"id": p.stem, "path": str(p)})
    return rows


def main() -> None:
    args = parse_args()

    try:
        import timm
        from torchvision import transforms
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Need timm + torchvision + pillow. pip install timm torchvision pillow") from e

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes: List[str] = ckpt.get("classes")
    if not classes:
        raise ValueError("Checkpoint missing 'classes'")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = timm.create_model(args.model, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    rows: List[Dict[str, str]] = []
    if args.jsonl:
        with Path(args.jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                rid = str(obj.get("id"))
                pth = obj.get("path") or obj.get("crop_path")
                rows.append({"id": rid, "path": str(pth) if pth is not None else None})

    else:
        if not args.folder:
            raise ValueError("Provide --folder or --jsonl")
        rows = _iter_images(Path(args.folder))

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def predict_one(img_path: str) -> Dict[str, Any]:
        im = Image.open(img_path).convert("RGB")
        x = tfm(im).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0]
            probs = _softmax(logits)

        # topk
        topk = min(int(args.topk), probs.numel())
        vals, idxs = torch.topk(probs, k=topk)
        top = [(classes[int(i)], float(v)) for v, i in zip(vals.detach().cpu(), idxs.detach().cpu())]
        best_label, best_conf = top[0]

        pred: Dict[str, Any] = {
            "label": best_label,
            "conf": float(best_conf),
            "topk": top,
        }

        if args.output == "full":
            pred["probs"] = [float(p) for p in probs.detach().cpu().tolist()]

        if args.save_logits:
            pred["logits"] = [float(z) for z in logits.detach().cpu().tolist()]

        if args.include_classes:
            pred["classes"] = classes

        return pred

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            rid = r["id"]
            pth = r["path"]
            cache_path = cache_dir / f"{rid}.json" if cache_dir else None
            if cache_path and cache_path.exists():
                pred = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                pred = predict_one(pth)
                if cache_path:
                    cache_path.write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")
            row = {"id": rid, "path": pth, "pred": pred}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[predict_classify] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
