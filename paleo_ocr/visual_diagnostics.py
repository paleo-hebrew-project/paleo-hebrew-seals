"""visual_diagnostics
===================

Must-have visual diagnostics for Paleo-Hebrew OCR R&D.

Patch update (both requested)
-----------------------------
- Adds *aligned* confusion matrix based on Levenshtein edit trace.
  Outputs:
    - confusion_matrix_aligned.csv (substitutions)
    - insertions.csv
    - deletions.csv
    - confusion_matrix_aligned.png (heatmap, truncated)

Existing features
-----------------
1) Overlays
   - draw predicted/GT bboxes
   - show confidence (det score)

2) Worst-CER gallery
   - build a ranked list from decoded.jsonl (gt_text, pred_text)
   - render an HTML gallery with thumbnails + metadata

3) Quick interactive viewer
   - a single self-contained HTML file with embedded thumbnails (base64)

Inputs expected
---------------
- manifest.jsonl (image paths and GT; may contain gt.bboxes/chars)
- detect_preds.jsonl (optional)
- decoded.jsonl (required for worst-gallery and confusion)

"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception as e:
    raise RuntimeError("Need pillow. pip install pillow") from e


# -------------------------
# I/O
# -------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def safe_get_image_path(man_rec: Dict[str, Any]) -> Optional[str]:
    img = man_rec.get("image") or {}
    p = img.get("abs_path") or img.get("path")
    if p and Path(p).exists():
        return str(p)
    rel = img.get("rel_path")
    root = man_rec.get("images_root")
    if root and rel and Path(root, rel).exists():
        return str(Path(root, rel))
    return None


# -------------------------
# Basic CER
# -------------------------

def _lev(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[lb]


def cer(gt: str, pr: str) -> float:
    gt = gt or ""
    pr = pr or ""
    return _lev(gt, pr) / max(1, len(gt))


# -------------------------
# Overlay rendering
# -------------------------

def _color(name: str) -> Tuple[int, int, int]:
    if name == "gt":
        return (0, 180, 0)
    if name == "pred":
        return (220, 0, 0)
    if name == "text":
        return (10, 10, 10)
    return (0, 0, 255)


def draw_overlay(
    im: Image.Image,
    gt_bboxes: Optional[List[List[float]]] = None,
    pred_bboxes: Optional[List[List[float]]] = None,
    pred_scores: Optional[List[float]] = None,
    title_lines: Optional[List[str]] = None,
) -> Image.Image:
    out = im.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    if gt_bboxes:
        for i, bb in enumerate(gt_bboxes):
            x1, y1, x2, y2 = map(float, bb)
            draw.rectangle([x1, y1, x2, y2], outline=_color("gt"), width=2)

    if pred_bboxes:
        for i, bb in enumerate(pred_bboxes):
            x1, y1, x2, y2 = map(float, bb)
            draw.rectangle([x1, y1, x2, y2], outline=_color("pred"), width=2)
            if pred_scores and i < len(pred_scores):
                draw.text((x1, y2 + 2), f"{pred_scores[i]:.2f}", fill=_color("pred"))

    if title_lines:
        y = 5
        for t in title_lines[:6]:
            draw.text((5, y), t, fill=_color("text"))
            y += 14

    return out


def img_to_base64_png(im: Image.Image, max_side: int = 640) -> str:
    im = im.convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# -------------------------
# Worst gallery
# -------------------------

def build_worst_gallery(
    manifest_jsonl: Path,
    decoded_jsonl: Path,
    out_html: Path,
    detect_jsonl: Optional[Path] = None,
    topk: int = 50,
    min_cer: float = 0.0,
    max_thumb_side: int = 700,
) -> None:
    manifest = read_jsonl(manifest_jsonl)
    decoded = read_jsonl(decoded_jsonl)

    man_by_uid = {m.get("uid"): m for m in manifest if m.get("uid")}

    det_by_uid: Dict[str, Dict[str, Any]] = {}
    if detect_jsonl and detect_jsonl.exists():
        det_rows = read_jsonl(detect_jsonl)
        det_by_uid = {r.get("uid"): r.get("pred") or {} for r in det_rows if r.get("uid")}

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in decoded:
        uid = r.get("uid")
        gt = r.get("gt_text") or ""
        pr = r.get("pred_text") or ""
        c = cer(gt, pr)
        if c < float(min_cer):
            continue
        rr = dict(r)
        rr["CER"] = float(c)
        scored.append((float(c), rr))

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[: int(topk)]

    cards = []
    for c, r in scored:
        uid = r.get("uid")
        m = man_by_uid.get(uid) or {}
        img_path = safe_get_image_path(m)
        if not img_path:
            continue

        im = Image.open(img_path).convert("RGB")

        gt_bboxes = None
        gt = (m.get("gt") or {})
        if isinstance(gt.get("bboxes"), list):
            gt_bboxes = gt.get("bboxes")

        pred_bboxes = None
        pred_scores = None
        dp = det_by_uid.get(uid)
        if dp:
            pred_bboxes = dp.get("bboxes")
            pred_scores = dp.get("scores")

        overlay = draw_overlay(
            im,
            gt_bboxes=gt_bboxes,
            pred_bboxes=pred_bboxes,
            pred_scores=pred_scores,
            title_lines=[
                f"uid={uid}",
                f"CER={c:.3f}",
                f"GT: {r.get('gt_text','')}",
                f"PR: {r.get('pred_text','')}",
            ],
        )

        b64 = img_to_base64_png(overlay, max_side=max_thumb_side)
        cards.append({"uid": uid, "cer": c, "gt": r.get("gt_text", ""), "pr": r.get("pred_text", ""), "b64": b64})

    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Worst CER Gallery</title>",
        "<style>",
        "body{font-family:Arial, sans-serif; margin:20px;} ",
        ".card{border:1px solid #ddd; border-radius:8px; padding:12px; margin:12px 0;} ",
        ".meta{margin:6px 0; white-space:pre-wrap;} ",
        "img{max-width:100%; height:auto;} ",
        ".controls{position:sticky; top:0; background:white; padding:8px; border-bottom:1px solid #eee;} ",
        "input{padding:6px; margin-right:8px;} ",
        "</style>",
        "</head><body>",
        "<div class='controls'>",
        "<label>CER >= </label><input id='mincer' type='number' step='0.01' value='0.0'/>",
        "<label>Search uid </label><input id='q' type='text' placeholder='syn_000123'/>",
        "<button onclick='applyFilter()'>Apply</button>",
        "</div>",
        f"<h2>Worst CER Gallery (top {len(cards)})</h2>",
        "<div id='cards'>",
    ]

    for it in cards:
        html.append(f"<div class='card' data-cer='{it['cer']}' data-uid='{it['uid']}'>")
        html.append(f"<div class='meta'><b>{it['uid']}</b>  CER={it['cer']:.3f}\nGT: {it['gt']}\nPR: {it['pr']}</div>")
        html.append(f"<img src='data:image/png;base64,{it['b64']}'/>")
        html.append("</div>")

    html += [
        "</div>",
        "<script>",
        "function applyFilter(){",
        "  const mincer=parseFloat(document.getElementById('mincer').value||'0');",
        "  const q=(document.getElementById('q').value||'').trim().toLowerCase();",
        "  const cards=document.querySelectorAll('.card');",
        "  cards.forEach(c=>{",
        "    const cer=parseFloat(c.dataset.cer);",
        "    const uid=(c.dataset.uid||'').toLowerCase();",
        "    const ok1=cer>=mincer;",
        "    const ok2=(q==='')||uid.includes(q);",
        "    c.style.display=(ok1 && ok2)?'block':'none';",
        "  });",
        "}",
        "</script>",
        "</body></html>",
    ]

    write_text(out_html, "\n".join(html))


# -------------------------
# Aligned confusion matrix
# -------------------------


def levenshtein_trace(a: str, b: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Return edit script from a->b.

    Each op is tuple: (op, a_char, b_char)
      op in {'eq','sub','ins','del'}

    This is standard DP with backtrace.
    """

    a = a or ""
    b = b or ""
    n, m = len(a), len(b)

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=diag,1=up(del),2=left(ins)

    for i in range(1, n + 1):
        dp[i, 0] = i
        bt[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = j
        bt[0, j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            diag = dp[i - 1, j - 1] + cost
            up = dp[i - 1, j] + 1
            left = dp[i, j - 1] + 1
            best = min(diag, up, left)
            dp[i, j] = best
            # tie-break: prefer diag, then del, then ins
            if best == diag:
                bt[i, j] = 0
            elif best == up:
                bt[i, j] = 1
            else:
                bt[i, j] = 2

    # backtrace
    ops: List[Tuple[str, Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i, j] == 0:
            ca, cb = a[i - 1], b[j - 1]
            if ca == cb:
                ops.append(("eq", ca, cb))
            else:
                ops.append(("sub", ca, cb))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or bt[i, j] == 1):
            ops.append(("del", a[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, b[j - 1]))
            j -= 1

    ops.reverse()
    return ops


def aligned_confusion(
    gt_seqs: List[str],
    pr_seqs: List[str],
    alphabet: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, int], Dict[str, int]]:
    """Build confusion (subs) + insertion/deletion counts from edit traces."""

    # Build alphabet if not provided
    if alphabet is None:
        chars = sorted({c for s in gt_seqs for c in (s or "")} | {c for s in pr_seqs for c in (s or "")})
    else:
        chars = list(alphabet)

    idx = {c: i for i, c in enumerate(chars)}
    M = np.zeros((len(chars), len(chars)), dtype=np.int64)  # gt x pred
    ins: Dict[str, int] = {c: 0 for c in chars}
    dele: Dict[str, int] = {c: 0 for c in chars}

    for gt, pr in zip(gt_seqs, pr_seqs):
        ops = levenshtein_trace(gt or "", pr or "")
        for op, ca, cb in ops:
            if op == "sub":
                if ca in idx and cb in idx:
                    M[idx[ca], idx[cb]] += 1
            elif op == "ins":
                if cb in ins:
                    ins[cb] += 1
            elif op == "del":
                if ca in dele:
                    dele[ca] += 1

    return M, chars, ins, dele


def save_aligned_confusion_outputs(
    M: np.ndarray,
    chars: List[str],
    ins: Dict[str, int],
    dele: Dict[str, int],
    out_dir: Path,
    topn: int = 40,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose topn chars by frequency (gt row sum + deletions)
    freqs = M.sum(axis=1) + np.array([dele.get(c, 0) for c in chars], dtype=np.int64)
    order = np.argsort(-freqs)
    keep = order[: min(int(topn), len(chars))]

    # Substitution confusion CSV
    csv_lines = ["," + ",".join(chars[i] for i in keep)]
    for ii in keep:
        row = [chars[ii]] + [str(int(M[ii, jj])) for jj in keep]
        csv_lines.append(",".join(row))
    write_text(out_dir / "confusion_matrix_aligned.csv", "\n".join(csv_lines))

    # Insertions / deletions CSVs (top)
    ins_sorted = sorted(ins.items(), key=lambda x: x[1], reverse=True)
    del_sorted = sorted(dele.items(), key=lambda x: x[1], reverse=True)

    write_text(out_dir / "insertions.csv", "char,count\n" + "\n".join(f"{c},{n}" for c, n in ins_sorted if n > 0))
    write_text(out_dir / "deletions.csv", "char,count\n" + "\n".join(f"{c},{n}" for c, n in del_sorted if n > 0))

    # Heatmap png
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        subM = M[np.ix_(keep, keep)]
        plt.imshow(subM)
        plt.xticks(range(len(keep)), [chars[i] for i in keep], rotation=90)
        plt.yticks(range(len(keep)), [chars[i] for i in keep])
        plt.title("Aligned substitutions (truncated)")
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix_aligned.png", dpi=200)
        plt.close()
    except Exception:
        pass


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual diagnostics for OCR experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gallery", help="Build worst-CER HTML gallery with overlays")
    g.add_argument("--manifest", type=str, required=True)
    g.add_argument("--decoded", type=str, required=True)
    g.add_argument("--detect", type=str, default=None)
    g.add_argument("--out", type=str, default="worst_gallery.html")
    g.add_argument("--topk", type=int, default=50)
    g.add_argument("--min-cer", type=float, default=0.0)

    c = sub.add_parser("confusion", help="Aligned confusion matrix via edit trace")
    c.add_argument("--decoded", type=str, required=True)
    c.add_argument("--out-dir", type=str, default="confusion_out")
    c.add_argument("--topn", type=int, default=40)

    o = sub.add_parser("overlay", help="Render overlay PNGs for selected uids")
    o.add_argument("--manifest", type=str, required=True)
    o.add_argument("--detect", type=str, default=None)
    o.add_argument("--uids", type=str, nargs="+", required=True)
    o.add_argument("--out-dir", type=str, default="overlay_out")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "gallery":
        build_worst_gallery(
            manifest_jsonl=Path(args.manifest),
            decoded_jsonl=Path(args.decoded),
            detect_jsonl=Path(args.detect) if args.detect else None,
            out_html=Path(args.out),
            topk=int(args.topk),
            min_cer=float(args.min_cer),
        )
        print(f"[visual] wrote {args.out}")

    elif args.cmd == "confusion":
        rows = read_jsonl(Path(args.decoded))
        gt = [r.get("gt_text", "") for r in rows]
        pr = [r.get("pred_text", "") for r in rows]
        M, chars, ins, dele = aligned_confusion(gt, pr)
        save_aligned_confusion_outputs(M, chars, ins, dele, Path(args.out_dir), topn=int(args.topn))
        print(f"[visual] aligned confusion saved to {args.out_dir}")

    elif args.cmd == "overlay":
        manifest = read_jsonl(Path(args.manifest))
        man_by_uid = {m.get("uid"): m for m in manifest if m.get("uid")}

        det_by_uid: Dict[str, Dict[str, Any]] = {}
        if args.detect:
            det_rows = read_jsonl(Path(args.detect))
            det_by_uid = {r.get("uid"): r.get("pred") or {} for r in det_rows if r.get("uid")}

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for uid in args.uids:
            m = man_by_uid.get(uid)
            if not m:
                continue
            img_path = safe_get_image_path(m)
            if not img_path:
                continue
            im = Image.open(img_path).convert("RGB")

            gt_bboxes = None
            gt = (m.get("gt") or {})
            if isinstance(gt.get("bboxes"), list):
                gt_bboxes = gt.get("bboxes")

            pred_bboxes = None
            pred_scores = None
            dp = det_by_uid.get(uid)
            if dp:
                pred_bboxes = dp.get("bboxes")
                pred_scores = dp.get("scores")

            overlay = draw_overlay(im, gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes, pred_scores=pred_scores)
            overlay.save(out_dir / f"{uid}_overlay.png")

        print(f"[visual] overlays saved to {out_dir}")


if __name__ == "__main__":
    main()
