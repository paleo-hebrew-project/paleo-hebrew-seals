"""ocr_decoder
================

A lightweight decoder that turns *symbol detections* into a linear text.

This module is intended for the pipeline:

    image -> detector (bboxes) -> classifier (class_probs per bbox) -> decoder -> text

It is **not** a language model. It will not invent content. It only:
- picks a character per detected bbox (or "[?]" if low confidence),
- groups bboxes into lines/columns,
- sorts within each line according to reading direction,
- optionally applies *conservative* rule-based cleanup (no hallucinations).

The design is archaeology-friendly:
- deterministic
- debuggable (returns rich intermediate structure)
- robust to missing/extra detections

Expected inputs
---------------
Detections are a list of dicts (or dataclasses) with at least:

    {
      "bbox": [x1, y1, x2, y2],           # pixel coordinates
      "probs": [p0, p1, ..., pK-1],      # class probabilities
      "classes": ["א", "ב", ..., "<blank>"],  # same length as probs OR provided globally
      "score": 0.93,                     # optional detector score
      "id": "..."                        # optional
    }

You can also pass `logits` instead of `probs`; softmax will be applied.

Outputs
-------
`decode_detections(...)` returns a `DecodedText` object with:
- text: final string
- lines: list of decoded lines with per-token info
- debug: grouping/sorting diagnostics

Notes for Paleo-Hebrew
----------------------
For Paleo-Hebrew / Phoenician-like scripts:
- reading direction is usually RTL
- seals may be mirrored; upstream `orientation_mirror` may set mirrored/reverse
- multi-register inscriptions exist; grouping can be vertical (columns)

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# -------------------------
# Data structures
# -------------------------

BBox = Tuple[float, float, float, float]


@dataclass
class Token:
    bbox: BBox
    char: str
    conf: float
    cls_idx: int
    det_score: Optional[float] = None
    token_id: Optional[str] = None


@dataclass
class DecodedLine:
    tokens: List[Token]
    # Bounding box covering the whole line (useful for debugging/visualization)
    bbox: BBox


@dataclass
class DecodedText:
    text: str
    lines: List[DecodedLine]
    debug: Dict[str, Any]


# -------------------------
# Utilities
# -------------------------


def _softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


def _bbox_center(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_union(boxes: Sequence[BBox]) -> BBox:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def _iou_y(a: BBox, b: BBox) -> float:
    """1D IoU on Y axis (vertical overlap ratio)."""
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    if inter <= 0:
        return 0.0
    union = (ay2 - ay1) + (by2 - by1) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _iou_x(a: BBox, b: BBox) -> float:
    """1D IoU on X axis (horizontal overlap ratio)."""
    ax1, ax2 = a[0], a[2]
    bx1, bx2 = b[0], b[2]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) + (bx2 - bx1) - inter
    if union <= 0:
        return 0.0
    return inter / union


# -------------------------
# Core decoding
# -------------------------


def pick_char(
    det: Dict[str, Any],
    classes: Optional[Sequence[str]] = None,
    blank_tokens: Sequence[str] = ("<blank>", "[blank]", "", "∅"),
    min_char_conf: float = 0.35,
    unknown_token: str = "[?]",
) -> Token:
    """Pick a single character for a detection.

    Rules:
    - use `probs` if provided; else softmax `logits`
    - take argmax
    - if chosen class is blank-like OR conf < min_char_conf -> unknown_token

    This function never invents characters.
    """

    bbox = tuple(map(float, det["bbox"]))  # type: ignore
    det_score = det.get("score")
    token_id = det.get("id")

    local_classes = det.get("classes")
    cls_list: Optional[Sequence[str]] = None
    if isinstance(local_classes, (list, tuple)) and local_classes:
        cls_list = local_classes
    elif classes is not None:
        cls_list = classes

    if cls_list is None:
        raise ValueError("Classes list must be provided globally or per detection (det['classes']).")

    if "probs" in det and det["probs"] is not None:
        probs = list(map(float, det["probs"]))
    elif "logits" in det and det["logits"] is not None:
        probs = _softmax(list(map(float, det["logits"])))
    else:
        raise ValueError("Detection must contain 'probs' or 'logits'.")

    if len(probs) != len(cls_list):
        raise ValueError(f"len(probs)={len(probs)} != len(classes)={len(cls_list)}")

    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    conf = float(probs[best_idx])
    char = str(cls_list[best_idx])

    if (char in blank_tokens) or (conf < float(min_char_conf)):
        char = unknown_token

    return Token(
        bbox=bbox, char=char, conf=conf, cls_idx=best_idx,
        det_score=float(det_score) if det_score is not None else None,
        token_id=str(token_id) if token_id is not None else None,
    )


def group_into_lines(
    tokens: Sequence[Token],
    mode: str = "auto",
    overlap_thresh: float = 0.35,
) -> Tuple[List[List[Token]], Dict[str, Any]]:
    """Group tokens into lines or columns.

    Parameters
    ----------
    tokens:
        Detected tokens with bboxes.
    mode:
        "horizontal" -> group by Y-overlap (text lines)
        "vertical"   -> group by X-overlap (columns/registers)
        "auto"       -> choose based on which grouping yields fewer groups
                        and more consistent group sizes.
    overlap_thresh:
        Minimal 1D overlap IoU to consider tokens belonging to the same line.

    Returns
    -------
    groups:
        List of groups, each group is a list of tokens.
    debug:
        Diagnostics.
    """

    if not tokens:
        return [], {"mode": mode, "reason": "no_tokens"}

    def _group(axis: str) -> List[List[Token]]:
        # greedy agglomeration based on overlap on chosen axis
        groups: List[List[Token]] = []
        for t in sorted(tokens, key=lambda x: (_bbox_center(x.bbox)[1], _bbox_center(x.bbox)[0])):
            placed = False
            for g in groups:
                # compare with representative bbox (union)
                rep = _bbox_union([z.bbox for z in g])
                ov = _iou_y(rep, t.bbox) if axis == "y" else _iou_x(rep, t.bbox)
                if ov >= overlap_thresh:
                    g.append(t)
                    placed = True
                    break
            if not placed:
                groups.append([t])
        return groups

    horiz = _group("y")
    vert = _group("x")

    chosen = mode
    reason = "user"
    if mode == "auto":
        # Choose grouping that yields fewer groups, but avoid degenerate single-token groups
        def score(groups: List[List[Token]]) -> Tuple[int, float]:
            sizes = [len(g) for g in groups]
            avg = sum(sizes) / max(1, len(sizes))
            frac_single = sum(1 for s in sizes if s == 1) / max(1, len(sizes))
            # lower groups and lower single-frac is better; higher avg is better
            return (len(groups), frac_single - 0.01 * avg)

        sh = score(horiz)
        sv = score(vert)
        if sh < sv:
            chosen = "horizontal"
            reason = "auto:horiz"
        else:
            chosen = "vertical"
            reason = "auto:vert"

    groups = horiz if chosen == "horizontal" else vert
    return groups, {
        "mode": chosen,
        "reason": reason,
        "n_groups": len(groups),
        "n_tokens": len(tokens),
        "overlap_thresh": overlap_thresh,
    }


def sort_tokens_in_group(
    group: Sequence[Token],
    reading_dir: str = "rtl",
    sort_primary: str = "x",
) -> List[Token]:
    """Sort tokens within a line/column.

    sort_primary:
      "x" means left-right ordering within a line (then y).
      "y" means top-down ordering within a column (then x).

    reading_dir:
      "rtl" reverses x-order for horizontal lines.
      "ltr" keeps x-order.

    For vertical groups, reading_dir affects the secondary x sort only.
    """

    if not group:
        return []

    if sort_primary == "y":
        # columns/registers
        sorted_g = sorted(group, key=lambda t: (_bbox_center(t.bbox)[1], _bbox_center(t.bbox)[0]))
        if reading_dir == "rtl":
            # keep primary (y) but reverse secondary x inside same y band is too heavy;
            # we keep as-is for simplicity.
            return sorted_g
        return sorted_g

    # default: lines
    sorted_g = sorted(group, key=lambda t: (_bbox_center(t.bbox)[0], _bbox_center(t.bbox)[1]))
    if reading_dir == "rtl":
        sorted_g = list(reversed(sorted_g))
    return sorted_g


def join_tokens(
    tokens: Sequence[Token],
    char_joiner: str = "",
    unknown_token: str = "[?]",
) -> str:
    """Join characters into a string.

    We do *not* infer spaces reliably from bboxes here (seals are noisy).
    Spaces can be inserted by an optional pattern postprocessor.
    """
    # Ensure unknown token is consistently bracketed
    out = []
    for t in tokens:
        if t.char.strip() == "?":
            out.append(unknown_token)
        else:
            out.append(t.char)
    return char_joiner.join(out)


# -------------------------
# Conservative postprocessing (no hallucinations)
# -------------------------


def conservative_cleanup(
    text: str,
    ruleset: str = "none",
) -> Tuple[str, Dict[str, Any]]:
    """Apply *conservative* cleanup rules.

    This is intentionally minimal and deterministic.

    Supported rulesets:
    - none: do nothing
    - paleo_basic: remove repeated unknown tokens, normalize common separators

    You can extend this to include archaeologically-justified templates, e.g.:
      - prefix 'ל' (to/for)
      - patronymic 'בן'

    But do **not** replace unknown chars with guessed letters.
    """

    dbg: Dict[str, Any] = {"ruleset": ruleset, "applied": []}
    if ruleset == "none":
        return text, dbg

    out = text

    if ruleset in ("paleo_basic", "basic"):
        before = out
        # collapse multiple unknown markers
        out = out.replace("[?][?]", "[?]")
        # normalize common separators
        out = out.replace("|", " ").replace("/", " ")
        # collapse multiple spaces
        out = " ".join(out.split())
        if out != before:
            dbg["applied"].append("collapse_unknown_and_spaces")

    return out, dbg


# -------------------------
# Public API
# -------------------------


def decode_detections(
    detections: Sequence[Dict[str, Any]],
    classes: Optional[Sequence[str]] = None,
    reading_dir: str = "rtl",
    group_mode: str = "auto",
    sort_primary: str = "x",
    overlap_thresh: float = 0.35,
    min_char_conf: float = 0.35,
    unknown_token: str = "[?]",
    postprocess: str = "none",
) -> DecodedText:
    """Decode detections into a text string.

    Parameters
    ----------
    detections:
        List of detections with bbox + probs/logits.
    classes:
        Global classes list. If omitted, each detection must include its own.
    reading_dir:
        'rtl' (default) or 'ltr'.
    group_mode:
        'horizontal', 'vertical', or 'auto'.
    sort_primary:
        'x' for line text, 'y' for columns/registers.
    overlap_thresh:
        Overlap threshold for line grouping.
    min_char_conf:
        Confidence threshold below which a token becomes unknown_token.
    unknown_token:
        Marker for low-confidence characters.
    postprocess:
        'none' or 'paleo_basic'.

    Returns
    -------
    DecodedText
    """

    tokens = [
        pick_char(
            det,
            classes=classes,
            min_char_conf=min_char_conf,
            unknown_token=unknown_token,
        )
        for det in detections
    ]

    groups, dbg_group = group_into_lines(tokens, mode=group_mode, overlap_thresh=overlap_thresh)

    # Sort groups themselves (top->bottom for horizontal, left->right for vertical)
    def group_key(g: List[Token]) -> Tuple[float, float]:
        bb = _bbox_union([t.bbox for t in g])
        cx, cy = _bbox_center(bb)
        return (cy, cx) if dbg_group.get("mode") == "horizontal" else (cx, cy)

    groups_sorted = sorted(groups, key=group_key)

    lines: List[DecodedLine] = []
    line_texts: List[str] = []

    for g in groups_sorted:
        sorted_tokens = sort_tokens_in_group(g, reading_dir=reading_dir, sort_primary=sort_primary)
        line_str = join_tokens(sorted_tokens, unknown_token=unknown_token)
        line_bbox = _bbox_union([t.bbox for t in sorted_tokens]) if sorted_tokens else (0, 0, 0, 0)
        lines.append(DecodedLine(tokens=sorted_tokens, bbox=line_bbox))
        line_texts.append(line_str)

    # Combine lines with newline (safe) — caller can later merge differently
    text_raw = "\n".join(line_texts)
    text_final, dbg_post = conservative_cleanup(text_raw, ruleset=postprocess)

    dbg = {
        "grouping": dbg_group,
        "postprocess": dbg_post,
        "reading_dir": reading_dir,
        "sort_primary": sort_primary,
        "n_detections": len(detections),
    }

    return DecodedText(text=text_final, lines=lines, debug=dbg)


# -------------------------
# Simple I/O helpers for common formats
# -------------------------


def load_detections_json(obj: Any) -> List[Dict[str, Any]]:
    """Accept a JSON-loaded object and normalize to a list of detections.

    Supports a few common schemas:
    - list[ {bbox, probs/logits, ...} ]
    - {"detections": [ ... ]}
    - {"instances": [ ... ]}
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return [dict(x) for x in obj]
    if isinstance(obj, dict):
        for k in ("detections", "instances", "preds", "objects"):
            if k in obj and isinstance(obj[k], list):
                return [dict(x) for x in obj[k]]
    raise ValueError("Unsupported detections JSON schema")
