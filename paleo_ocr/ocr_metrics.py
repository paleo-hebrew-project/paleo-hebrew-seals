"""paleo_ocr.ocr_metrics
====================

Lightweight OCR metrics (no heavy deps).

This module is used across the toolkit (baseline engines, experiment runner,
decoder evaluation). Keep it dependency-free and stable.

Conventions
-----------
* :func:`cer` / :func:`wer` accept arguments as ``(ref, pred)``.
* :func:`compute_metrics_for_pairs` expects ``[(ref, pred), ...]``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import re


WS_RE = re.compile(r"\s+")


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    if a == b:
        return 0
    a = list(a)
    b = list(b)
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a
    # Ensure a is shorter
    if len_a > len_b:
        a, b = b, a
        len_a, len_b = len_b, len_a
    prev = list(range(len_a + 1))
    cur = [0] * (len_a + 1)
    for i in range(1, len_b + 1):
        cur[0] = i
        bi = b[i - 1]
        for j in range(1, len_a + 1):
            cost = 0 if a[j - 1] == bi else 1
            cur[j] = min(
                prev[j] + 1,      # del
                cur[j - 1] + 1,   # ins
                prev[j - 1] + cost,  # sub
            )
        prev, cur = cur, prev
    return prev[len_a]


def character_error_rate(pred: str, ref: str, ignore_spaces: bool = False) -> float:
    """CER(pred, ref) = edit_distance(chars) / len(ref)."""
    if ignore_spaces:
        pred = WS_RE.sub("", pred)
        ref = WS_RE.sub("", ref)
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    dist = levenshtein_distance(pred, ref)
    return float(dist) / float(max(1, len(ref)))


def word_error_rate(pred: str, ref: str, ignore_spaces: bool = False) -> float:
    """WER(pred, ref) over whitespace tokenization (or single token if ignore_spaces)."""
    if ignore_spaces:
        pred_toks = [WS_RE.sub("", pred)]
        ref_toks = [WS_RE.sub("", ref)]
    else:
        pred_toks = pred.split()
        ref_toks = ref.split()
    if len(ref_toks) == 0:
        return 0.0 if len(pred_toks) == 0 else 1.0
    dist = levenshtein_distance(pred_toks, ref_toks)
    return float(dist) / float(max(1, len(ref_toks)))


# ---- friendly aliases (ref, pred) order ----


def cer(ref: str, pred: str, ignore_spaces: bool = False) -> float:
    return character_error_rate(pred=pred or "", ref=ref or "", ignore_spaces=ignore_spaces)


def wer(ref: str, pred: str, ignore_spaces: bool = False) -> float:
    return word_error_rate(pred=pred or "", ref=ref or "", ignore_spaces=ignore_spaces)


def _cer_micro(pairs: List[Tuple[str, str]], ignore_spaces: bool = False) -> float:
    num = 0
    den = 0
    for ref, pred in pairs:
        r = ref or ""
        p = pred or ""
        if ignore_spaces:
            r = WS_RE.sub("", r)
            p = WS_RE.sub("", p)
        num += levenshtein_distance(p, r)
        den += len(r)
    return float(num) / float(max(1, den))


def _wer_micro(pairs: List[Tuple[str, str]]) -> float:
    num = 0
    den = 0
    for ref, pred in pairs:
        r_t = (ref or "").split()
        p_t = (pred or "").split()
        num += levenshtein_distance(p_t, r_t)
        den += len(r_t)
    return float(num) / float(max(1, den))


def compute_metrics_for_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    """Compute standard OCR summary metrics for a list of (ref, pred) pairs."""

    n = len(pairs)
    if n == 0:
        return {
            "n_images": 0,
            "exact@img": 0.0,
            "exact_nospace@img": 0.0,
            "CER_macro": 0.0,
            "CER_micro": 0.0,
            "CER_nospace_macro": 0.0,
            "CER_nospace_micro": 0.0,
            "WER_macro": 0.0,
            "WER_micro": 0.0,
        }

    cer_list = [cer(ref, pred) for ref, pred in pairs]
    cer_ns_list = [cer(ref, pred, ignore_spaces=True) for ref, pred in pairs]
    wer_list = [wer(ref, pred) for ref, pred in pairs]

    exact = sum(1 for ref, pred in pairs if (ref or "") == (pred or "")) / float(n)
    exact_ns = sum(1 for ref, pred in pairs if WS_RE.sub("", ref or "") == WS_RE.sub("", pred or "")) / float(n)

    return {
        "n_images": float(n),
        "exact@img": float(exact),
        "exact_nospace@img": float(exact_ns),
        "CER_macro": float(sum(cer_list) / float(n)),
        "CER_micro": float(_cer_micro(pairs, ignore_spaces=False)),
        "CER_nospace_macro": float(sum(cer_ns_list) / float(n)),
        "CER_nospace_micro": float(_cer_micro(pairs, ignore_spaces=True)),
        "WER_macro": float(sum(wer_list) / float(n)),
        "WER_micro": float(_wer_micro(pairs)),
    }
