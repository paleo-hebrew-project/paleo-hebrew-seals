#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""synthetic_v_2_generator (doc-aware + lexicon-aware + randomness + color + final-forms)
========================================================================================
Synthetic dataset generator for Paleo-Hebrew / Phoenician-like seal inscriptions.

What’s new (doc-aware):
- Two text "document kinds":
  * seal  : template-driven epigraphic formulas using names/places/gods lexicons
  * plain : corpus snippets (Torah/Tanakh) with LIGHT text aug, no 'בן/ל' formulas injected
- Template groups + weights:
  --template-group-weights "name:0.75,title:0.20,royal:0.03,blessing:0.01,plain:0.01"
- Places phrases:
  --places-phrases data/places_phrases_hebrew.txt, accessible via {PP} placeholder in templates.
- Backward compatible with old flags:
  --texts / --text-prob still mean "plain corpus" sampling.

NEW (final forms / sofit support):
- Hebrew has 5 final letter forms (ך ם ן ף ץ).
- This generator keeps an internal "22-letter" normalized form (final->normal) for stability,
  but can OUTPUT and/or RENDER text with final forms at word ends:
    --final-forms none|gt|render|both
  Default: gt (store GT with finals; render without finals to keep 22-letter chars/bboxes).
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import math
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm
import numpy as np

try:
    from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont
except Exception as e:
    raise RuntimeError("Pillow is required: pip install pillow") from e


# -------------------------
# Character inventory & templates
# -------------------------

# 22 "base" letters (no finals). We keep this for internal normalization and for random fallbacks.
DEFAULT_HEB_CHARS = list("אבגדהוזחטיכלמנסעפצקרשת")
ALPH_22_SET = set(DEFAULT_HEB_CHARS)

# Hebrew final forms (sofit) used at word end.
FINAL_FORMS = set("ךםןףץ")

# separators/marks (optional, but letters-only will remove them)
MAQAF = "־"
DOT_MID = "·"
DOT_BULLET = "∙"


def build_templates_grouped() -> Dict[str, List[str]]:
    """
    Template groups:
      - name:  ownership/patronymic formulas
      - title: with office/title
      - royal: lmlk / king administration
      - blessing: deity protection/blessing
      - plain: short non-formula tokens (still "seal-ish", no corpus)
    Placeholders:
      {X},{Y},{Z} names
      {G} deity
      {P} place token (no spaces)
      {PP} place phrase (may have spaces)
      {T} title
      {SEP} separator (" " or "")
      {N} digit 1..9
    """
    name = [
        "ל{X}",
        "ל{X}{SEP}בן{SEP}{Y}",
        "ל{X}{SEP}בר{SEP}{Y}",
        "ל{X}\nבן{SEP}{Y}",
        "ל{X}{SEP}בן{SEP}{Y}{SEP}בן{SEP}{Z}",
        "ל{X}{SEP}מ{P}",
        "ל{X}{SEP}בן{SEP}{Y}{SEP}מ{P}",
        "ל{X}{SEP}עבד{SEP}{Y}",
        "ל{X}{SEP}עבד{SEP}המלכ",
        "ל{X}{SEP}בן{SEP}{Y}{SEP}עבד{SEP}המלכ",
    ]

    title = [
        "ל{X}{SEP}{T}",
        "ל{X}\n{T}",
        "ל{X}{SEP}{T}{SEP}בן{SEP}{Y}",
        "ל{X}{SEP}בן{SEP}{Y}{SEP}{T}",
        "ל{X}{SEP}הכהן",
        "ל{X}{SEP}בן{SEP}{Y}{SEP}הכהן",
        "ל{X}{SEP}הסופר",
        "ל{X}{SEP}בן{SEP}{Y}{SEP}הסופר",
        "ל{X}{SEP}שר",
        "ל{X}{SEP}בן{SEP}{Y}{SEP}שר",
        "ל{X}{SEP}נצב",
        "ל{X}{SEP}בן{SEP}{Y}{SEP}נצב",
    ]

    royal = [
        "למלכ",
        "למלכ{SEP}{P}",
        "למלכ\n{P}",
        "למלכ{SEP}{P}{SEP}{N}",
        "למלכ{SEP}{PP}",
        "למלכ\n{PP}",
        "ל{X}{SEP}שר{SEP}המלכ",
        "ל{X}{SEP}בן{SEP}{Y}{SEP}שר{SEP}המלכ",
    ]

    blessing = [
        "ברכ{SEP}{X}",
        "{G}{SEP}שמר{SEP}{X}",
        "{G}{SEP}ברכ{SEP}{X}",
        "{G}\nשמר{SEP}{X}",
        "{G}{SEP}שמר\n{X}",
    ]

    plain = [
        "{X}",
        "{X}{SEP}{Y}",
        "{X}\n{Y}",
        "{G}",
        "{P}",
        "{PP}",
        "בן{SEP}{X}",
        "עבד{SEP}{X}",
        "הכהן",
        "הסופר",
        "המלכ",
    ]

    return {"name": name, "title": title, "royal": royal, "blessing": blessing, "plain": plain}


def build_templates() -> List[str]:
    """Backward-compatible: flat list"""
    groups = build_templates_grouped()
    out: List[str] = []
    seen = set()
    for k in ["name", "title", "royal", "blessing", "plain"]:
        for t in groups[k]:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _parse_kv_weights(s: str, keys: List[str]) -> Optional[List[float]]:
    """
    Parse: "name:0.7,title:0.2,royal:0.05,blessing:0.03,plain:0.02"
    Missing keys => 0.
    """
    if not s:
        return None
    raw = {}
    for part in s.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        try:
            raw[k] = float(v.strip())
        except Exception:
            pass
    w = [max(0.0, float(raw.get(k, 0.0))) for k in keys]
    return w if sum(w) > 0 else None


# -------------------------
# Hebrew normalization + final forms helpers
# -------------------------

# final -> normal (22-letter canonicalization)
FINAL_TO_NORMAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}

# normal -> final (apply at word ends)
NORMAL_TO_FINAL = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}

_HEB_DIACRITICS_RE = re.compile(r"[\u0591-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C5\u05C7]")
_HEB_RUN_RE = re.compile(r"[\u05D0-\u05EA]+")  # includes finals too (they are inside this range)


def apply_final_forms_end(text: str, *, min_len: int = 2) -> str:
    """
    Convert last letter of each Hebrew run to sofit if:
      - run length >= min_len
      - last letter is one of {כ,מ,נ,פ,צ}
    If the run already ends in a final form, it stays unchanged (since last letter won't match NORMAL_TO_FINAL keys).
    """
    s = text or ""

    def repl(m: re.Match) -> str:
        w = m.group(0)
        if not w:
            return w
        if len(w) < int(min_len):
            return w
        last = w[-1]
        return w[:-1] + NORMAL_TO_FINAL[last] if last in NORMAL_TO_FINAL else w

    return _HEB_RUN_RE.sub(repl, s)


def normalize_hebrew_for_seals(
    s: str,
    keep_marks: bool = True,
    allow_newlines: bool = True,
    letters_only: bool = False,
    *,
    map_finals_to_normal: bool = True,
) -> str:
    """
    - Strips niqqud/cantillation.
    - Optionally maps final forms -> normal (22-letter).
    - Keeps:
        - always: Hebrew letters + spaces
        - optionally: newlines
        - optionally (if keep_marks): · ∙ ־
      If letters_only=True -> removes all marks/punctuation (keep_marks forced off).
    """
    s = s or ""
    s = _HEB_DIACRITICS_RE.sub("", s)
    if map_finals_to_normal:
        s = "".join(FINAL_TO_NORMAL.get(ch, ch) for ch in s)

    if letters_only:
        keep_marks = False

    allowed = set(DEFAULT_HEB_CHARS) | FINAL_FORMS  # allow finals if present
    allowed.add(" ")
    if allow_newlines:
        allowed.add("\n")
    if keep_marks:
        allowed |= {DOT_MID, DOT_BULLET, MAQAF}

    out = []
    for ch in s:
        out.append(ch if ch in allowed else " ")

    txt = "".join(out)
    txt = re.sub(r"[ \t\r\f\v]+", " ", txt)
    txt = re.sub(r" *\n *", "\n", txt).strip()
    return txt


def inject_intra_word_breaks(
    text: str,
    prob: float,
    break_char: str = " ",
    min_token_len: int = 3,
) -> str:
    """Rarely split a word into two parts by inserting break_char inside it."""
    if prob <= 0:
        return text

    bc = "\n" if break_char in ("\\n", "NEWLINE") else break_char
    tokens = re.split(r"(\s+)", text)  # keep whitespace chunks
    for i in range(0, len(tokens), 2):
        tok = tokens[i]
        if "\n" in tok:
            continue
        if len(tok) >= min_token_len and random.random() < prob:
            cut = random.randint(1, len(tok) - 1)
            tokens[i] = tok[:cut] + bc + tok[cut:]
    return "".join(tokens)


def inject_inter_word_newline(text: str, prob: float) -> str:
    """Replace one random space with newline with given probability."""
    if prob <= 0:
        return text
    if "\n" in text:
        return text
    if " " not in text:
        return text
    if random.random() >= prob:
        return text
    parts = text.split(" ")
    if len(parts) < 2:
        return text
    cut = random.randint(1, len(parts) - 1)
    return " ".join(parts[:cut]) + "\n" + " ".join(parts[cut:])


# -------------------------
# Lexicon loaders & samplers
# -------------------------

def _load_lines(paths: Optional[List[str]]) -> List[str]:
    if not paths:
        return []
    items: List[str] = []
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            print(f"[WARN] lexicon not found: {fp}", file=sys.stderr)
            continue
        for line in fp.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def _explode_tokens(lines: List[str]) -> List[str]:
    """For token lexicons: if a line has spaces, split into tokens."""
    out: List[str] = []
    for s in lines:
        parts = re.split(r"\s+", s.strip())
        for p in parts:
            pp = p.strip()
            if pp:
                out.append(pp)
    return out


def _sanitize_token_list(lines: List[str], letters_only: bool) -> List[str]:
    """Normalize each line/token to 22-letter token (no spaces)."""
    out: List[str] = []
    for s in _explode_tokens(lines):
        t = normalize_hebrew_for_seals(
            s,
            keep_marks=False,
            allow_newlines=False,
            letters_only=letters_only,
            map_finals_to_normal=True,  # keep lexicons canonical in 22-letter form
        )
        t = t.replace(" ", "")
        if not t:
            continue
        # drop trivial glue words
        if t in {"בן", "בנ", "בת", "בר", "עבד", "שר", "כהן", "סופר", "נצב"}:
            continue
        out.append(t)
    return out


def _sanitize_phrase_list(lines: List[str], letters_only: bool, max_tokens: int = 3) -> List[str]:
    """Normalize each line as a phrase of up to max_tokens tokens (space-separated)."""
    out: List[str] = []
    for s in lines:
        t = normalize_hebrew_for_seals(
            s,
            keep_marks=False,
            allow_newlines=False,
            letters_only=letters_only,
            map_finals_to_normal=True,  # canonicalize in 22-letter form; finals will be re-added later if desired
        )
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            continue
        toks = t.split()
        if 1 <= len(toks) <= max_tokens:
            out.append(" ".join(toks))
    return out


class ListSampler:
    def __init__(self, items: List[str], fallback_chars: Sequence[str]):
        self.items = [x for x in items if x]
        self.fallback_chars = fallback_chars

    def sample(self, min_len: int = 3, max_len: int = 6) -> str:
        if self.items:
            return random.choice(self.items)
        # fallback random token
        L = random.randint(min_len, max_len)
        return "".join(random.choice(self.fallback_chars) for _ in range(L))


class PhraseSampler:
    def __init__(self, phrases: List[str], token_sampler: ListSampler):
        self.phrases = [x for x in phrases if x]
        self.token_sampler = token_sampler

    def sample(self) -> str:
        if self.phrases:
            return random.choice(self.phrases)
        # fallback single token
        return self.token_sampler.sample(2, 6)


# -------------------------
# Text corpus (plain documents)
# -------------------------

class TextCorpus:
    def __init__(self, paths: List[str], keep_marks: bool, letters_only: bool):
        self.items: List[str] = []
        for p in paths or []:
            fp = Path(p)
            if not fp.exists():
                print(f"[WARN] corpus file not found: {fp}", file=sys.stderr)
                continue
            for line in fp.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                norm = normalize_hebrew_for_seals(
                    line,
                    keep_marks=keep_marks,
                    allow_newlines=False,
                    letters_only=letters_only,
                    map_finals_to_normal=True,  # canonicalize; finals may be re-added later
                )
                if norm:
                    self.items.append(norm)

    def sample(self, min_chars: int, max_chars: int) -> Optional[str]:
        if not self.items:
            return None
        for _ in range(80):
            s = random.choice(self.items)
            if min_chars <= len(s) <= max_chars:
                return s
            if len(s) > max_chars:
                # take a window by words
                words = s.split()
                if not words:
                    continue
                start = random.randrange(0, len(words))
                cur: List[str] = []
                j = start
                while j < len(words) and len(" ".join(cur + [words[j]])) <= max_chars:
                    cur.append(words[j])
                    if len(" ".join(cur)) >= min_chars and random.random() < 0.35:
                        break
                    j += 1
                out = " ".join(cur).strip()
                if min_chars <= len(out) <= max_chars:
                    return out
        return None


def plain_text_augment(
    text: str,
    *,
    letters_only: bool,
    keep_marks: bool,
    join_all_prob: float,
    drop_some_spaces_prob: float,
    sep_maqaf_prob: float,
    sep_dot_prob: float,
) -> str:
    """
    Plain-doc augmentation (does NOT inject seal formulas).
    Operates mostly on whitespace/separators to increase variety.
    """
    out = text

    # Replace spaces with separators only if allowed
    if (not letters_only) and keep_marks:
        r = random.random()
        if r < sep_maqaf_prob:
            out = out.replace(" ", MAQAF)
        elif r < sep_maqaf_prob + sep_dot_prob:
            out = out.replace(" ", DOT_MID)

    # Drop some spaces
    if random.random() < drop_some_spaces_prob:
        chars = []
        for ch in out:
            if ch == " " and random.random() < 0.35:
                continue
            chars.append(ch)
        out = "".join(chars)

    # Join all
    if random.random() < join_all_prob:
        out = out.replace(" ", "")

    out = normalize_hebrew_for_seals(
        out,
        keep_marks=keep_marks,
        allow_newlines=False,
        letters_only=letters_only,
        map_finals_to_normal=True,
    )
    return out


# -------------------------
# Renderer Utilities
# -------------------------

def _normalize_range(min_val: float, max_val: float) -> Tuple[float, float]:
    low, high = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
    return low, high


def _rand_gray(gray_range: Tuple[int, int]) -> int:
    low, high = _normalize_range(int(gray_range[0]), int(gray_range[1]))
    return random.randint(int(low), int(high))


def _rand_contrast(contrast_range: Tuple[float, float]) -> float:
    low, high = _normalize_range(float(contrast_range[0]), float(contrast_range[1]))
    if low == high:
        return float(low)
    return random.uniform(float(low), float(high))


def _gray_color(gray: int) -> Tuple[int, int, int]:
    return (gray, gray, gray)


def _clamp_u8(x: int) -> int:
    return 0 if x < 0 else 255 if x > 255 else int(x)


def _apply_morphology_to_alpha(img_rgba: Image.Image, delta: float, power: float = 1.0) -> Image.Image:
    """Alpha dilation/erosion using MaxFilter/MinFilter."""
    if delta == 0:
        return img_rgba
    r, g, b, a = img_rgba.split()
    steps = int(math.ceil(abs(delta)))
    filt = ImageFilter.MaxFilter(3) if delta > 0 else ImageFilter.MinFilter(3)
    for _ in range(steps):
        a = a.filter(filt)
    return Image.merge("RGBA", (r, g, b, a))


@dataclass
class RenderConfig:
    canvas_size: Tuple[int, int] = (512, 512)
    margin: int = 40
    line_spacing: int = 14
    char_spacing: int = 6

    x_random: bool = False
    y_random: bool = False

    rotate_deg: float = 0.0
    shear: float = 0.0

    blur_radius: float = 0.0
    noise_std: float = 0.0

    stroke_width: float = 0.0  # positive: stroke in draw.text; negative: erosion via alpha morphology

    glyph_scale_range: Tuple[float, float] = (1.0, 1.0)
    glyph_scale_std: float = 0.0
    glyph_rotate_range: Tuple[float, float] = (0.0, 0.0)

    # grayscale fallback
    bg_gray_range: Tuple[int, int] = (200, 255)
    ink_gray_range: Tuple[int, int] = (0, 40)
    contrast_range: Tuple[float, float] = (1.0, 1.0)
    ink_gray_jitter: int = 4

    # optional per-example bg/ink RGB
    bg_rgb_fixed: Optional[Tuple[int, int, int]] = None
    ink_rgb_fixed: Optional[Tuple[int, int, int]] = None
    ink_rgb_jitter: int = 8

    color_enhance: float = 1.0

    depth_strength: float = 0.0
    depth_blur: float = 1.5
    depth_offset: int = 1

    taper_strength: float = 0.0  # if >0 => probability apply morph; if 0 => always (old)
    taper_delta: float = 0.0
    taper_power: float = 1.0


def render_text_glyph_by_glyph(
    text: str,
    font: ImageFont.FreeTypeFont,
    cfg: RenderConfig,
    rtl: bool = True,
    glyph_scale_mean: Optional[float] = None,
) -> Tuple[Optional[Image.Image], List[str], List[List[int]], Optional[float]]:

    W, H = cfg.canvas_size

    # Background
    if cfg.bg_rgb_fixed is not None:
        bg_color = tuple(int(x) for x in cfg.bg_rgb_fixed)
    else:
        bg_gray = _rand_gray(cfg.bg_gray_range)
        bg_color = _gray_color(bg_gray)

    img_bg = Image.new("RGB", (W, H), bg_color)
    txt_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    lines = text.split("\n")
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + cfg.line_spacing
    text_h = len(lines) * line_h

    if text_h > (H - 2 * cfg.margin):
        return None, [], [], None

    if cfg.y_random and (H - 2 * cfg.margin - text_h) > 0:
        y = random.randint(cfg.margin, H - cfg.margin - text_h)
    else:
        y = cfg.margin

    chars_out: List[str] = []
    boxes_out: List[List[int]] = []

    font_cache: Dict[int, ImageFont.FreeTypeFont] = {int(getattr(font, "size", 0)): font}

    def _font_variant(font_obj, size):
        if size in font_cache:
            return font_cache[size]
        try:
            v = font_obj.font_variant(size=int(size))
        except Exception:
            v = font_obj
        font_cache[size] = v
        return v

    def _sample_scale():
        low, high = _normalize_range(cfg.glyph_scale_range[0], cfg.glyph_scale_range[1])
        if cfg.glyph_scale_std > 0 and glyph_scale_mean is not None:
            s = random.gauss(float(glyph_scale_mean), float(cfg.glyph_scale_std))
        elif low != high:
            s = random.uniform(float(low), float(high))
        else:
            s = float(low)
        return max(float(low), min(float(high), float(s)))

    def _sample_rotate():
        low, high = _normalize_range(cfg.glyph_rotate_range[0], cfg.glyph_rotate_range[1])
        return random.uniform(float(low), float(high)) if low != high else float(low)

    def _get_glyph_metrics(ch, f):
        try:
            adv = float(f.getlength(ch))
        except Exception:
            adv = float(f.getsize(ch)[0])
        try:
            x0, y0, x1, y1 = f.getbbox(ch)
        except Exception:
            sw, sh = f.getsize(ch)
            x0, x1, y0, y1 = 0, sw, 0, sh
        return adv, x0, x1, y0, y1

    max_w = W - 2 * cfg.margin
    ink_fixed = cfg.ink_rgb_fixed

    for line in lines:
        base_size = int(getattr(font, "size", 0) or 0)
        line_glyphs: List[Tuple[str, ImageFont.FreeTypeFont]] = []

        for ch in list(line):
            sc = _sample_scale()
            sz = max(1, int(round(base_size * sc)))
            f_var = _font_variant(font, sz)
            line_glyphs.append((ch, f_var))

        total_w = 0.0
        for ch, f_var in line_glyphs:
            adv, _, _, _, _ = _get_glyph_metrics(ch, f_var)
            total_w += (adv + cfg.char_spacing)
        total_w -= cfg.char_spacing

        if total_w > max_w:
            return None, [], [], None

        low = cfg.margin
        high = (W - cfg.margin) - total_w
        if high < low:
            return None, [], [], None

        anchor_x = random.randint(int(low), int(high)) if cfg.x_random else int(high if rtl else low)
        baseline = y + ascent
        pen_x = anchor_x + total_w if rtl else anchor_x

        for ch, f_var in line_glyphs:
            adv, gx0, gx1, gy0, gy1 = _get_glyph_metrics(ch, f_var)

            if ch == " ":
                step = adv + cfg.char_spacing
                pen_x = pen_x - step if rtl else pen_x + step
                continue

            draw_x = pen_x - adv if rtl else pen_x
            draw_y = baseline - ascent

            glyph_w = int(gx1 - gx0)
            glyph_h = int(gy1 - gy0)
            if glyph_w <= 0:
                glyph_w = int(adv)
            if glyph_h <= 0:
                glyph_h = ascent

            pad = int(max(W, H) * 0.1) + 20
            temp_w, temp_h = glyph_w + 2 * pad, glyph_h + 2 * pad

            glyph_img = Image.new("RGBA", (temp_w, temp_h), (0, 0, 0, 0))
            g_draw = ImageDraw.Draw(glyph_img)

            # Ink color
            if ink_fixed is not None:
                br, bg, bb = ink_fixed
                j = int(cfg.ink_rgb_jitter)
                r = _clamp_u8(int(br) + random.randint(-j, j))
                g = _clamp_u8(int(bg) + random.randint(-j, j))
                b = _clamp_u8(int(bb) + random.randint(-j, j))
                ink_rgba = (r, g, b, 255)
            else:
                base_gray = _rand_gray(cfg.ink_gray_range)
                jit = random.randint(-cfg.ink_gray_jitter, cfg.ink_gray_jitter)
                ink_val = _clamp_u8(int(base_gray) + int(jit))
                ink_rgba = (ink_val, ink_val, ink_val, 255)

            stroke_int = int(max(0.0, math.floor(cfg.stroke_width)))
            draw_stroke = stroke_int if stroke_int > 0 else 0

            origin_in_temp_x = pad - gx0
            origin_in_temp_y = pad - gy0

            g_draw.text(
                (origin_in_temp_x, origin_in_temp_y),
                ch,
                font=f_var,
                fill=ink_rgba,
                stroke_width=draw_stroke,
                stroke_fill=ink_rgba,
            )

            rot_angle = _sample_rotate()
            if abs(rot_angle) > 0.1:
                vis_cx = origin_in_temp_x + (gx0 + gx1) / 2
                vis_cy = origin_in_temp_y + (gy0 + gy1) / 2
                glyph_img = glyph_img.rotate(
                    rot_angle,
                    resample=Image.BICUBIC,
                    expand=False,
                    center=(vis_cx, vis_cy),
                )

            paste_x = int(draw_x - origin_in_temp_x)
            paste_y = int(draw_y - origin_in_temp_y)
            txt_layer.paste(glyph_img, (paste_x, paste_y), glyph_img)

            bbox = glyph_img.getbbox()
            if bbox:
                b_x0, b_y0, b_x1, b_y1 = bbox
                glob_x0 = max(0, min(W, b_x0 + paste_x))
                glob_y0 = max(0, min(H, b_y0 + paste_y))
                glob_x1 = max(0, min(W, b_x1 + paste_x))
                glob_y1 = max(0, min(H, b_y1 + paste_y))
                chars_out.append(ch)
                boxes_out.append([glob_x0, glob_y0, glob_x1, glob_y1])

            step = adv + cfg.char_spacing
            pen_x = pen_x - step if rtl else pen_x + step

        y += line_h

    # Morphology (taper + negative stroke)
    need_morph = (cfg.taper_delta != 0) or (cfg.stroke_width < 0)
    if need_morph:
        effective_delta = float(cfg.taper_delta) + (float(cfg.stroke_width) if cfg.stroke_width < 0 else 0.0)
        if abs(effective_delta) > 1e-9:
            prob = float(cfg.taper_strength)
            apply_prob = prob if prob > 0 else 1.0
            if random.random() < apply_prob:
                txt_layer = _apply_morphology_to_alpha(txt_layer, effective_delta, power=float(cfg.taper_power))

    img_final = Image.alpha_composite(img_bg.convert("RGBA"), txt_layer).convert("RGB")

    # Depth/emboss
    if cfg.depth_strength > 0:
        grayscale = img_final.convert("L")
        blurred = grayscale.filter(ImageFilter.GaussianBlur(radius=float(cfg.depth_blur)))

        off = int(cfg.depth_offset)
        highlight = ImageChops.offset(blurred, -off, -off)
        shadow = ImageChops.offset(blurred, off, off)

        arr = np.array(img_final).astype(np.float32)
        h_arr = np.array(highlight).astype(np.float32)
        s_arr = np.array(shadow).astype(np.float32)

        amt = float(cfg.depth_strength)
        diff = (h_arr - s_arr) * (amt / 255.0)

        arr = arr + diff[..., None] * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img_final = Image.fromarray(arr, mode="RGB")

    # Blur/Noise
    if cfg.blur_radius > 0:
        img_final = img_final.filter(ImageFilter.GaussianBlur(radius=float(cfg.blur_radius)))

    if cfg.noise_std > 0:
        arr = np.array(img_final).astype(np.float32)
        noise = np.random.normal(0, float(cfg.noise_std), size=arr.shape).astype(np.float32)
        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img_final = Image.fromarray(arr, mode="RGB")

    # Contrast
    contrast_factor = _rand_contrast(cfg.contrast_range)
    if contrast_factor != 1.0:
        img_final = ImageEnhance.Contrast(img_final).enhance(float(contrast_factor))

    # Saturation
    if float(cfg.color_enhance) != 1.0:
        img_final = ImageEnhance.Color(img_final).enhance(float(cfg.color_enhance))

    # Global rotation (optional)
    if cfg.rotate_deg != 0:
        img_final = img_final.rotate(
            cfg.rotate_deg,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=bg_color,
        )

    return img_final, chars_out, boxes_out, float(contrast_factor)


# -------------------------
# IO
# -------------------------

def _atomic_save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def write_example(
    out_root: Path,
    uid: str,
    img: Image.Image,
    *,
    text_render: str,
    text_gt: str,
    text_norm: str,
    chars: List[str],
    bboxes: List[List[int]],
    meta: Dict[str, Any],
) -> None:
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)
    img_path = out_root / "images" / f"{uid}.png"
    lbl_path = out_root / "labels" / f"{uid}.json"
    _atomic_save_png(img, img_path)

    # Backward compatible: "text" matches rendered chars/bboxes
    lbl = {
        "uid": uid,
        "text": text_render,
        "text_gt": text_gt,
        "text_norm": text_norm,
        "chars": chars,
        "bboxes": bboxes,
        "meta": meta,
    }
    with open(lbl_path, "w", encoding="utf-8") as f:
        json.dump(lbl, f, ensure_ascii=False)


def append_manifest(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------------
# Sampling helpers for per-example randomness + color
# -------------------------

def _sample_float(base: float, vmin: Optional[float], vmax: Optional[float]) -> float:
    if vmin is None and vmax is None:
        return float(base)
    if vmin is None:
        vmin = float(base)
    if vmax is None:
        vmax = float(base)
    lo, hi = _normalize_range(float(vmin), float(vmax))
    if lo == hi:
        return float(lo)
    return float(random.uniform(lo, hi))


def _sample_int(base: int, vmin: Optional[int], vmax: Optional[int]) -> int:
    if vmin is None and vmax is None:
        return int(base)
    if vmin is None:
        vmin = int(base)
    if vmax is None:
        vmax = int(base)
    lo, hi = (vmin, vmax) if vmin <= vmax else (vmax, vmin)
    if lo == hi:
        return int(lo)
    return int(random.randint(int(lo), int(hi)))


def _has_any(vals: List[Optional[int]]) -> bool:
    return any(v is not None for v in vals)


def _range_or_fallback(vmin: Optional[int], vmax: Optional[int], fallback_min: int, fallback_max: int) -> Tuple[int, int]:
    if vmin is None and vmax is None:
        return int(fallback_min), int(fallback_max)
    if vmin is None:
        vmin = vmax
    if vmax is None:
        vmax = vmin
    lo, hi = (int(vmin), int(vmax)) if int(vmin) <= int(vmax) else (int(vmax), int(vmin))
    return lo, hi


def sample_cfg_for_example(cfg: RenderConfig, args: argparse.Namespace) -> Tuple[RenderConfig, Dict[str, Any]]:
    stroke = _sample_float(args.stroke, args.stroke_min, args.stroke_max)
    blur = _sample_float(args.blur, args.blur_min, args.blur_max)
    noise = _sample_float(args.noise, args.noise_min, args.noise_max)

    taper_delta = _sample_float(args.taper_delta, args.taper_delta_min, args.taper_delta_max)

    depth_strength = _sample_float(args.depth_strength, args.depth_strength_min, args.depth_strength_max)
    depth_blur = _sample_float(args.depth_blur, args.depth_blur_min, args.depth_blur_max)
    depth_offset = _sample_int(args.depth_offset, args.depth_offset_min, args.depth_offset_max)

    color_enh = _sample_float(args.color_enhance, args.color_enhance_min, args.color_enhance_max)

    blur = max(0.0, blur)
    noise = max(0.0, noise)
    depth_strength = max(0.0, depth_strength)
    depth_blur = max(0.0, depth_blur)
    depth_offset = max(0, depth_offset)
    color_enh = max(0.0, color_enh)

    # per-example bg RGB
    bg_rgb_fixed: Optional[Tuple[int, int, int]] = None
    if _has_any([args.bg_r_min, args.bg_r_max, args.bg_g_min, args.bg_g_max, args.bg_b_min, args.bg_b_max]):
        r0, r1 = _range_or_fallback(args.bg_r_min, args.bg_r_max, args.bg_gray_min, args.bg_gray_max)
        g0, g1 = _range_or_fallback(args.bg_g_min, args.bg_g_max, args.bg_gray_min, args.bg_gray_max)
        b0, b1 = _range_or_fallback(args.bg_b_min, args.bg_b_max, args.bg_gray_min, args.bg_gray_max)
        bg_rgb_fixed = (random.randint(r0, r1), random.randint(g0, g1), random.randint(b0, b1))

    # per-example ink RGB
    ink_rgb_fixed: Optional[Tuple[int, int, int]] = None
    if _has_any([args.ink_r_min, args.ink_r_max, args.ink_g_min, args.ink_g_max, args.ink_b_min, args.ink_b_max]):
        r0, r1 = _range_or_fallback(args.ink_r_min, args.ink_r_max, args.ink_gray_min, args.ink_gray_max)
        g0, g1 = _range_or_fallback(args.ink_g_min, args.ink_g_max, args.ink_gray_min, args.ink_gray_max)
        b0, b1 = _range_or_fallback(args.ink_b_min, args.ink_b_max, args.ink_gray_min, args.ink_gray_max)
        ink_rgb_fixed = (random.randint(r0, r1), random.randint(g0, g1), random.randint(b0, b1))

    cfg2 = replace(
        cfg,
        stroke_width=float(stroke),
        blur_radius=float(blur),
        noise_std=float(noise),
        taper_delta=float(taper_delta),
        depth_strength=float(depth_strength),
        depth_blur=float(depth_blur),
        depth_offset=int(depth_offset),
        color_enhance=float(color_enh),
        bg_rgb_fixed=bg_rgb_fixed,
        ink_rgb_fixed=ink_rgb_fixed,
        ink_rgb_jitter=int(args.ink_rgb_jitter),
    )

    sampled = {
        "stroke_width": float(stroke),
        "blur_radius": float(blur),
        "noise_std": float(noise),
        "taper_delta": float(taper_delta),
        "taper_strength": float(cfg2.taper_strength),
        "taper_power": float(cfg2.taper_power),
        "depth_strength": float(depth_strength),
        "depth_blur": float(depth_blur),
        "depth_offset": int(depth_offset),
        "color_enhance": float(color_enh),
        "bg_rgb_fixed": bg_rgb_fixed,
        "ink_rgb_fixed": ink_rgb_fixed,
        "ink_rgb_jitter": int(args.ink_rgb_jitter),
    }
    return cfg2, sampled


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--uid-prefix", type=str, default="syn")
    p.add_argument("--fonts", type=str, nargs="+", required=True)
    p.add_argument("--font-size", type=int, default=84)
    p.add_argument("--main-font", type=str, default=None)
    p.add_argument("--main-font-weight", type=float, default=1.0)
    p.add_argument("--canvas", type=int, nargs=2, default=[512, 512])
    p.add_argument("--margin", type=int, default=40)
    p.add_argument("--rtl", action="store_true")
    p.add_argument("--x-random", action="store_true")
    p.add_argument("--y-random", action="store_true")

    # Fixed values (back-compat)
    p.add_argument("--stroke", type=float, default=0.0)
    p.add_argument("--blur", type=float, default=0.0)
    p.add_argument("--noise", type=float, default=0.0)

    # per-sample ranges
    p.add_argument("--stroke-min", type=float, default=None)
    p.add_argument("--stroke-max", type=float, default=None)
    p.add_argument("--blur-min", type=float, default=None)
    p.add_argument("--blur-max", type=float, default=None)
    p.add_argument("--noise-min", type=float, default=None)
    p.add_argument("--noise-max", type=float, default=None)

    # grayscale fallback
    p.add_argument("--bg-gray-min", type=int, default=200)
    p.add_argument("--bg-gray-max", type=int, default=255)
    p.add_argument("--ink-gray-min", type=int, default=0)
    p.add_argument("--ink-gray-max", type=int, default=40)

    # optional RGB background ranges
    p.add_argument("--bg-r-min", type=int, default=None)
    p.add_argument("--bg-r-max", type=int, default=None)
    p.add_argument("--bg-g-min", type=int, default=None)
    p.add_argument("--bg-g-max", type=int, default=None)
    p.add_argument("--bg-b-min", type=int, default=None)
    p.add_argument("--bg-b-max", type=int, default=None)

    # optional RGB ink ranges
    p.add_argument("--ink-r-min", type=int, default=None)
    p.add_argument("--ink-r-max", type=int, default=None)
    p.add_argument("--ink-g-min", type=int, default=None)
    p.add_argument("--ink-g-max", type=int, default=None)
    p.add_argument("--ink-b-min", type=int, default=None)
    p.add_argument("--ink-b-max", type=int, default=None)
    p.add_argument("--ink-rgb-jitter", type=int, default=10)

    p.add_argument("--contrast-min", type=float, default=1.0)
    p.add_argument("--contrast-max", type=float, default=1.0)

    # saturation
    p.add_argument("--color-enhance", type=float, default=1.0)
    p.add_argument("--color-enhance-min", type=float, default=None)
    p.add_argument("--color-enhance-max", type=float, default=None)

    # depth
    p.add_argument("--depth-strength", type=float, default=0.0)
    p.add_argument("--depth-blur", type=float, default=1.5)
    p.add_argument("--depth-offset", type=int, default=1)
    p.add_argument("--depth-strength-min", type=float, default=None)
    p.add_argument("--depth-strength-max", type=float, default=None)
    p.add_argument("--depth-blur-min", type=float, default=None)
    p.add_argument("--depth-blur-max", type=float, default=None)
    p.add_argument("--depth-offset-min", type=int, default=None)
    p.add_argument("--depth-offset-max", type=int, default=None)

    # taper/morph
    p.add_argument("--taper-strength", type=float, default=0.0)
    p.add_argument("--taper-delta", type=float, default=0.0)
    p.add_argument("--taper-power", type=float, default=1.0)
    p.add_argument("--taper-delta-min", type=float, default=None)
    p.add_argument("--taper-delta-max", type=float, default=None)

    # glyph-level randomness
    p.add_argument("--glyph-scale-min", type=float, default=1.0)
    p.add_argument("--glyph-scale-max", type=float, default=1.0)
    p.add_argument("--glyph-scale-std", type=float, default=0.0)
    p.add_argument("--glyph-rotate-min", type=float, default=0.0)
    p.add_argument("--glyph-rotate-max", type=float, default=0.0)

    # Lexicons
    p.add_argument("--names", type=str, nargs="*", default=None)
    p.add_argument("--places", type=str, nargs="*", default=None)
    p.add_argument("--gods", type=str, nargs="*", default=None)
    p.add_argument("--places-phrases", type=str, nargs="*", default=None)

    # Corpus texts (plain docs) — backward compatible aliases
    p.add_argument("--texts", type=str, nargs="*", default=None, help="(alias) plain corpus txts")
    p.add_argument("--plain-texts", type=str, nargs="*", default=None, help="plain corpus txts (one snippet per line)")
    p.add_argument("--text-prob", type=float, default=0.0, help="(alias) probability to use plain corpus")
    p.add_argument("--plain-prob", type=float, default=None, help="probability to use plain corpus (overrides --text-prob)")
    p.add_argument("--text-min-chars", type=int, default=6)
    p.add_argument("--text-max-chars", type=int, default=20)

    # Doc-kind mix override (optional)
    p.add_argument("--doc-probs", type=str, default="", help='e.g. "seal:0.6,plain:0.4"')

    # Plain augment (does not inject formulas)
    p.add_argument("--plain-augment", action="store_true", default=False)
    p.add_argument("--plain-join-all-prob", type=float, default=0.12)
    p.add_argument("--plain-drop-spaces-prob", type=float, default=0.20)
    p.add_argument("--plain-sep-maqaf-prob", type=float, default=0.15)
    p.add_argument("--plain-sep-dot-prob", type=float, default=0.10)
    p.add_argument("--plain-inter-word-newline-prob", type=float, default=0.0)

    # Seal template controls
    p.add_argument("--template-group-weights", type=str, default="", help='e.g. "name:0.75,title:0.20,royal:0.03,blessing:0.01,plain:0.01"')

    # Rare intra-word breaks (both doc kinds)
    p.add_argument("--intra-word-break-prob", type=float, default=0.0)
    p.add_argument("--intra-word-break-char", type=str, default=" ", help="space | '·' | '־' | '\\n'")

    # Normalization flags
    p.add_argument("--keep-marks", type=int, default=1)
    p.add_argument("--letters-only", action="store_true")

    # Final forms / sofit output mode:
    # none   -> no finals anywhere (pure 22-letter strings)
    # gt     -> store GT text with finals; render text stays 22-letter (recommended for 22-class chars/bboxes)
    # render -> render with finals; GT stays 22-letter
    # both   -> render + GT with finals
    p.add_argument("--final-forms", type=str, default="gt", choices=["none", "gt", "render", "both"])
    p.add_argument("--final-forms-min-len", type=int, default=2)

    return p.parse_args()


# -------------------------
# Main
# -------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    # Fonts
    fonts: List[Tuple[str, ImageFont.FreeTypeFont]] = []
    for fp in args.fonts:
        try:
            fonts.append((str(fp), ImageFont.truetype(str(fp), size=int(args.font_size))))
        except Exception as e:
            print(f"Skip {fp}: {e}", file=sys.stderr)
    if not fonts:
        raise RuntimeError("No fonts loaded")

    font_paths = [fp for fp, _ in fonts]
    main_font = str(args.main_font) if args.main_font else None
    weights = [float(args.main_font_weight) if main_font and fp == main_font else 1.0 for fp in font_paths]

    # Config
    base_cfg = RenderConfig(
        canvas_size=(args.canvas[0], args.canvas[1]),
        margin=args.margin,
        stroke_width=float(args.stroke),
        blur_radius=float(args.blur),
        noise_std=float(args.noise),
        bg_gray_range=(args.bg_gray_min, args.bg_gray_max),
        ink_gray_range=(args.ink_gray_min, args.ink_gray_max),
        contrast_range=(args.contrast_min, args.contrast_max),
        depth_strength=float(args.depth_strength),
        depth_blur=float(args.depth_blur),
        depth_offset=int(args.depth_offset),
        taper_strength=float(args.taper_strength),
        taper_delta=float(args.taper_delta),
        taper_power=float(args.taper_power),
        glyph_scale_range=(args.glyph_scale_min, args.glyph_scale_max),
        glyph_scale_std=float(args.glyph_scale_std),
        glyph_rotate_range=(args.glyph_rotate_min, args.glyph_rotate_max),
        ink_rgb_jitter=int(args.ink_rgb_jitter),
        color_enhance=float(args.color_enhance),
    )
    base_cfg.x_random = bool(args.x_random)
    base_cfg.y_random = bool(args.y_random)

    letters_only = bool(args.letters_only)
    keep_marks = bool(int(args.keep_marks)) and (not letters_only)

    # enforce intra-word break char in letters-only mode
    break_char = args.intra_word_break_char
    if letters_only and break_char not in (" ", "\\n", "NEWLINE"):
        break_char = " "

    # Load lexicons
    name_items = _sanitize_token_list(_load_lines(args.names), letters_only=letters_only)
    place_items = _sanitize_token_list(_load_lines(args.places), letters_only=letters_only)
    god_items = _sanitize_token_list(_load_lines(args.gods), letters_only=letters_only)
    place_phrases = _sanitize_phrase_list(_load_lines(args.places_phrases), letters_only=letters_only, max_tokens=3)

    name_s = ListSampler(name_items, DEFAULT_HEB_CHARS)
    place_s = ListSampler(place_items, DEFAULT_HEB_CHARS)
    god_s = ListSampler(god_items, DEFAULT_HEB_CHARS)
    place_phrase_s = PhraseSampler(place_phrases, place_s)

    # Corpus (plain texts) — accept both flags
    plain_paths = (args.plain_texts or []) + (args.texts or [])
    corpus = TextCorpus(plain_paths, keep_marks=keep_marks, letters_only=letters_only) if plain_paths else None

    # Templates
    templates = build_templates()
    tpl_groups = build_templates_grouped()
    tpl_group_keys = ["name", "title", "royal", "blessing", "plain"]
    tpl_group_weights = _parse_kv_weights(args.template_group_weights, tpl_group_keys)

    # Decide doc-probs
    doc_probs = _parse_kv_weights(args.doc_probs, ["seal", "plain"]) if args.doc_probs else None
    plain_prob = float(args.plain_prob) if args.plain_prob is not None else float(args.text_prob)

    # If user didn't set doc_probs explicitly, use plain_prob only when corpus exists
    if doc_probs is None:
        if corpus and plain_prob > 0:
            doc_probs = [max(0.0, 1.0 - plain_prob), max(0.0, plain_prob)]  # seal, plain
        else:
            doc_probs = [1.0, 0.0]

    uid_prefix = args.uid_prefix
    written = 0
    max_tries = 20
    progress_bar = tqdm(total=args.n)

    # Helper: sample fillers for seal templates
    def sample_fillers() -> Dict[str, str]:
        sep = random.choice([" ", ""])
        titles = ["עבד", "שר", "כהן", "סופר", "נצב"]
        return {
            "X": name_s.sample(3, 7),
            "Y": name_s.sample(3, 7),
            "Z": name_s.sample(3, 7),
            "T": random.choice(titles),
            "P": place_s.sample(2, 6),
            "PP": place_phrase_s.sample(),
            "G": god_s.sample(2, 6),
            "SEP": sep,
            "N": str(random.randint(1, 9)),
        }

    def instantiate_template(tmpl: str, fillers: Dict[str, str]) -> str:
        text = tmpl
        for k, v in fillers.items():
            text = text.replace("{" + k + "}", v)
        return text

    def make_text_variants(text_norm: str) -> Tuple[str, str]:
        """
        Returns (text_gt, text_render) based on --final-forms.
        We finalize word-ends on the 22-letter norm form to produce canonical finals.
        """
        if args.final_forms == "none":
            return text_norm, text_norm

        finalized = apply_final_forms_end(text_norm, min_len=int(args.final_forms_min_len))
        text_gt = finalized if args.final_forms in ("gt", "both") else text_norm
        text_render = finalized if args.final_forms in ("render", "both") else text_norm
        return text_gt, text_render

    while written < args.n:
        uid = f"{uid_prefix}_{written:06d}"
        glyph_scale_mean = random.uniform(args.glyph_scale_min, args.glyph_scale_max) if args.glyph_scale_std > 0 else None

        ok = False
        for _ in range(max_tries):
            cfg, sampled_params = sample_cfg_for_example(base_cfg, args)

            # pick doc kind
            doc_kind = random.choices(["seal", "plain"], weights=doc_probs, k=1)[0]

            source = "template"
            text_raw: Optional[str] = None

            if doc_kind == "plain":
                # corpus snippet only
                if corpus:
                    text_raw = corpus.sample(args.text_min_chars, args.text_max_chars)
                if text_raw:
                    source = "plain:corpus"
                    # optional plain augmentation (no formulas)
                    if args.plain_augment:
                        text_raw = plain_text_augment(
                            text_raw,
                            letters_only=letters_only,
                            keep_marks=keep_marks,
                            join_all_prob=float(args.plain_join_all_prob),
                            drop_some_spaces_prob=float(args.plain_drop_spaces_prob),
                            sep_maqaf_prob=float(args.plain_sep_maqaf_prob),
                            sep_dot_prob=float(args.plain_sep_dot_prob),
                        )
                    text_raw = inject_inter_word_newline(text_raw, float(args.plain_inter_word_newline_prob))
                else:
                    # fallback: plain template group (still "seal-ish", but not corpus)
                    doc_kind = "seal"
                    text_raw = None

            if doc_kind == "seal":
                if tpl_group_weights is not None:
                    g = random.choices(tpl_group_keys, weights=tpl_group_weights, k=1)[0]
                    tmpl = random.choice(tpl_groups[g])
                    source = f"seal:template:{g}"
                else:
                    tmpl = random.choice(templates)
                    source = "seal:template"
                fillers = sample_fillers()
                text_raw = instantiate_template(tmpl, fillers)

            assert text_raw is not None

            # normalize (canonical 22-letter), then breaks
            text_norm = normalize_hebrew_for_seals(
                text_raw,
                keep_marks=keep_marks,
                allow_newlines=True,
                letters_only=letters_only,
                map_finals_to_normal=True,  # keep stable base; finals re-added later if requested
            )
            text_norm = inject_intra_word_breaks(text_norm, float(args.intra_word_break_prob), break_char=break_char)

            # derive GT/render variants
            text_gt, text_render = make_text_variants(text_norm)

            # Pick font
            fp, font = random.choices(fonts, weights=weights, k=1)[0]

            # Render: use text_render (may be with or without finals depending on mode)
            res = render_text_glyph_by_glyph(text_render, font, cfg, rtl=args.rtl, glyph_scale_mean=glyph_scale_mean)

            if res[0] is None:
                # rescue: add newline at space if possible (operate on text_norm, then recompute variants)
                if " " in text_norm and "\n" not in text_norm:
                    parts = text_norm.split()
                    if len(parts) >= 2:
                        cut = random.randint(1, len(parts) - 1)
                        text_norm2 = " ".join(parts[:cut]) + "\n" + " ".join(parts[cut:])
                        text_gt2, text_render2 = make_text_variants(text_norm2)
                        res = render_text_glyph_by_glyph(text_render2, font, cfg, rtl=args.rtl, glyph_scale_mean=glyph_scale_mean)
                        if res[0] is None:
                            continue
                        text_norm, text_gt, text_render = text_norm2, text_gt2, text_render2
                    else:
                        continue
                else:
                    continue

            img, chars, bboxes, cf = res

            meta = {
                "text_raw": text_raw,
                "text_norm": text_norm,        # 22-letter canonical
                "text_gt": text_gt,            # GT (optionally with finals)
                "text_render": text_render,    # rendered text (optionally with finals)
                "final_forms": str(args.final_forms),
                "final_forms_min_len": int(args.final_forms_min_len),
                "font": fp,
                "contrast": cf,
                "source": source,
                "doc_kind": "plain" if source.startswith("plain:") else "seal",
                "sampled": sampled_params,
            }

            write_example(
                out_root,
                uid,
                img,
                text_render=text_render,
                text_gt=text_gt,
                text_norm=text_norm,
                chars=chars,
                bboxes=bboxes,
                meta=meta,
            )

            rec = {
                "uid": uid,
                "image": {"rel_path": f"images/{uid}.png", "width": img.width, "height": img.height},
                "gt": {
                    "hebrew": text_gt,           # recommended for VLM target text
                    "hebrew_norm": text_norm,    # 22-letter stable form
                    "hebrew_render": text_render,
                    "bboxes": bboxes,
                    "chars": chars,
                },
                "split": "train",
                "meta": meta,
            }
            append_manifest(manifest_path, rec)
            ok = True
            break

        if ok:
            written += 1
            progress_bar.update(1)

    progress_bar.close()
    print(f"Done. Wrote {written} images to {out_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()