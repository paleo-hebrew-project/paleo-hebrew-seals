"""Shared Hebrew 22-letter constants matching notebooks/Paleo_OCR.ipynb conventions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# 22 letters (canonical forms; finals mapped elsewhere)
HEB22_LETTERS: List[str] = [
    "א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י",
    "כ", "ל", "מ", "נ", "ס", "ע", "פ", "צ", "ק", "ר", "ש", "ת",
]

# Latin names for folder prefixes (same order as cls_22_from_manifests_v1/classes.txt)
HEB22_ENGLISH_NAMES: List[str] = [
    "aleph", "bet", "gimel", "dalet", "he", "vav", "zayin", "het", "tet", "yod",
    "kaf", "lamed", "mem", "nun", "samekh", "ayin", "pe", "tsadi", "qof", "resh", "shin", "tav",
]

FINAL_TO_BASE: Dict[str, str] = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}

HEB_SET = frozenset(HEB22_LETTERS)


def class_names_heb22() -> List[str]:
    """Folder names like 01_aleph_א — matches notebook CLASS_NAMES."""
    return [
        f"{i:02d}_{name}_{letter}"
        for i, (name, letter) in enumerate(zip(HEB22_ENGLISH_NAMES, HEB22_LETTERS), start=1)
    ]


def letter_to_class_map() -> Dict[str, str]:
    names = class_names_heb22()
    return {letter: cls_name for cls_name, letter in zip(names, HEB22_LETTERS)}


LETTER_TO_CLASS: Dict[str, str] = letter_to_class_map()
