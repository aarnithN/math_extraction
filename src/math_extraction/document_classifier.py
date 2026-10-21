"""Per-page routing. Heuristic; every decision records its reasons.

The routes are hypotheses used to decide how much to trust the text layer,
not ground truth. Thresholds are untuned defaults.
"""
from __future__ import annotations

import re

import pymupdf

from .schemas import PageRoute

MATH_FONT_RE = re.compile(
    r"(CMMI|CMSY|CMEX|CMMIB|LMMath|MathItalic|MathSymbols|MathExtension|MSBM|MSAM|"
    r"Symbol|Cambria\s*Math|STIX|Euler|MnSymbol)", re.I)
# OS-generated OCR text layers (e.g. Apple Live Text) use system UI fonts.
SYSTEM_FONT_RE = re.compile(r"(^\.|SFUI|SFNS|AppleSDGothic|PingFang|HiraKaku|AppleColor)", re.I)

INK_DRAWINGS_THRESHOLD = 50


def is_math_font(name: str) -> bool:
    return bool(MATH_FONT_RE.search(name))


def is_system_font(name: str) -> bool:
    return bool(SYSTEM_FONT_RE.search(name))


def classify_page(page: pymupdf.Page, text_dict: dict) -> tuple[PageRoute, list[str], dict]:
    chars = sys_chars = math_chars = n_img = n_txt = 0
    for b in text_dict["blocks"]:
        if b["type"] == 1:
            n_img += 1
            continue
        n_txt += 1
        for l in b["lines"]:
            for s in l["spans"]:
                n = len(s["text"].strip())
                chars += n
                if is_system_font(s["font"]):
                    sys_chars += n
                if is_math_font(s["font"]):
                    math_chars += n
    n_draw = len(page.get_drawings())
    stats = dict(n_chars=chars, n_math_font_chars=math_chars, n_text_blocks=n_txt,
                 n_image_blocks=n_img, n_vector_drawings=n_draw)
    reasons: list[str] = []
    if chars == 0:
        if n_img or n_draw >= INK_DRAWINGS_THRESHOLD:
            reasons.append("no text layer but page has images/vector ink")
            return PageRoute.SCANNED_NO_TEXT, reasons, stats
        reasons.append("no text, no images, little vector content")
        return PageRoute.EMPTY, reasons, stats
    if sys_chars / chars > 0.5:
        reasons.append(f"{sys_chars}/{chars} chars use OS system fonts (typical of OS-generated OCR layers)")
        if n_draw >= INK_DRAWINGS_THRESHOLD:
            reasons.append(f"{n_draw} vector drawings (likely handwriting strokes or ruled background)")
        return PageRoute.EMBEDDED_OCR_LAYER, reasons, stats
    reasons.append("text layer uses embedded document fonts")
    if math_chars:
        reasons.append(f"{math_chars} chars in math fonts")
    return PageRoute.BORN_DIGITAL, reasons, stats
