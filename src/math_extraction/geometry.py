"""Coordinate helpers. See schemas.py for the coordinate-system definition."""
from __future__ import annotations

import pymupdf

from .schemas import BBox


def render_scale(dpi: int) -> float:
    return dpi / 72.0


def page_to_render_rect(bbox: BBox, page: pymupdf.Page, dpi: int) -> tuple[float, float, float, float]:
    """Map an unrotated-page-space bbox to pixel coords in `page.get_pixmap(dpi=dpi)`."""
    m = page.rotation_matrix * pymupdf.Matrix(render_scale(dpi), render_scale(dpi))
    r = pymupdf.Rect(bbox).transform(m)
    return (r.x0, r.y0, r.x1, r.y1)


def render_to_page_rect(px: tuple[float, float, float, float], page: pymupdf.Page, dpi: int) -> BBox:
    """Inverse of page_to_render_rect (used by tests and by later OCR alignment)."""
    s = render_scale(dpi)
    m = (page.rotation_matrix * pymupdf.Matrix(s, s))
    r = pymupdf.Rect(px).transform(~m)
    return (r.x0, r.y0, r.x1, r.y1)
