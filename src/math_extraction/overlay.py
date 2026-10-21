"""Draw block boxes on page renders for visual alignment checks."""
from __future__ import annotations

from pathlib import Path

import pymupdf

from .schemas import PageInfo, SourceBlock

COLORS = {"born_digital": (0, 0.4, 0.9), "embedded_ocr_layer": (0.9, 0.3, 0), "other": (0.5, 0.5, 0.5)}


def write_overlays(pdf_path: Path, root: Path, pages: list[PageInfo], blocks: list[SourceBlock],
                   page_indices: list[int] | None = None, dpi: int = 100) -> list[Path]:
    doc = pymupdf.open(pdf_path)
    out = []
    (root / "overlays").mkdir(exist_ok=True)
    for p in pages:
        if page_indices is not None and p.page_index not in page_indices:
            continue
        page = doc[p.page_index]
        col = COLORS.get(p.route.value, COLORS["other"])
        for b in blocks:
            if b.page_index == p.page_index:
                page.draw_rect(pymupdf.Rect(b.bbox), color=col, width=0.8)
                page.insert_text((b.bbox[0], max(b.bbox[1] - 1, 6)), str(b.reading_order), fontsize=6, color=(1, 0, 0))
        path = root / "overlays" / f"page_{p.page_index:04d}.png"
        page.get_pixmap(dpi=dpi).save(path)
        out.append(path)
    return out
