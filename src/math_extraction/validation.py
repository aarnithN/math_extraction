"""Completeness and consistency checks over Phase 1 output."""
from __future__ import annotations

from pathlib import Path

import pymupdf

from .schemas import PageInfo, PageRoute, SourceBlock

BBOX_TOL = 2.0


def validate(pdf_path: Path, root: Path, pages: list[PageInfo], blocks: list[SourceBlock]) -> list[dict]:
    issues: list[dict] = []
    add = lambda sev, code, msg, **kw: issues.append(dict(severity=sev, code=code, message=msg, **kw))
    doc = pymupdf.open(pdf_path)

    if len(pages) != len(doc):
        add("error", "page_count", f"{len(pages)} PageInfo vs {len(doc)} pages")
    ids = [b.block_id for b in blocks]
    if len(ids) != len(set(ids)):
        add("error", "duplicate_block_id", "block ids are not unique")
    ro = [b.reading_order for b in blocks]
    if ro != list(range(len(ro))):
        add("error", "reading_order", "reading_order is not a gapless 0..n-1 sequence")

    by_page: dict[int, list[SourceBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page_index, []).append(b)
        page = doc[b.page_index]
        x0, y0, x1, y1 = b.bbox
        r = page.rect
        if x1 <= x0 or y1 <= y0:
            add("warning", "degenerate_bbox", "zero/negative-size bbox", block_id=b.block_id)
        if x0 < r.x0 - BBOX_TOL or y0 < r.y0 - BBOX_TOL or x1 > r.x1 + BBOX_TOL or y1 > r.y1 + BBOX_TOL:
            add("error", "bbox_outside_page", f"{b.bbox} outside {tuple(r)}", block_id=b.block_id)
        if b.source_image_ref and not (root / b.source_image_ref).exists():
            add("error", "missing_crop", b.source_image_ref, block_id=b.block_id)

    for p in pages:
        page = doc[p.page_index]
        ref = sum(len(c) for c in page.get_text("text").split())
        got = sum(len(c) for b in by_page.get(p.page_index, []) if b.raw_text for c in b.raw_text.split())
        if ref != got:
            add("error", "text_coverage", f"page {p.page_index}: {got} non-space chars in blocks vs {ref} in page text",
                page_index=p.page_index)
        if p.route is PageRoute.EMBEDDED_OCR_LAYER:
            add("warning", "embedded_ocr_page", "text layer is OS OCR on ink; needs vision/OCR recovery (Phase 6)",
                page_index=p.page_index)
        if p.route in (PageRoute.SCANNED_NO_TEXT, PageRoute.EMPTY):
            add("warning", "no_text_layer", p.route.value, page_index=p.page_index)
    return issues
