"""Read user annotations (highlights, comments, free text, ink) as separate objects."""
from __future__ import annotations

import pymupdf

from .schemas import PdfAnnotation, SourceBlock

MARKUP = {"Highlight", "Underline", "StrikeOut", "Squiggly"}


def _overlap(a, b) -> bool:
    r = pymupdf.Rect(a) & pymupdf.Rect(b)
    return not r.is_empty and r.get_area() > 0


def read_annotations(doc: pymupdf.Document, document_id: str, blocks: list[SourceBlock]) -> list[PdfAnnotation]:
    out: list[PdfAnnotation] = []
    by_page: dict[int, list[SourceBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page_index, []).append(b)
    for page in doc:
        for n, a in enumerate(page.annots() or []):
            subtype = a.type[1]
            info = a.info or {}
            r = a.rect
            warnings: list[str] = []
            hl = None
            if subtype in MARKUP:
                try:
                    quads = a.vertices or []
                    rects = [pymupdf.Quad(quads[i:i + 4]).rect for i in range(0, len(quads), 4)]
                    hl = " ".join(page.get_textbox(q).strip() for q in rects).strip() or None
                except Exception as e:  # malformed quads happen in real files
                    warnings.append(f"could not read highlighted text: {e}")
            ink_pts = None
            if subtype == "Ink":
                ink_pts = sum(len(s) for s in (a.vertices or []) if isinstance(s, list)) or None
            bbox = tuple(round(v, 3) for v in (r.x0, r.y0, r.x1, r.y1))
            out.append(PdfAnnotation(
                document_id=document_id, annotation_id=f"{document_id}-p{page.number:04d}-a{n:03d}",
                page_index=page.number, subtype=subtype, bbox=bbox, content=info.get("content", "") or "",
                author=info.get("title") or None, highlighted_text=hl,
                overlapping_block_ids=[b.block_id for b in by_page.get(page.number, []) if _overlap(b.bbox, bbox)],
                n_ink_points=ink_pts, warnings=warnings))
    return out
