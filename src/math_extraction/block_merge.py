"""Merge parser-split fragments (typically an equation broken into pieces) into LogicalBlocks.

Source blocks are untouched; this builds a second layer. Rule: consecutive text blocks on the same page
whose boxes sit side by side on the same line (small horizontal gap, large vertical overlap).
Pages whose text is an embedded OCR layer are never merged (their boxes are not trustworthy).
"""
from __future__ import annotations

from .schemas import LogicalBlock, PageInfo, PageRoute, Provenance, SourceBlock

MAX_HGAP = 8.0
MIN_HGAP = -3.0
MIN_VOVERLAP = 0.35
MIN_OVERLAP_GAP = -15.0


def _mergeable(a: SourceBlock, b: SourceBlock) -> bool:
    if a.page_index != b.page_index or a.block_type != "text" or b.block_type != "text":
        return False
    gap = b.bbox[0] - a.bbox[2]
    ov = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
    h = min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1])
    if h <= 0:
        return False
    r = ov / h
    if MIN_HGAP <= gap <= MAX_HGAP and r >= MIN_VOVERLAP:
        return True
    # Sub/superscript boxes can overlap the previous box horizontally (e.g. h_1^{-1} then h'):
    # accept a deeper overlap only if the boxes are almost exactly on one line and B extends right of A.
    return MIN_OVERLAP_GAP <= gap < MIN_HGAP and r >= 0.8 and b.bbox[0] > a.bbox[0] and b.bbox[2] > a.bbox[2]


def build_logical_blocks(document_id: str, blocks: list[SourceBlock], pages: list[PageInfo]) -> list[LogicalBlock]:
    route = {p.page_index: p.route for p in pages}
    groups: list[list[SourceBlock]] = []
    for b in blocks:
        can = (groups and route.get(b.page_index) is PageRoute.BORN_DIGITAL and _mergeable(groups[-1][-1], b))
        if can:
            groups[-1].append(b)
        else:
            groups.append([b])
    out: list[LogicalBlock] = []
    counters: dict[int, int] = {}
    for g in groups:
        pg = g[0].page_index
        n = counters.get(pg, 0)
        counters[pg] = n + 1
        x0 = min(b.bbox[0] for b in g); y0 = min(b.bbox[1] for b in g)
        x1 = max(b.bbox[2] for b in g); y1 = max(b.bbox[3] for b in g)
        raw = "\n".join(b.raw_text or "" for b in g)
        warns = [w for b in g for w in b.warnings]
        if len(g) > 1:
            warns.append(f"merged {len(g)} parser blocks that sit on one line (likely a split equation); "
                         "text is concatenated in parser order and may still be lossy")
        out.append(LogicalBlock(
            document_id=document_id, logical_id=f"{document_id}-L{pg:04d}-{n:04d}", page_index=pg,
            member_block_ids=[b.block_id for b in g], reading_order=len(out), bbox=(x0, y0, x1, y1),
            raw_text=raw, normalized_text=" ".join(raw.split()),
            has_math_font=any(b.has_math_font for b in g), has_bold=any(b.has_bold for b in g),
            max_font_size=max((b.max_font_size or 0) for b in g) or None,
            provenance=g[0].provenance, merge_reason="same_line_fragments" if len(g) > 1 else None,
            needs_review=any(b.needs_review for b in g) or len(g) > 1, warnings=warns))
    return out
