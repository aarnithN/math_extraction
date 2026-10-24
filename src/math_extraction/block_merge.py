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
MIN_OVERLAP_GAP = -150.0


def _mergeable(a: SourceBlock, b: SourceBlock) -> bool:
    """Same-line fragments of one expression (symmetric: `a` is whichever starts further left)."""
    if a.page_index != b.page_index or a.block_type != "text" or b.block_type != "text":
        return False
    if a.bbox[0] > b.bbox[0]:
        a, b = b, a
    gap = b.bbox[0] - a.bbox[2]
    ov = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
    h = min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1])
    if h <= 0:
        return False
    if a.baseline is not None and b.baseline is not None:
        size = max(a.max_font_size or 10, b.max_font_size or 10)
        if abs(a.baseline - b.baseline) > 0.75 * size:
            return False                      # different text lines that merely overlap vertically (tall glyphs)
    r = ov / h
    if MIN_HGAP <= gap <= MAX_HGAP and r >= MIN_VOVERLAP:
        return True
    # Sub/superscript, brace and big-operator fragments overlap the previous box horizontally.
    return MIN_OVERLAP_GAP <= gap < MIN_HGAP and r >= 0.55 and b.bbox[2] > a.bbox[2]


def build_logical_blocks(document_id: str, blocks: list[SourceBlock], pages: list[PageInfo]) -> list[LogicalBlock]:
    route = {p.page_index: p.route for p in pages}
    parent = list(range(len(blocks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    by_page: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        if b.block_type == "text" and route.get(b.page_index) is PageRoute.BORN_DIGITAL:
            by_page.setdefault(b.page_index, []).append(i)
    for idxs in by_page.values():
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                if _mergeable(blocks[idxs[x]], blocks[idxs[y]]):
                    parent[find(idxs[y])] = find(idxs[x])
    groups_d: dict[int, list[SourceBlock]] = {}
    for i, b in enumerate(blocks):
        groups_d.setdefault(find(i), []).append(b)
    groups = sorted(groups_d.values(), key=lambda g: g[0].reading_order)
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