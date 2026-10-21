"""Two-column reading-order recovery on LogicalBlocks (per page). Heuristic and conservative:
a page is treated as two-column only if a clear vertical gutter exists with enough blocks on both sides."""
from __future__ import annotations

from .schemas import LogicalBlock, PageInfo, PageRoute

FULL_WIDTH_FRAC = 0.6
MIN_BLOCKS_PER_COL = 3


def _find_gutter(narrow: list[LogicalBlock], page_w: float) -> float | None:
    if len(narrow) < 2 * MIN_BLOCKS_PER_COL:
        return None
    best = None
    for x in range(int(page_w * 0.35), int(page_w * 0.65)):
        left = [b for b in narrow if b.bbox[2] <= x]
        right = [b for b in narrow if b.bbox[0] >= x]
        crossing = len(narrow) - len(left) - len(right)
        if crossing == 0 and len(left) >= MIN_BLOCKS_PER_COL and len(right) >= MIN_BLOCKS_PER_COL:
            gap = min(b.bbox[0] for b in right) - max(b.bbox[2] for b in left)
            if best is None or gap > best[1]:
                best = (float(x), gap)
    return best[0] if best and best[1] >= 8 else None


def reorder_columns(blocks: list[LogicalBlock], pages: list[PageInfo]) -> tuple[list[LogicalBlock], list[int]]:
    """Returns (blocks in recovered order with reading_order renumbered, pages detected as two-column)."""
    width = {p.page_index: p.width for p in pages}
    trusted = {p.page_index for p in pages if p.route is PageRoute.BORN_DIGITAL}
    by_page: dict[int, list[LogicalBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page_index, []).append(b)
    out: list[LogicalBlock] = []
    twocol: list[int] = []
    for pg in sorted(by_page):
        bl = by_page[pg]
        w = width[pg]
        narrow = [b for b in bl if (b.bbox[2] - b.bbox[0]) < FULL_WIDTH_FRAC * w]
        gx = _find_gutter(narrow, w) if pg in trusted else None   # never reorder OCR-layer pages
        if gx is None:
            out.extend(bl)
            continue
        twocol.append(pg)
        band_l: list[LogicalBlock] = []; band_r: list[LogicalBlock] = []

        def flush():
            out.extend(sorted(band_l, key=lambda b: b.bbox[1])); out.extend(sorted(band_r, key=lambda b: b.bbox[1]))
            band_l.clear(); band_r.clear()
        for b in sorted(bl, key=lambda b: (b.bbox[1], b.bbox[0])):
            if (b.bbox[2] - b.bbox[0]) >= FULL_WIDTH_FRAC * w:
                flush(); out.append(b)
            elif b.bbox[2] <= gx:
                band_l.append(b)
            else:
                band_r.append(b)
        flush()
    for i, b in enumerate(out):
        b.reading_order = i
    return out, twocol
