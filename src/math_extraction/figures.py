"""Deterministic detection of diagrams, plots and graphs in a PDF's vector layer.

Pipeline per page (no OCR, no model):
  1. Read vector paths (PyMuPDF get_drawings), de-duplicate, drop page rules/grids and text decorations
     (fraction bars, radical tops, underbraces, QED boxes, underlines).
  2. Recognise arrows: a shaft path with a small arrowhead path at one end (+ dash pattern, hook tails).
  3. Cluster the remaining paths into figures and CLAIM the label text that sits inside them, so diagram
     labels no longer pollute the prose segments.
  4. Classify (arrow_diagram, graph_network, function_plot, region_diagram, line_diagram, raster_image,
     ink_drawing) and recover structure: nodes (text tokens, circles, closed regions), edges (direction,
     style, labels), plot axes/ticks/curves with data coordinates when tick labels are numeric.

Everything is geometric. Output is marked structure_status 'recovered' only when the structure was fully
resolved, otherwise 'partial' and needs_review, so a later VLM/OCR pass can double-check.

Limits: curved/multi-segment arrows, double (=>) arrows, dense graphs, 3D plots, pgfplots markers and
figures made of rasters or text-only art are only partially handled; handwritten drawings are located but not
interpreted.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf

from .math_symbols import Ch, latex_for_chars, page_bars, page_chars
from .schemas import (FigureBlock, FigureEdge, FigureNode, LogicalBlock, PageInfo, PageRoute, Provenance, SourceBlock)

HEAD_MAX = 10.0            # arrowheads are smaller than this (pt)
SHAFT_MIN = 6.0
CLUSTER_GAP = 7.0
LABEL_GAP = 16.0
NODE_REACH = 24.0
INK_MIN_AREA = 3200.0


# ----------------------------------------------------------------------------- drawing model
@dataclass
class Dr:
    rect: pymupdf.Rect
    items: list
    typ: str
    color: Optional[tuple]
    fill: Optional[tuple]
    dashed: bool
    width: float
    closed: bool
    idx: int
    pts: list = field(default_factory=list)       # flattened polyline(s)
    role: str = ""                                 # shaft | head | decor | circle | closed | line | other

    @property
    def w(self): return self.rect.width
    @property
    def h(self): return self.rect.height
    @property
    def start(self): return self.pts[0] if self.pts else None
    @property
    def end(self): return self.pts[-1] if self.pts else None


def _bez(p0, p1, p2, p3, n=10):
    out = []
    for i in range(n + 1):
        t = i / n; u = 1 - t
        out.append((u**3 * p0.x + 3 * u**2 * t * p1.x + 3 * u * t**2 * p2.x + t**3 * p3.x,
                    u**3 * p0.y + 3 * u**2 * t * p1.y + 3 * u * t**2 * p2.y + t**3 * p3.y))
    return out


def _flatten(items) -> tuple[list, bool]:
    pts: list = []
    for it in items:
        k = it[0]
        if k == "l":
            seg = [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        elif k == "c":
            seg = _bez(it[1], it[2], it[3], it[4])
        elif k == "re":
            r = it[1]; seg = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
        elif k == "qu":
            q = it[1]; seg = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)]
        else:
            continue
        if pts and math.dist(pts[-1], seg[0]) < 0.6:
            pts.extend(seg[1:])
        else:
            pts.extend(seg)
    closed = len(pts) > 3 and math.dist(pts[0], pts[-1]) < 1.2
    return pts, closed


def read_drawings(page: pymupdf.Page) -> list[Dr]:
    seen, out = set(), []
    for i, d in enumerate(page.get_drawings()):
        r = d["rect"]
        sig = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1), tuple(it[0] for it in d["items"]))
        if sig in seen:
            continue                                     # tikz sometimes strokes a path twice
        seen.add(sig)
        dash = d.get("dashes") or ""
        dashed = bool(dash) and dash.strip("[] 0") != "" and "[] 0" not in dash
        pts, closed = _flatten(d["items"])
        out.append(Dr(r, d["items"], d.get("type", ""), tuple(d["color"]) if d.get("color") else None,
                      tuple(d["fill"]) if d.get("fill") else None, dashed, d.get("width") or 0.0,
                      bool(d.get("closePath")) or closed, i, pts))
    return out


def _plen(pts) -> float:
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def _is_grid_or_frame(d: Dr, page: pymupdf.Rect) -> bool:
    if d.w >= 0.6 * page.width or d.h >= 0.6 * page.height:
        return True
    if d.fill and all(c > 0.7 for c in d.fill) and (d.w < 1.6 or d.h < 1.6):
        return True
    if d.color and all(c > 0.8 for c in d.color) and d.fill is None:
        return True
    return False


# ----------------------------------------------------------------------------- arrows
def _classify(drs: list[Dr], chars: list[Ch], page: pymupdf.Rect) -> None:
    for d in drs:
        d.role = "other"
        if _is_grid_or_frame(d, page):
            d.role = "grid"
    live = [d for d in drs if d.role != "grid"]
    heads = [d for d in live if max(d.w, d.h) <= HEAD_MAX and d.items and d.items[0][0] in "cl" and _plen(d.pts) < 30]
    shafts = [d for d in live if d not in heads and all(it[0] in "lc" for it in d.items) and _plen(d.pts) >= SHAFT_MIN and not d.closed]
    for d in shafts:
        d.role = "shaft"; d.head_at = None; d.tail = None; d.heads = []
    used = set()
    for sh in shafts:
        for end_name, p in (("end", sh.end), ("start", sh.start)):
            best = None
            for h in heads:
                if id(h) in used:
                    continue
                c = h.rect.tl + (h.rect.br - h.rect.tl) * 0.5
                dist = math.dist((c.x, c.y), p)
                if dist <= 5.5 and (best is None or dist < best[0]):
                    best = (dist, h)
            if best:
                used.add(id(best[1])); sh.heads.append((end_name, best[1]))
    for sh in shafts:
        if len(sh.heads) == 2:
            (e0, h0), (e1, h1) = sh.heads
            sig = lambda h: (len(h.items), round(h.w), round(h.h))
            if sig(h0) != sig(h1):                         # unlike shapes: the odd one on the start side is a hook tail
                tail = h0 if e0 == "start" else h1
                if tail.items and len(tail.items) < max(len(h0.items), len(h1.items)) or sig(h0)[1:] != sig(h1)[1:]:
                    sh.tail = "hook"; tail.role = "head"
                    sh.heads = [(e, h) for e, h in sh.heads if e == "end"] or sh.heads
        kinds = {e for e, _ in sh.heads}
        sh.head_at = "both" if len(kinds) == 2 else (next(iter(kinds)) if kinds else None)
        for _, h in sh.heads:
            h.role = "head"
    # unmatched small drawings are decorations/dots, not arrowheads
    for h in heads:
        if h.role != "head":
            h.role = "small"
    # shafts without a head: plain lines. Thin horizontal lines hugging text are math decorations.
    for sh in shafts:
        if sh.head_at is None and abs(sh.start[1] - sh.end[1]) < 0.5 and _plen(sh.pts) < 0.35 * page.width:
            y = sh.start[1]; x0, x1 = sorted((sh.start[0], sh.end[0]))
            if any(x0 - 1 <= c.cx <= x1 + 1 and (abs(c.y1 - y) < 3.5 or abs(c.y0 - y) < 3.5) for c in chars):
                sh.role = "decor"
    for d in live:
        if d.fill and all(c < 0.12 for c in d.fill) and len(d.items) == 1 and d.items[0][0] == "re" and d.w >= 12 and d.h >= 4:
            d.role = "redaction"
    for d in live:
        if d.role in ("other", "small") and d.rect.width <= 14 and d.rect.height <= 14 and abs(d.w - d.h) < 2.5 and d.closed:
            d.role = "decor"                              # QED square and similar glyph-like boxes
        elif d.role == "other" and d.rect.height <= 1.8 and d.rect.width < 0.35 * page.width:
            d.role = "decor"
    for d in live:
        if d.role in ("other", "small"):
            if _is_circle(d):
                d.role = "circle"
            elif d.closed:
                d.role = "closed"
            elif d.role == "other":
                d.role = "line" if d.items and all(it[0] == "l" for it in d.items) else "other"


def _is_circle(d: Dr) -> bool:
    if len(d.items) < 4 or not d.closed:
        return False
    if sum(it[0] == "c" for it in d.items) < 4:
        return False
    return 0.93 <= d.w / max(d.h, 1e-6) <= 1.07 and d.w <= 60


# ----------------------------------------------------------------------------- clustering
def _looks_like_table(cl: list[Dr]) -> bool:
    """Only axis-aligned thin rules (no heads, curves, circles): a table or ruled box, not a diagram."""
    rules = [d for d in cl if d.role in ("other", "line", "shaft", "small")]
    if len(rules) < 2 or any(d.role in ("circle", "closed") for d in cl):
        return False
    if any(getattr(d, "head_at", None) for d in cl):
        return False
    axis = lambda d: min(d.w, d.h) <= 1.6
    if not all(axis(d) for d in rules):
        return False
    horiz = [d for d in rules if d.w > d.h]; vert = [d for d in rules if d.h >= d.w]
    return (len(rules) >= 3 and horiz and vert) or (not horiz or not vert)   # grid, or only parallel rules


def _inflate(r: pymupdf.Rect, g: float) -> pymupdf.Rect:
    return pymupdf.Rect(r.x0 - g, r.y0 - g, r.x1 + g, r.y1 + g)


def _cluster(elems: list[Dr]) -> list[list[Dr]]:
    parent = list(range(len(elems)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            if _inflate(elems[i].rect, CLUSTER_GAP).intersects(elems[j].rect):
                parent[find(j)] = find(i)
    groups: dict[int, list[Dr]] = {}
    for i, e in enumerate(elems):
        groups.setdefault(find(i), []).append(e)
    return list(groups.values())


# ----------------------------------------------------------------------------- text tokens
def _tokens(chars: list[Ch]) -> list[list[Ch]]:
    """Group glyphs into label tokens (a node name, an arrow label) by spatial adjacency."""
    n = len(chars)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            a, b = chars[i], chars[j]
            gx = max(b.x0 - a.x1, a.x0 - b.x1)
            gy = max(b.y0 - a.y1, a.y0 - b.y1)
            s = max(a.size, b.size)
            if gx <= 0.45 * s and gy <= 0.05 * s:
                parent[find(j)] = find(i)
    groups: dict[int, list[Ch]] = {}
    for i, c in enumerate(chars):
        groups.setdefault(find(i), []).append(c)
    return [sorted(g, key=lambda c: c.ox) for g in groups.values()]


def _strip_dollars(s: str) -> str:
    s = s.strip()
    return s[1:-1] if s.startswith("$") and s.endswith("$") and s.count("$") == 2 else s


def _bbox(chs) -> tuple:
    return (min(c.x0 for c in chs), min(c.y0 for c in chs), max(c.x1 for c in chs), max(c.y1 for c in chs))


def _dist_pt_rect(p, b) -> float:
    dx = max(b[0] - p[0], 0, p[0] - b[2]); dy = max(b[1] - p[1], 0, p[1] - b[3])
    return math.hypot(dx, dy)


def _inside(p, poly) -> bool:
    x, y = p; c = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            c = not c
    return c


# ----------------------------------------------------------------------------- structure builders
def _build_arrow_edges(fig_dr: list[Dr], tokens, node_tok) -> list[dict]:
    edges = []
    for d in fig_dr:
        if d.role == "shaft":
            a, b = d.start, d.end
            at = getattr(d, "head_at", None)
            if at == "start":
                a, b = b, a
            directed = at in ("start", "end")
            tail = getattr(d, "tail", None)
            edges.append(dict(kind="arrow" if directed else "line", a=a, b=b, directed=directed and at != "both",
                              style=("dashed" if d.dashed else ("curved" if any(it[0] == "c" for it in d.items) else "solid")),
                              pts=d.pts, rect=d.rect, both=at == "both", tail=tail, dr=d))
        elif d.role == "line":
            edges.append(dict(kind="line", a=d.start, b=d.end, directed=False, style="dashed" if d.dashed else "solid",
                              pts=d.pts, rect=d.rect, both=False, tail=None, dr=d))
    return edges


def _nearest_token(p, tokens, reach=NODE_REACH, exclude=()):
    best = None
    for i, t in enumerate(tokens):
        if i in exclude:
            continue
        dd = _dist_pt_rect(p, t["bbox"])
        if dd <= reach and (best is None or dd < best[0]):
            best = (dd, i)
    return best[1] if best else None


def _axes(lines: list[Dr], tokens) -> Optional[dict]:
    hz = [d for d in lines if d.start and abs(d.start[1] - d.end[1]) < 0.8 and abs(d.start[0] - d.end[0]) >= 35]
    vt = [d for d in lines if d.start and abs(d.start[0] - d.end[0]) < 0.8 and abs(d.start[1] - d.end[1]) >= 35]
    for h in hz:
        for v in vt:
            hx0, hx1 = sorted((h.start[0], h.end[0])); vy0, vy1 = sorted((v.start[1], v.end[1]))
            vx, hy = v.start[0], h.start[1]
            if hx0 - 8 <= vx <= hx1 + 8 and vy0 - 8 <= hy <= vy1 + 8:
                return dict(x_axis=dict(y=hy, x0=hx0, x1=hx1, arrow=getattr(h, "head_at", None)),
                            y_axis=dict(x=vx, y0=vy0, y1=vy1, arrow=getattr(v, "head_at", None)),
                            origin=(vx, hy), dr=(h, v))
    return None


def _calibrate(axes: dict, tokens, all_dr: list[Dr]) -> dict:
    """Affine page->data mapping from numeric tick labels (needs >=2 per axis)."""
    out = {}
    ox, oy = axes["origin"]
    nums = []
    for t in tokens:
        s = re.sub(r"\s+", "", t["text"].replace("−", "-"))
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            nums.append((float(s), t["bbox"]))
    xs = [(v, (b[0] + b[2]) / 2) for v, b in nums if abs(((b[1] + b[3]) / 2) - oy) < 22 and (b[1] > oy - 2)]
    ys = [(v, (b[1] + b[3]) / 2) for v, b in nums if abs(((b[0] + b[2]) / 2) - ox) < 22 and (b[0] < ox + 2)]
    for name, pairs in (("x", xs), ("y", ys)):
        if len(pairs) >= 2:
            (v0, p0), (v1, p1) = min(pairs), max(pairs)
            if abs(p1 - p0) > 1e-6 and v1 != v0:
                a = (v1 - v0) / (p1 - p0); b0 = v0 - a * p0
                resid = max(abs(a * p + b0 - v) for v, p in pairs)
                if resid < 0.05 * max(abs(v1 - v0), 1):
                    out[name] = dict(scale=a, offset=b0, n_labels=len(pairs))
    return out


# ----------------------------------------------------------------------------- driver
def detect_figures(pdf_path, document_id: str, source_blocks: list[SourceBlock], logical: list[LogicalBlock],
                   pages: list[PageInfo], out_root: Path, dpi: int = 150) -> list[FigureBlock]:
    doc = pymupdf.open(pdf_path)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    ids_by_page: dict[int, list[str]] = {}
    for b in source_blocks:
        ids_by_page.setdefault(b.page_index, []).append(b.block_id)
    figures: list[FigureBlock] = []
    pinfo = {p.page_index: p for p in pages}
    for page in doc:
        pi = pinfo[page.number]
        chars_all = page_chars(page)
        drs = read_drawings(page)
        figs_page: list[FigureBlock] = []
        if pi.route is PageRoute.BORN_DIGITAL:
            _classify(drs, chars_all, page.rect)
            figs_page += _vector_figures(page, drs, chars_all, document_id, ids_by_page.get(page.number, []), source_blocks, len(figures))
        else:
            figs_page += _ink_figures(page, drs, source_blocks, document_id, len(figures))
        figs_page += _raster_figures(page, document_id, len(figures) + len(figs_page), pi.route is PageRoute.BORN_DIGITAL)
        for f in figs_page:
            clip = pymupdf.Rect(f.bbox) + (-4, -4, 4, 4)
            rel = f"figures/{f.figure_id}.png"
            page.get_pixmap(dpi=dpi, clip=clip & page.rect).save(out_root / rel)
            f.crop_ref, f.crop_dpi = rel, dpi
        figures.extend(figs_page)
    # claimed text blocks leave the prose stream
    claimed = {bid: f.figure_id for f in figures for bid in f.claimed_block_ids}
    for lb in logical:
        if lb.member_block_ids and all(m in claimed for m in lb.member_block_ids):
            lb.figure_id = claimed[lb.member_block_ids[0]]
    return figures


def _vector_figures(page, drs, chars_all, document_id, block_ids, source_blocks, offset) -> list[FigureBlock]:
    keep = [d for d in drs if d.role in ("shaft", "head", "circle", "closed", "line", "small", "other")]
    keep = [d for d in keep if d.role not in ("decor", "redaction")]
    clusters = _cluster(keep)
    ordinal_of = {bid: int(bid[-4:]) for bid in block_ids}
    pblocks = {int(b.block_id[-4:]): b for b in source_blocks if b.page_index == page.number}
    out: list[FigureBlock] = []
    # merge clusters that would share label blocks (e.g. two arrows of one diagram far apart)
    bboxes = [pymupdf.Rect(min(d.rect.x0 for d in c), min(d.rect.y0 for d in c), max(d.rect.x1 for d in c), max(d.rect.y1 for d in c)) for c in clusters]
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                gi, gj = _inflate(bboxes[i], LABEL_GAP), _inflate(bboxes[j], LABEL_GAP)
                if gi.intersects(gj) and any(isinstance(d, Dr) and d.role in ("shaft", "line", "circle", "closed") for d in clusters[i] + clusters[j]):
                    shared = [o for o, b in pblocks.items() if gi.intersects(b.bbox) and gj.intersects(b.bbox)]
                    if shared:
                        clusters[i] += clusters[j]; bboxes[i] |= bboxes[j]
                        del clusters[j]; del bboxes[j]; merged = True; break
            if merged:
                break
    for cl, bb in zip(clusters, bboxes):
        structural = [d for d in cl if d.role in ("shaft", "circle", "closed", "line", "other")]
        if not structural or _looks_like_table(cl):
            continue
        if not any(d.role in ("shaft", "circle", "closed", "line") for d in cl):
            continue
        plain = [d for d in structural if d.role in ("line", "shaft", "other") and not getattr(d, "head_at", None)]
        if len(plain) == len(structural) and (len(structural) < 2 or all(_plen(d.pts) < 10 for d in structural)):
            continue                        # a lone short stroke or two is a rule/bar, not a diagram
        # claim short text blocks sitting inside the (inflated) cluster
        region = _inflate(bb, LABEL_GAP)
        claimed = []
        for o, b in pblocks.items():
            if b.block_type != "text":
                continue
            r = pymupdf.Rect(b.bbox)
            txt = b.raw_text or ""
            cells = [ln for ln in txt.split("\n") if ln.strip()]
            short = len(cells) <= 10 and all(len(ln) <= 16 for ln in cells) and len(txt) <= 90
            if r.get_area() > 0 and (r & region).get_area() >= 0.6 * r.get_area() and r.width <= 0.8 * page.rect.width and short:
                claimed.append(o)
        ch = [c for c in chars_all if c.blk in claimed]
        toks = []
        bars = [b for b in page_bars(page) if not any(d.rect.x0 - 1 <= b[0] and b[1] <= d.rect.x1 + 1 and d.rect.y0 - 1 <= b[2] <= d.rect.y1 + 1 and d.role in ("shaft", "line", "head") for d in cl)]
        for g in _tokens(ch):
            lt = _strip_dollars(latex_for_chars(g, bars)).replace("\n", " ").strip()
            toks.append(dict(chars=g, bbox=_bbox(g), text="".join(c.c for c in g), latex=lt))
        fbbox = bb
        for t in toks:
            fbbox = fbbox | pymupdf.Rect(t["bbox"])
        fig = _structure(document_id, offset + len(out), page, cl, toks, fbbox)
        fig.claimed_block_ids = [block_ids[o] for o in sorted(claimed) if o < len(block_ids)]
        out.append(fig)
    return out


def _structure(document_id, n, page, cl: list[Dr], toks: list[dict], bb: pymupdf.Rect) -> FigureBlock:
    fid = f"{document_id}-F{page.number:04d}-{n:03d}"
    shafts = [d for d in cl if d.role == "shaft"]
    lines = [d for d in cl if d.role == "line"]
    circles = [d for d in cl if d.role == "circle"]
    closed = [d for d in cl if d.role == "closed"]
    warns: list[str] = []
    axes = _axes(shafts + lines, toks)
    edges_raw = _build_arrow_edges(cl, toks, None)
    fig = FigureBlock(document_id=document_id, figure_id=fid, page_index=page.number,
                      bbox=tuple(round(v, 3) for v in (bb.x0, bb.y0, bb.x1, bb.y1)), kind="unclassified_drawing",
                      n_drawings=len(cl), provenance=Provenance.RULE_BASED)
    nodes: list[FigureNode] = []; edges: list[FigureEdge] = []
    if axes:
        fig.kind = "function_plot"
        h, v = axes["dr"]
        axis_ids = {id(h), id(v)}
        ticks = [d for d in lines + shafts if id(d) not in axis_ids and _plen(d.pts) < 8]
        curves = []
        for d in cl:
            if id(d) in axis_ids or d.role in ("head",) or d in ticks or _plen(d.pts) < 8:
                continue
            curves.append(d)
        cal = _calibrate(axes, toks, cl)
        ox, oy = axes["origin"]
        fig.axes = dict(origin=[round(ox, 2), round(oy, 2)], x_axis={k: (round(val, 2) if isinstance(val, float) else val) for k, val in axes["x_axis"].items()},
                        y_axis={k: (round(val, 2) if isinstance(val, float) else val) for k, val in axes["y_axis"].items()},
                        n_ticks=len(ticks), calibration=cal,
                        labels=[dict(text=t["text"], latex=t["latex"], bbox=[round(x, 1) for x in t["bbox"]]) for t in toks])
        for d in curves:
            pts = d.pts[::max(1, len(d.pts) // 40)] + [d.pts[-1]]
            rec = dict(style="dashed" if d.dashed else "solid", points_page=[[round(x, 2), round(y, 2)] for x, y in pts])
            if "x" in cal and "y" in cal:
                rec["points_data"] = [[round(cal["x"]["scale"] * x + cal["x"]["offset"], 4), round(cal["y"]["scale"] * y + cal["y"]["offset"], 4)] for x, y in pts]
            curves_rec = rec
            fig.curves.append(curves_rec)
        fig.structure_status = "recovered" if (cal.get("x") and cal.get("y") and fig.curves) else "partial"
        if fig.structure_status == "partial":
            warns.append("plot axes found; data calibration needs >=2 numeric tick labels per axis" if not (cal.get("x") and cal.get("y")) else "no curve found")
    elif len(circles) >= 2 and edges_raw:
        fig.kind = "graph_network"
        cnodes = []
        for i, c in enumerate(sorted(circles, key=lambda d: (d.rect.y0, d.rect.x0))):
            cx, cy = (c.rect.x0 + c.rect.x1) / 2, (c.rect.y0 + c.rect.y1) / 2
            r = max(c.w, c.h) / 2
            inside = [t for t in toks if c.rect.x0 - 1 <= (t["bbox"][0] + t["bbox"][2]) / 2 <= c.rect.x1 + 1 and c.rect.y0 - 1 <= (t["bbox"][1] + t["bbox"][3]) / 2 <= c.rect.y1 + 1]
            label = inside[0] if inside else None
            cnodes.append(dict(id=f"n{i}", cx=cx, cy=cy, r=r, tok=label))
            nodes.append(FigureNode(node_id=f"n{i}", kind="circle", bbox=tuple(round(v, 2) for v in (c.rect.x0, c.rect.y0, c.rect.x1, c.rect.y1)),
                                    text=label["text"] if label else "", latex=label["latex"] if label else None))
        used_tok = {id(n["tok"]) for n in cnodes if n["tok"]}
        weights = [t for t in toks if id(t) not in used_tok]
        for k, e in enumerate(edges_raw):
            def snap(p):
                best = min(cnodes, key=lambda n: math.dist(p, (n["cx"], n["cy"])))
                return best["id"] if math.dist(p, (best["cx"], best["cy"])) <= best["r"] + 5 else None
            s, t_ = snap(e["a"]), snap(e["b"])
            mid = ((e["a"][0] + e["b"][0]) / 2, (e["a"][1] + e["b"][1]) / 2)
            lab = min(weights, key=lambda t: _dist_pt_rect(mid, t["bbox"]), default=None)
            lab = lab if lab and _dist_pt_rect(mid, lab["bbox"]) <= LABEL_GAP else None
            if lab: weights = [w for w in weights if w is not lab]
            ed = FigureEdge(edge_id=f"e{k}", kind="arrow" if e["directed"] else "line", src=s, dst=t_, directed=e["directed"],
                            style=e["style"], label_text=lab["text"] if lab else None, label_latex=lab["latex"] if lab else None,
                            points=[(round(x, 2), round(y, 2)) for x, y in e["pts"][::max(1, len(e["pts"]) // 8)]],
                            bbox=tuple(round(v, 2) for v in (e["rect"].x0, e["rect"].y0, e["rect"].x1, e["rect"].y1)))
            if s is None or t_ is None:
                ed.warnings.append("edge endpoint did not snap to a node")
            edges.append(ed)
        fig.structure_status = "recovered" if edges and all(not e.warnings for e in edges) else "partial"
    elif (closed or circles) and not shafts and not lines:
        fig.kind = "region_diagram"
        regs = sorted(closed + circles, key=lambda d: -d.rect.get_area())
        for i, c in enumerate(regs):
            inside = [t for t in toks if _inside(((t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2), c.pts)]
            nodes.append(FigureNode(node_id=f"r{i}", kind="region", bbox=tuple(round(v, 2) for v in (c.rect.x0, c.rect.y0, c.rect.x1, c.rect.y1)),
                                    text=" ".join(t["text"] for t in inside), latex=" ".join(t["latex"] for t in inside) or None))
        for i, a in enumerate(regs):
            parent_of = None
            for j, b in enumerate(regs):
                if i != j and b.rect.get_area() > a.rect.get_area() and _inside(((a.rect.x0 + a.rect.x1) / 2, (a.rect.y0 + a.rect.y1) / 2), b.pts):
                    if parent_of is None or regs[parent_of].rect.get_area() > b.rect.get_area():
                        parent_of = j
            if parent_of is not None:
                edges.append(FigureEdge(edge_id=f"c{i}", kind="contains", src=f"r{parent_of}", dst=f"r{i}", directed=True,
                                        bbox=nodes[i].bbox))
        used = set()
        for nd in nodes:
            pass
        free = [t for t in toks if not any(_inside(((t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2), c.pts) for c in regs)]
        for k, t in enumerate(free):
            nodes.append(FigureNode(node_id=f"t{k}", kind="text", bbox=tuple(round(v, 2) for v in t["bbox"]), text=t["text"], latex=t["latex"]))
        fig.structure_status = "recovered"
        warns.append("regions are closed curves; membership of points/labels inferred by containment only")
    elif edges_raw and toks:
        fig.kind = "arrow_diagram" if any(e["kind"] == "arrow" for e in edges_raw) else "line_diagram"
        endpoint_tokens: set[int] = set()
        for k, e in enumerate(edges_raw):
            ia = _nearest_token(e["a"], toks); ib = _nearest_token(e["b"], toks, exclude=(ia,) if ia is not None else ())
            e["ia"], e["ib"] = ia, ib
            endpoint_tokens |= {x for x in (ia, ib) if x is not None}
        label_of: dict[int, int] = {}
        for i, t in enumerate(toks):
            if i in endpoint_tokens:
                continue
            best = None
            for k, e in enumerate(edges_raw):
                mid = ((e["a"][0] + e["b"][0]) / 2, (e["a"][1] + e["b"][1]) / 2)
                dd = min(_dist_pt_rect(mid, t["bbox"]), min(_dist_pt_rect(p, t["bbox"]) for p in e["pts"][::max(1, len(e["pts"]) // 10)]))
                if dd <= LABEL_GAP and (best is None or dd < best[0]):
                    best = (dd, k)
            if best:
                label_of[i] = best[1]
        nid = {}
        for i, t in enumerate(toks):
            if i in label_of:
                continue
            nid[i] = f"n{len(nid)}"
            nodes.append(FigureNode(node_id=nid[i], kind="text", bbox=tuple(round(v, 2) for v in t["bbox"]), text=t["text"], latex=t["latex"],
                                    source_block_ids=[]))
        for k, e in enumerate(edges_raw):
            labs = [toks[i] for i, kk in label_of.items() if kk == k]
            ed = FigureEdge(edge_id=f"e{k}", kind=e["kind"], src=nid.get(e["ia"]), dst=nid.get(e["ib"]), directed=e["directed"],
                            style=e["style"], head="arrow" if e["kind"] == "arrow" else None, tail=e.get("tail"),
                            label_text=" ".join(t["text"] for t in labs) or None, label_latex=" ".join(t["latex"] for t in labs) or None,
                            points=[(round(x, 2), round(y, 2)) for x, y in e["pts"][::max(1, len(e["pts"]) // 8)]],
                            bbox=tuple(round(v, 2) for v in (e["rect"].x0, e["rect"].y0, e["rect"].x1, e["rect"].y1)))
            if e.get("both"):
                ed.directed = False; ed.head = "both"; ed.warnings.append("heads at both ends")
            if ed.src is None or ed.dst is None or ed.src == ed.dst:
                ed.warnings.append("edge endpoint did not resolve to two distinct nodes")
            edges.append(ed)
        fig.structure_status = "recovered" if edges and all(not e.warnings for e in edges) else "partial"
    else:
        fig.kind = "unclassified_drawing"
        for k, t in enumerate(toks):
            nodes.append(FigureNode(node_id=f"t{k}", kind="text", bbox=tuple(round(v, 2) for v in t["bbox"]), text=t["text"], latex=t["latex"]))
        warns.append("vector content did not match a known diagram pattern; send crop to VLM")
    fig.nodes, fig.edges, fig.warnings = nodes, edges, warns
    fig.needs_review = fig.structure_status != "recovered"
    return fig


# ----------------------------------------------------------------------------- ink & raster
def _ink_figures(page, drs: list[Dr], source_blocks, document_id, offset) -> list[FigureBlock]:
    """Handwriting pages: strokes that are NOT covered by the (OCR) text blocks are candidate drawings. Not interpreted."""
    pr = page.rect
    ink = [d for d in drs if not _is_grid_or_frame(d, pr) and d.rect.width < 0.8 * pr.width]
    if not ink:
        return []
    text_rects = [pymupdf.Rect(b.bbox) for b in source_blocks if b.page_index == page.number and b.block_type == "text"]
    free = []
    for d in ink:
        r = d.rect
        if r.get_area() == 0:
            continue
        covered = max((((r & _inflate(t, 2)).get_area()) / r.get_area() for t in text_rects), default=0)
        if covered < 0.4:
            free.append(d)
    out: list[FigureBlock] = []
    for cl in _cluster(free):
        bb = pymupdf.Rect(min(d.rect.x0 for d in cl), min(d.rect.y0 for d in cl), max(d.rect.x1 for d in cl), max(d.rect.y1 for d in cl))
        if bb.get_area() < INK_MIN_AREA or bb.height < 38 or bb.width / max(bb.height, 1) > 6 or len(cl) < 5:
            continue
        out.append(FigureBlock(document_id=document_id, figure_id=f"{document_id}-F{page.number:04d}-{offset + len(out):03d}", page_index=page.number,
                               bbox=tuple(round(v, 3) for v in (bb.x0, bb.y0, bb.x1, bb.y1)), kind="ink_drawing", n_drawings=len(cl),
                               structure_status="not_attempted", needs_review=True,
                               warnings=["handwritten strokes outside any text block; located only. Needs VLM/OCR to interpret."]))
    return out


def _raster_figures(page, document_id, offset, born_digital: bool) -> list[FigureBlock]:
    out = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if r.is_empty or r.get_area() > 0.85 * page.rect.get_area():
            continue                                          # a full-page scan is the page, not a figure
        if r.width < 20 or r.height < 20:
            continue
        out.append(FigureBlock(document_id=document_id, figure_id=f"{document_id}-F{page.number:04d}-{offset + len(out):03d}", page_index=page.number,
                               bbox=tuple(round(v, 3) for v in (r.x0, r.y0, r.x1, r.y1)), kind="raster_image", n_drawings=0,
                               structure_status="not_attempted", needs_review=True,
                               warnings=["embedded raster image; content not interpreted (needs VLM/OCR)"]))
    return out


def detect_redactions(pdf_path, document_id: str) -> list[dict]:
    """Solid black filled rectangles (typical redaction boxes). They hide content; record where."""
    out = []
    for page in pymupdf.open(pdf_path):
        for d in read_drawings(page):
            if d.fill and all(c < 0.12 for c in d.fill) and len(d.items) == 1 and d.items[0][0] == "re" and d.w >= 12 and d.h >= 4 \
                    and d.w < 0.9 * page.rect.width:
                out.append(dict(document_id=document_id, page_index=page.number,
                                bbox=[round(v, 2) for v in (d.rect.x0, d.rect.y0, d.rect.x1, d.rect.y1)],
                                note="solid black rectangle: content likely redacted or hidden"))
    return out