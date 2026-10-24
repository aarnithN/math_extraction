"""Deterministic recovery of math symbols and LaTeX from PDF glyph data (PyMuPDF `rawdict`).

How it works (no OCR, no model):
  1. Every glyph is read with its Unicode value, font, size, origin and box.
  2. Each glyph is mapped to a LaTeX token using the symbol table plus FONT-AWARE rules for glyphs the parser
     reports wrongly (e.g. CMSY '7' = \\mapstochar, CMMI ',' that is really \\lhook). Unknown glyphs are returned
     UNMAPPED and flagged; they are never guessed.
  3. Geometry restores structure: sub/superscripts (size + baseline offset), fractions and radicals (drawn rules),
     accents, negation overlays, composite/long arrows (with labels above/below), `\\cong` (stacked ~ and =).
  4. Each text line is split into text and math zones; every zone becomes a MathExpression with symbols,
     deterministic LaTeX and honest status ('partial' + `unsupported` when structure was not reconstructed).

Limits: only reliable for TeX-like PDFs with real fonts. Matrices, big delimiters, multi-level nested scripts,
aligned multi-line math and text/math boundaries in non-TeX fonts are flagged, not solved. Verify with
latex_verify.py and/or a VLM/OCR pass.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

import pymupdf

from .schemas import LogicalBlock, MathExpression, MathSymbol, PageInfo, PageRoute, SourceBlock
from .symbols_table import (ACCENTS, ASCII_MATH, BLACKBOARD, CALLIGRAPHIC, FRAKTUR, MATH_ITALIC, NEGATED,
                            OPERATOR_MACROS, UNICODE_MACROS, decode_math_alnum, font_family, font_kind,
                            unicode_is_mathy)

SCRIPT_RATIO = 0.82           # glyph size / main size below which a glyph is a script
REL_N = set("=<>+-:")         # neutral chars that carry relation/operator spacing in math
PUNCT_TRIM = set(",.;:!?")
OPEN, CLOSE = set("([") | {"\\{", "\\langle", "\\lfloor", "\\lceil"}, set(")]") | {"\\}", "\\rangle", "\\rfloor", "\\rceil"}
CMEX_OPS = {**{k: "\\sum" for k in "PX"}, **{k: "\\prod" for k in "QY"}, **{k: "\\int" for k in "RZ"},
            **{k: "\\bigcup" for k in "S["}, **{k: "\\bigcap" for k in "T\\"}, **{k: "\\biguplus" for k in "U]"},
            **{k: "\\bigwedge" for k in "V^"}, **{k: "\\bigvee" for k in "W_"}, **{k: "\\coprod" for k in "`a"}}
EXTRA = {"ı": ("\\imath", "letter"), "ȷ": ("\\jmath", "letter")}


@dataclass
class Ch:
    c: str
    x0: float; y0: float; x1: float; y1: float
    ox: float; oy: float
    size: float
    font: str
    fam: str
    fkind: str
    blk: int = -1
    dead: bool = False
    tok: Optional[str] = None       # LaTeX token once resolved
    cls: Optional[str] = None
    src: str = "table"
    mapped: bool = True
    unsup: Optional[str] = None
    accent: Optional[str] = None
    neg: bool = False
    synthetic: bool = False
    label_over: list = field(default_factory=list)
    label_under: list = field(default_factory=list)
    structs: set = field(default_factory=set)
    sub_symbols: list = field(default_factory=list)   # symbols of reconstructed sub-structures
    sup: list = field(default_factory=list)
    sub: list = field(default_factory=list)
    kind: str = "T"                 # M (math), T (text), N (neutral)

    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2
    @property
    def w(self): return self.x1 - self.x0


# ----------------------------------------------------------------------------- extraction
def page_chars(page: pymupdf.Page) -> list[Ch]:
    out: list[Ch] = []
    for bi, b in enumerate(page.get_text("rawdict")["blocks"]):
        if b["type"] != 0:
            continue
        for ln in b["lines"]:
            dx, dy = ln.get("dir", (1, 0))
            if abs(dy) > 0.1 or dx < 0:
                continue                                   # rotated text: out of scope here
            for sp in ln["spans"]:
                fam = font_family(sp["font"])
                for ch in sp["chars"]:
                    if ch["c"].isspace():
                        continue
                    x0, y0, x1, y1 = ch["bbox"]
                    ox, oy = ch["origin"]
                    out.append(Ch(ch["c"], x0, y0, x1, y1, ox, oy, sp["size"], sp["font"], fam, font_kind(fam), bi))
    return out


def page_bars(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    """Thin horizontal rules (fraction bars, radical tops, overlines) as (x0, x1, y)."""
    bars = []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l" and abs(it[1].y - it[2].y) < 0.3 and abs(it[1].x - it[2].x) >= 2.5:
                bars.append((min(it[1].x, it[2].x), max(it[1].x, it[2].x), (it[1].y + it[2].y) / 2))
            elif it[0] == "re":
                r = it[1]
                if r.height <= 1.3 and r.width >= 2.5:
                    bars.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
    return bars


# ----------------------------------------------------------------------------- glyph -> token
def map_char(ch: Ch) -> None:
    """Fill ch.tok/cls/src/mapped/unsup (font-aware, never guessing)."""
    c, fam = ch.c, ch.fam
    ch.mapped, ch.src, ch.unsup = True, "table", None
    if ch.synthetic:
        return
    if c == "\u0338":
        ch.tok, ch.cls, ch.src = "", "ord", "font_rule"; return
    if fam in ("CMEX", "LMMathExtension", "MTEX") and c in "()[]":
        ch.tok, ch.cls, ch.src = ("\\bigl" + c if c in "([" else "\\bigr" + c), ("open" if c in "([" else "close"), "font_rule"
        ch.structs.add("tall_delimiter"); return
    if fam == "CMEX" and c in CMEX_OPS:
        ch.tok, ch.cls, ch.src = CMEX_OPS[c], "op", "font_rule"; return
    if fam == "CMEX":
        if c == "\x0e":
            ch.tok, ch.cls, ch.src = "\\big/", "bin", "font_rule"; return
        if 0xF8EB <= ord(c) <= 0xF8FF or c in "{}|\ufe02":
            ch.tok, ch.cls, ch.src, ch.mapped, ch.unsup = "", "ord", "unmapped", False, "tall_delimiter_or_brace_pieces"; return
    if fam in BLACKBOARD and c.isascii() and c.isalpha():
        ch.tok, ch.cls, ch.src = f"\\mathbb{{{c}}}", "letter", "font_rule"; return
    if fam in CALLIGRAPHIC and c.isascii() and c.isupper():
        ch.tok, ch.cls, ch.src = f"\\mathcal{{{c}}}", "letter", "font_rule"; return
    if fam in FRAKTUR and c.isascii() and c.isalpha():
        ch.tok, ch.cls, ch.src = f"\\mathfrak{{{c}}}", "letter", "font_rule"; return
    if fam in ("CMSY", "CMBSY") and c == "p":
        ch.tok, ch.cls, ch.src = "\\surd", "ord", "font_rule"; return
    if fam in ("CMSY", "CMBSY") and c == "7":
        ch.tok, ch.cls, ch.src = "\\mapstochar", "arrow", "font_rule"; return
    if fam in MATH_ITALIC and c in ",-" and ch.w > 0.6 * ch.size:
        ch.tok, ch.cls, ch.src = ("\\lhook" if c == "," else "\\rhook"), "arrow", "font_rule"; return
    if "\ufe00" <= c <= "\ufe0f":
        ch.tok, ch.cls, ch.src = "", None, "font_rule"; return          # zero-width companion of a big operator/delimiter
    if c in ACCENTS:
        ch.tok, ch.cls, ch.src = ACCENTS[c], "accent", "font_rule"; return
    if c in "’‘“”–—‐‑":
        ch.tok, ch.cls, ch.src = c, "punct", "literal"; return          # prose punctuation, not math
    if c in EXTRA:
        ch.tok, ch.cls = EXTRA[c]; return
    if c in UNICODE_MACROS:
        ch.tok, ch.cls = UNICODE_MACROS[c]; return
    dec = decode_math_alnum(c)
    if dec:
        ch.tok, ch.cls = dec; ch.src = "font_rule"; return
    if c in ASCII_MATH and (c in "{}\\%&#_$~^" or ch.fkind != "roman" or True):
        ch.tok, ch.cls = ASCII_MATH[c]; ch.src = "literal"; return
    if c.isascii() and c.isalpha():
        ch.tok, ch.cls, ch.src = c, "letter", "literal"; return
    if c.isascii() and c.isdigit():
        ch.tok, ch.cls, ch.src = c, "digit", "literal"; return
    if c.isalpha() and not (0xE000 <= ord(c) <= 0xF8FF):
        ch.tok, ch.cls, ch.src = c, "letter", "literal"; return      # accented/other text letters
    ch.tok, ch.cls, ch.src, ch.mapped = "", None, "unmapped", False
    ch.unsup = f"unmapped_glyph U+{ord(c):04X} in {ch.font}"


def _kind(ch: Ch) -> str:
    if ch.synthetic:
        return "M"
    if ch.fkind in ("mathital", "mathsym"):
        return "M"
    if ch.fkind == "roman":
        if ch.c in UNICODE_MACROS and UNICODE_MACROS[ch.c][1] in ("greek", "rel", "bin", "arrow", "log", "op", "ord"):
            return "M"
        return "T" if ch.c.isalpha() or ord(ch.c) > 127 else "N"
    return "M" if unicode_is_mathy(ch.c) else ("T" if ch.c.isalpha() else "N")


def _sym(ch: Ch, role: str) -> MathSymbol:
    return MathSymbol(char=ch.c, code=ord(ch.c) if len(ch.c) == 1 else 0, macro=ch.tok or None, cls=ch.cls, role=role,
                      font=ch.font.split("+")[-1], bbox=(round(ch.x0, 2), round(ch.y0, 2), round(ch.x1, 2), round(ch.y1, 2)),
                      mapped=ch.mapped, source=ch.src)


# ----------------------------------------------------------------------------- analysis result
@dataclass
class Zone:
    items: list
    latex: str
    text: str
    symbols: list
    structures: list
    unsupported: list
    bbox: tuple
    block_ordinals: set
    chain: Optional[dict] = None


@dataclass
class Analysis:
    latex_text: str
    zones: list


def _med(v, default):
    return statistics.median(v) if v else default


# ----------------------------------------------------------------------------- structure passes
def _accents(chs: list[Ch], flags: set) -> None:
    for a in chs:
        if a.dead or a.c not in ACCENTS or a.fkind not in ("roman", "mathital", "mathsym", "other"):
            continue
        best, bd = None, 1e9
        for b in chs:
            if b is a or b.dead or b.c in ACCENTS or not (b.c.isalnum() or b.c in UNICODE_MACROS):
                continue
            if abs(a.cx - b.cx) <= max(0.45 * b.w, 1.5) and a.cy < b.cy - 0.2 * b.size and b.cy - a.cy < 1.3 * b.size:
                d = abs(a.cx - b.cx) + abs(a.cy - b.cy) * 0.1
                if d < bd:
                    best, bd = b, d
        if best is not None:
            best.accent = ACCENTS[a.c]; a.dead = True; flags.add("accent")


def _composites(chs: list[Ch], flags: set) -> None:
    alive = [c for c in chs if not c.dead]
    for sl in [c for c in alive if c.c == "\u0338" or (c.c == "/" and c.fkind in ("mathital", "mathsym"))]:
        best, bd = None, 1e9
        for p in alive:
            if p is sl or p.dead or p.c not in NEGATED or p.c == "/":
                continue
            if sl.c == "\u0338":
                ok = p.x0 - 0.8 * p.size <= sl.x0 <= p.x1 and abs(p.cy - sl.cy) <= 0.9 * p.size
            else:
                ok = p.x0 + 0.15 * p.w < sl.x0 < p.x1 - 0.15 * p.w and abs(p.cy - sl.cy) <= 0.8 * p.size
            if ok and abs(sl.x0 - p.x0) < bd:
                best, bd = p, abs(sl.x0 - p.x0)
        if best is not None:
            best.neg = True; sl.dead = True; flags.add("negation")
    for t in [c for c in alive if c.c == "∼" and not c.dead]:
        for e in alive:
            if e.dead or e.c != "=" or e is t:
                continue
            ov = min(t.x1, e.x1) - max(t.x0, e.x0)
            if ov >= 0.6 * min(t.w, e.w) and abs(t.oy - e.oy) < 0.7 * t.size:
                t.c, t.src = "≅", "composite"; e.dead = True; flags.add("composite_symbol"); break


def _arrows(chs: list[Ch], main: float, flags: set) -> None:
    sym = {"−": "-", "→": ">", "←": "<", "=": "e", "⇒": "R", "⇐": "l"}
    cand = []
    for c in chs:
        if c.dead:
            continue
        if c.fam in ("CMSY", "CMBSY") and c.c == "7":
            cand.append((c, "M"))
        elif c.fam in MATH_ITALIC and c.c == ",":
            cand.append((c, "L"))          # only becomes a hook if glued to a following arrow (see pattern)
        elif c.fam in MATH_ITALIC and c.c == "-":
            cand.append((c, "H"))
        elif c.c in sym and c.size >= SCRIPT_RATIO * main:
            cand.append((c, sym[c.c]))
    cand.sort(key=lambda t: t[0].cx)
    runs: list[list] = []
    for c, sy in cand:
        for r in runs:
            last = r[-1][0]
            if abs(c.oy - last.oy) < 0.2 * main and c.x0 <= max(x[0].x1 for x in r) + 0.2 * main and c.x1 >= min(x[0].x0 for x in r) - 0.2 * main:
                r.append((c, sy)); break
        else:
            runs.append([(c, sy)])
    pat = re.compile(r"[LM]?-+>|[LM]>|<-*>|H?<-+|l=*R|=+R|l=+")
    names = {"->": "\\longrightarrow", "L->": "\\hookrightarrow", "L>": "\\hookrightarrow", "M->": "\\longmapsto", "M>": "\\mapsto",
             "<>": "\\longleftrightarrow", "<->": "\\longleftrightarrow", "<-": "\\longleftarrow", "H<-": "\\hookleftarrow",
             "l=R": "\\Longleftrightarrow", "lR": "\\Longleftrightarrow", "=R": "\\Longrightarrow", "l=": "\\Longleftarrow"}
    for run in runs:
        st = "".join(x[1] for x in run)
        for m in pat.finditer(st):
            grp = [run[i][0] for i in range(m.start(), m.end())]
            key = re.sub(r"-+", "-", re.sub(r"=+", "=", m.group(0)))
            if key not in names or any(g.dead for g in grp):
                continue
            tok = names[key]
            x0, x1 = min(c.x0 for c in grp), max(c.x1 for c in grp)
            ay = _med([c.cy for c in grp], grp[0].cy)
            labels_o = [c for c in chs if not c.dead and c not in grp and c.size < SCRIPT_RATIO * main
                        and x0 - 2 <= c.cx <= x1 + 2 and ay - 1.7 * main < c.cy < ay - 0.25 * main]
            labels_u = [c for c in chs if not c.dead and c not in grp and c.size < SCRIPT_RATIO * main
                        and x0 - 2 <= c.cx <= x1 + 2 and ay + 0.25 * main < c.cy < ay + 1.7 * main]
            first = grp[0]
            first.synthetic, first.tok, first.cls, first.src, first.mapped = True, tok, "arrow", "composite", True
            first.x0, first.x1 = x0, x1
            first.label_over, first.label_under = labels_o, labels_u
            for c in grp[1:] + labels_o + labels_u:
                c.dead = True
            first.structs.add("long_arrow" if tok != "\\mapsto" else "composite_symbol")
            flags.add("composite_arrow")
            if labels_o or labels_u:
                flags.add("labelled_arrow")


def _underbrace(chs: list[Ch], bars: list, flags: set, depth: int):
    """CMEX brace pieces ('|','{','z','}') + two drawn rules = \\underbrace{content}_{label}. Returns (synth, bars_left)."""
    pcs = [c for c in chs if not c.dead and c.fam == "CMEX" and c.c in "|{z}"]
    if len(pcs) < 3:
        return [], bars
    xl, xr = min(c.x0 for c in pcs), max(c.x1 for c in pcs)
    by = _med([c.cy for c in pcs], pcs[0].cy)
    size = _med([c.size for c in pcs], 10.0)
    keep = [b for b in bars if not (abs(b[2] - by) <= 4 and b[0] >= xl - 3 and b[1] <= xr + 3)]
    above = [c for c in chs if not c.dead and c not in pcs and xl - 8 <= c.cx <= xr + 8 and by - 2.2 * size <= c.cy < by - 0.2 * size]
    below = [c for c in chs if not c.dead and c not in pcs and xl - 8 <= c.cx <= xr + 8 and by + 0.2 * size < c.cy <= by + 2.2 * size]
    avg = lambda v: sum(c.size for c in v) / len(v) if v else 0
    over = avg(below) > avg(above) + 0.5          # bigger text below the brace => the brace is OVER the content
    body, lab = (below, above) if over else (above, below)
    body = [c for c in body if xl - 2 <= c.cx <= xr + 2]
    while body and max(body, key=lambda c: c.x1).c in ".,;" and max(c.x1 for c in body) > xr - 1.5:
        body.remove(max(body, key=lambda c: c.x1))
    if not body:
        return [], keep
    bl, bs, _ = _render_math(body, keep, depth + 1)
    ll, ls, _ = _render_math(lab, keep, depth + 1) if lab else ("", [], set())
    for x in bs: x.role = "over"
    for x in ls: x.role = "under"
    syn = _synth((f"\\overbrace{{{bl}}}" + (f"^{{{ll}}}" if ll else "")) if over else (f"\\underbrace{{{bl}}}" + (f"_{{{ll}}}" if ll else "")),
                 body + lab + pcs, "overbrace" if over else "underbrace", size)
    syn.oy = _med([c.oy for c in body], body[0].oy)
    syn.sub_symbols = bs + ls
    syn.structs.add("overbrace" if over else "underbrace")
    for c in body + lab + pcs: c.dead = True
    flags.add("underbrace")
    return [syn], keep


def _bars(chs: list[Ch], bars: list, flags: set, depth: int) -> list[Ch]:
    """Fractions, radicals and overlines from drawn rules. Returns synthetic chars (already removed from stream)."""
    synth: list[Ch] = []
    if depth > 2:
        return synth
    for bar in sorted(bars, key=lambda b: -(b[1] - b[0])):
        bx0, bx1, by = bar
        bars = [b for b in bars if b is not bar]
        alive = [c for c in chs if not c.dead and not c.synthetic]
        near = [c for c in alive if bx0 - 25 <= c.cx <= bx1 + 25 and abs(c.cy - by) < 30]
        if not near:
            continue
        size = _med([c.size for c in near], 10.0)
        rad = [c for c in alive if (c.c == "√" or (c.c == "p" and c.fam in ("CMSY", "CMBSY"))) and abs(c.x1 - bx0) <= 2.5 and abs(c.y0 - by) <= 0.6 * size + 3]
        if rad:
            under = [c for c in alive if c is not rad[0] and bx0 - 1 <= c.cx <= bx1 + 1 and c.y0 >= by - 1 and c.y1 <= by + 2.3 * size]
            if under:
                latex, sym, _fl = _render_math(under, bars, depth + 1)
                s = _synth(f"\\sqrt{{{latex}}}", under + rad, "radical", rad[0].size)
                s.oy = _med([c.oy for c in under], rad[0].oy)
                s.sub_symbols = sym
                for c in under + rad: c.dead = True
                s.structs.add("sqrt"); flags.add("sqrt"); synth.append(s)
            continue
        num = [c for c in alive if bx0 - 2 <= c.cx <= bx1 + 2 and by - 1.8 * size <= c.y0 and c.y1 <= by + 0.6 and by - c.y1 <= 0.9 * size]
        den = [c for c in alive if bx0 - 2 <= c.cx <= bx1 + 2 and c.y0 >= by - 0.6 and c.y1 <= by + 1.8 * size + 4 and c.y0 - by <= 0.9 * size]
        if num and den:
            nw = max(c.x1 for c in num) - min(c.x0 for c in num); dw = max(c.x1 for c in den) - min(c.x0 for c in den)
            bw = bx1 - bx0
            ncx = (min(c.x0 for c in num) + max(c.x1 for c in num)) / 2; dcx = (min(c.x0 for c in den) + max(c.x1 for c in den)) / 2
            bcx = (bx0 + bx1) / 2
            hugs = (bw <= 160 and max(nw, dw) >= 0.6 * bw and max(nw, dw) <= bw + 6
                    and abs(ncx - bcx) <= 0.25 * bw + 2 and abs(dcx - bcx) <= 0.25 * bw + 2)
            if not hugs:
                continue                       # a long rule over/under unrelated text (header rule, table rule): not a fraction
            nl, ns, _ = _render_math(num, bars, depth + 1)
            dl, ds, _ = _render_math(den, bars, depth + 1)
            for s in ns: s.role = "num"
            for s in ds: s.role = "den"
            s = _synth(f"\\frac{{{nl}}}{{{dl}}}", num + den, "fraction", max(c.size for c in num + den))
            s.cy_bar = by
            s.sub_symbols = ns + ds
            s.x0, s.x1 = min(s.x0, bx0), max(s.x1, bx1)
            for c in num + den: c.dead = True
            s.structs.add("fraction"); flags.add("fraction"); synth.append(s)
        elif den and not num:
            ov = [c for c in alive if bx0 - 1 <= c.cx <= bx1 + 1 and 0 <= c.y0 - by <= 0.5 * size]
            if ov:
                latex, sym, _ = _render_math(ov, bars, depth + 1)
                s = _synth(f"\\overline{{{latex}}}", ov, "overline", size)
                s.sub_symbols = sym
                for c in ov: c.dead = True
                s.structs.add("overline"); flags.add("overline"); synth.append(s)
    return synth


def _synth(tok: str, parts: list[Ch], why: str, size: float) -> Ch:
    x0, y0 = min(c.x0 for c in parts), min(c.y0 for c in parts)
    x1, y1 = max(c.x1 for c in parts), max(c.y1 for c in parts)
    s = Ch(why, x0, y0, x1, y1, x0, y1, size, parts[0].font, parts[0].fam, "mathsym", parts[0].blk)
    s.synthetic, s.tok, s.cls, s.src, s.kind = True, tok, "ord", "composite", "M"
    s.c = why
    return s


# ----------------------------------------------------------------------------- line / script structure
def _render_math(chs: list[Ch], bars: list, depth: int = 0) -> tuple[str, list[MathSymbol], set]:
    """Render a char set as one math expression (used for fraction parts, radicands, arrow labels)."""
    a = analyze([_copy(c) for c in chs], bars, depth=depth, force_math=True)
    z = a.zones[0] if a.zones else None
    return (z.latex if z else ""), (z.symbols if z else []), set(z.structures if z else [])


def _copy(c: Ch) -> Ch:
    d = Ch(**{k: getattr(c, k) for k in ("c", "x0", "y0", "x1", "y1", "ox", "oy", "size", "font", "fam", "fkind", "blk")})
    d.synthetic, d.tok, d.cls, d.src, d.mapped, d.structs = c.synthetic, c.tok, c.cls, c.src, c.mapped, set(c.structs)
    return d


def _token(ch: Ch) -> str:
    t = ch.tok or ""
    if ch.neg and ch.c in NEGATED:
        t = NEGATED[ch.c]
    if ch.accent:
        t = f"{ch.accent}{{{t}}}"
    return t


def _script_tok(chars: list[Ch]) -> str:
    chars = sorted(chars, key=lambda c: c.ox)
    toks = [_token(c) for c in chars]
    if toks and all(t == "\\prime" for t in toks):
        return "'" * len(toks)
    out = ""
    for t in toks:
        if not t:
            continue
        if out and re.search(r"\\[A-Za-z]+$", out) and t[0].isalnum():
            out += " "
        out += t
    return out


def analyze(chs: list[Ch], bars: list, depth: int = 0, force_math: bool = False) -> Analysis:
    chs = [c for c in chs if not c.c.isspace()]
    if not chs:
        return Analysis("", [])
    for c in chs:
        map_char(c)
        c.kind = _kind(c)
    for c in chs:
        if c.cls == "op" and not c.synthetic:
            c.oy = c.cy + 0.25 * c.size          # big operators are centred on the math axis, not on the baseline
    flags: set[str] = set()
    sizes = [round(c.size, 1) for c in chs]
    cands = [z for z in set(sizes) if sizes.count(z) >= 0.2 * len(sizes)]
    main = max(cands) if cands else max(sizes)
    _accents(chs, flags)
    for c in chs:
        if "\ufe00" <= c.c <= "\ufe0f":
            c.dead = True
    ub, bars = _underbrace(chs, bars, flags, depth)
    synth = ub + _bars(chs, bars, flags, depth)
    _composites(chs, flags)
    _arrows(chs, main, flags)
    for c in chs:
        if c.synthetic and c.tok and c.label_over + c.label_under:
            lo = _render_math(c.label_over, bars, depth + 1) if c.label_over else ("", [], set())
            lu = _render_math(c.label_under, bars, depth + 1) if c.label_under else ("", [], set())
            if c.tok == "\\longrightarrow":
                c.tok = (f"\\xrightarrow[{lu[0]}]{{{lo[0]}}}" if lu[0] else f"\\xrightarrow{{{lo[0]}}}")
            elif c.tok in ("\\longleftarrow",):
                c.tok = (f"\\xleftarrow[{lu[0]}]{{{lo[0]}}}" if lu[0] else f"\\xleftarrow{{{lo[0]}}}")
            else:
                c.tok = f"\\overset{{{lo[0]}}}{{{c.tok}}}" if lo[0] and not lu[0] else c.tok
            c.sub_symbols = [*lo[1], *lu[1]]
            for s in lo[1]: s.role = "over"
            for s in lu[1]: s.role = "under"
    alive = [c for c in chs if not c.dead] + synth
    if not alive:
        return Analysis("", [])
    # main / script split
    mains, scripts = [], []
    for c in alive:
        if c.synthetic and getattr(c, "cy_bar", None) is not None:
            scripts.append(c) if False else mains.append(c)   # fractions placed after lines are known
        elif c.size >= SCRIPT_RATIO * main or c.synthetic:
            mains.append(c)
        else:
            scripts.append(c)
    frac = [c for c in mains if getattr(c, "cy_bar", None) is not None]
    mains = [c for c in mains if c not in frac]
    # lines by baseline
    lines: list[list[Ch]] = []
    for c in sorted(mains, key=lambda c: c.oy):
        if lines and abs(c.oy - _med([x.oy for x in lines[-1]], c.oy)) <= 0.5 * main:
            lines[-1].append(c)
        else:
            lines.append([c])
    if not lines and (scripts or frac):
        for c in frac + scripts:
            lines.append([c]); break
        mains = [lines[0][0]]
        scripts = [s for s in scripts if s is not mains[0]]; frac = [f for f in frac if f is not mains[0]]
    base_y = [_med([c.oy for c in L], 0) for L in lines]

    def line_of(c: Ch) -> int:
        y = c.oy if getattr(c, "cy_bar", None) is None else c.cy_bar + 0.25 * main
        return min(range(len(lines)), key=lambda i: abs(base_y[i] - y))

    for f in frac:                                 # role of a fraction: main / sup / sub of its line
        i = line_of(f); axis = base_y[i] - 0.25 * main; dy = axis - f.cy_bar
        if f.size >= 0.62 * main and abs(dy) <= 0.45 * main:
            lines[i].append(f)
        else:
            f.role_hint = "sup" if dy > 0 else "sub"; scripts.append(f)
    nested = False
    for s in scripts:
        i = line_of(s)
        L = sorted(lines[i], key=lambda c: c.ox)
        y = getattr(s, "oy", s.oy) if not hasattr(s, "cy_bar") else s.cy_bar + 0.25 * main
        dy = base_y[i] - y
        if s.size < 0.62 * main and not s.synthetic:
            nested = True
        if abs(dy) < 0.12 * main and not hasattr(s, "role_hint"):
            L.append(s); lines[i] = L; continue
        role = getattr(s, "role_hint", None) or ("sup" if dy > 0 else "sub")
        ops = [b for b in L if b.cls == "op" and b.x0 - 1 <= s.cx <= b.x1 + 1]
        left = [b for b in L if b.x1 <= s.x0 + 0.35 * main and b is not s]
        base = ops[0] if ops else (max(left, key=lambda b: b.x1) if left else None)
        if base is None:
            s.role_orphan = True; L.append(s); lines[i] = L; continue
        (base.sup if role == "sup" else base.sub).append(s)
    if nested:
        flags.add("nested_scripts_flattened")
    return _build(lines, base_y, main, flags, bars, depth, force_math)


# ----------------------------------------------------------------------------- tokens, zones, text
def _item_info(c: Ch):
    syms, structs, unsup = [], set(c.structs), []
    syms.append(_sym(c, "base"))
    syms[-1].macro = _token(c) if not c.synthetic else (c.tok or None)
    if c.neg: structs.add("negation")
    if c.accent: structs.add("accent")
    if c.unsup: unsup.append(c.unsup)
    syms.extend(c.sub_symbols)
    tok = _token(c)
    for role, group in (("sup", c.sup), ("sub", c.sub)):
        if group:
            structs.add("script")
            for g in group:
                map_char(g) if not g.synthetic else None
                gs = _sym(g, role)
                syms.append(gs)
                syms.extend(g.sub_symbols)
                if g.unsup: unsup.append(g.unsup)
                structs |= g.structs
            inner = _script_tok(group)
            if role == "sup" and inner and set(inner) == {"'"}:
                tok += inner
            else:
                tok += ("^" if role == "sup" else "_") + "{" + inner + "}"
    return tok, syms, structs, unsup


def _build(lines, base_y, main, flags, bars, depth, force_math) -> Analysis:
    all_text, zones = [], []
    order = sorted(range(len(lines)), key=lambda i: base_y[i])
    for li in order:
        L = sorted(lines[li], key=lambda c: c.ox if not c.synthetic else c.x0)
        items = []
        for c in L:
            if not c.synthetic:
                map_char(c)
            tok, syms, structs, unsup = _item_info(c)
            if c.synthetic and not c.sub_symbols and c.cls != "arrow":
                pass
            kind = c.kind if c.kind in "MTN" else "T"
            has_script = bool(c.sup or c.sub)
            stok = tok[len(_token(c)):] if has_script else ""
            ext = max([c.x1] + [g.x1 for g in c.sup + c.sub])
            items.append(dict(ch=c, x0=c.x0, x1=max(ext, c.x0), size=c.size, kind=kind, tok=tok, syms=syms, structs=structs,
                              unsup=unsup, text=c.c if not c.synthetic else "", word=None, has_script=has_script, stok=stok))
        for k, it in enumerate(items):               # f\colon A  (tight before, thin after)  vs  relation ':'
            if it["ch"].c == ":" and it["tok"] == ":" and k > 0 and not it["ch"].synthetic:
                if it["x0"] - items[k - 1]["x1"] <= 0.18 * it["size"] and (k + 1 >= len(items) or items[k + 1]["x0"] - it["x1"] <= 0.32 * it["size"]):
                    it["tok"] = "\\colon"; it["syms"][0].macro = "\\colon"
        # text of scripts for the unicode string
        for it in items:
            if it["ch"].synthetic:
                it["text"] = "".join(x.char for x in it["syms"] if x.role == "base" and x is not it["syms"][0]) or it["tok"]
            extra = "".join(g.c for g in sorted(it["ch"].sup + it["ch"].sub, key=lambda g: g.ox) if not g.synthetic)
            it["text"] += extra
        if force_math:
            _mark_math(items, main, force=True)
            z = _zone(items, 0, len(items), main)
            zones.append(z); all_text.append(z.latex); continue
        flags_local = _mark_math(items, main)
        i = 0
        out = []
        while i < len(items):
            if flags_local[i]:
                j = i
                while j + 1 < len(items) and flags_local[j + 1] and items[j + 1]["x0"] - items[j]["x1"] < 1.2 * main:
                    if items[j]["ch"].c in ",;." and items[j]["kind"] == "N" and items[j + 1]["x0"] - items[j]["x1"] > 0.26 * items[j]["size"]:
                        break
                    j += 1
                a, b = _trim(items, i, j, flags_local)
                for k in range(i, a): out.append(("text", items[k]))
                if a <= b and any(items[k]["kind"] == "M" or items[k]["word"] for k in range(a, b + 1)):
                    z = _zone(items, a, b + 1, main)
                    zones.append(z)
                    out.append(("zone", items[a], z, items[b]))
                else:
                    for k in range(a, b + 1): out.append(("text", items[k]))
                for k in range(b + 1, j + 1): out.append(("text", items[k]))
                i = j + 1
            else:
                out.append(("text", items[i])); i += 1
        all_text.append(_join_text(out, main))
    return Analysis("\n".join(t for t in all_text if t), zones)


def _mark_math(items: list[dict], main: float, force: bool = False) -> list[bool]:
    n = len(items)
    m = [force or it["kind"] == "M" for it in items]
    words: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        if items[i]["kind"] == "T" and items[i]["ch"].c.isalpha():
            j = i
            while j + 1 < n and items[j + 1]["kind"] == "T" and items[j + 1]["x0"] - items[j]["x1"] <= 0.22 * items[j]["size"] and not items[j]["has_script"]:
                j += 1
            words.append((i, j, "".join(items[k]["ch"].c for k in range(i, j + 1))))
            i = j + 1
        else:
            i += 1
    changed = not force
    while changed:
        changed = False
        for i, it in enumerate(items):
            if m[i] or it["kind"] != "N":
                continue
            c0 = it["ch"].c
            for j in (i - 1, i + 1):
                if 0 <= j < n and m[j]:
                    nb = items[j]["ch"]
                    spaced = c0 in REL_N or nb.c in REL_N or nb.cls in ("rel", "bin", "arrow") or items[j]["tok"] in ("=", "<", ">")
                    thr = (0.45 if spaced else 0.2) * it["size"]
                    gap = (it["x0"] - items[j]["x1"]) if j < i else (items[j]["x0"] - it["x1"])
                    if c0 == ":" and j > i:
                        continue            # a colon joins a zone only from the left ("f : G -> H"), not "Item 1: $x$"
                    if gap <= thr:
                        m[i] = True; changed = True; break
        for (a, b, w) in words:
            if m[a]:
                continue
            L, R = a - 1, b + 1
            gl = items[a]["x0"] - items[L]["x1"] if L >= 0 else 9e9
            gr = items[R]["x0"] - items[b]["x1"] if R < n else 9e9
            sz = items[a]["size"]
            opn = w in OPERATOR_MACROS
            adj = (L >= 0 and m[L] and gl <= (0.3 if opn else 0.2) * sz) or (R < n and m[R] and gr <= (0.3 if opn else 0.2) * sz)
            tight = (L >= 0 and m[L] and gl <= 0.1 * sz) or (R < n and m[R] and gr <= 0.1 * sz)
            paren = R < n and items[R]["ch"].c == "(" and gr <= 0.1 * sz
            colon = (R + 1 < n and items[R]["ch"].c == ":" and gr <= 0.25 * sz and m[R + 1]
                     and items[R + 1]["x0"] - items[R]["x1"] <= 0.45 * sz)
            if (opn and (adj or paren or colon)) or (colon and len(w) <= 5) or (items[b]["has_script"] and len(w) <= 5) or (len(w) <= 4 and adj and (not w[0].isupper() or tight)):
                for k in range(a, b + 1):
                    m[k] = True
                changed = True
    for (a, b, w) in words:
        if not all(m[k] for k in range(a, b + 1)) or (len(w) == 1 and not force and not (items[a]["has_script"] or True)):
            continue
        if len(w) == 1 and force:
            continue
        last = items[b]
        tok = OPERATOR_MACROS.get(w) or (f"\\mathrm{{{w}}}" if len(w) == 1 else (f"\\text{{{w}}}" if force and len(w) >= 4 else f"\\operatorname{{{w}}}"))
        first = items[a]
        sy = MathSymbol(char=w, code=0, macro=tok, cls="name", role="base", font=first["ch"].font.split("+")[-1],
                        bbox=(round(first["x0"], 2), round(first["ch"].y0, 2), round(last["x1"], 2), round(last["ch"].y1, 2)),
                        mapped=True, source="font_rule")
        first["word"], first["x1"] = w, last["x1"]
        first["syms"] = [sy] + last["syms"][1:]
        first["tok"] = tok + last["stok"]
        first["text"] = w + "".join(g.c for g in sorted(last["ch"].sup + last["ch"].sub, key=lambda g: g.ox) if not g.synthetic)
        first["structs"] = {"operator_name"} | set(last["structs"] if last is not first else set())
        first["unsup"] = list(last["unsup"])
        for k in range(a + 1, b + 1):
            items[k]["tok"] = ""; items[k]["syms"] = []; items[k]["text"] = ""; items[k]["merged"] = True
    return m


def _btok(t: str) -> str:
    m = re.match(r"\\[A-Za-z]+|\\.|.", t or "")
    return m.group(0) if m else ""


def _trim(items, i, j, m):
    def bal(a, b):
        toks = [items[k]["tok"] for k in range(a, b + 1) if not items[k].get("merged")]
        stack = []
        return toks
    changed = True
    while changed and i <= j:
        changed = False
        t0, t1 = items[i]["tok"], items[j]["tok"]
        if items[i]["ch"].c in PUNCT_TRIM and items[i]["kind"] == "N" and items[i]["ch"].c != ":" :
            i += 1; changed = True; continue
        if items[j]["ch"].c in PUNCT_TRIM and items[j]["kind"] == "N":
            j -= 1; changed = True; continue
        toks = [_btok(items[k]["tok"]) for k in range(i, j + 1)]
        opens = sum(t in OPEN for t in toks); closes = sum(t in CLOSE for t in toks)
        t0, t1 = _btok(t0), _btok(t1)
        if t0 in OPEN and opens > closes:
            i += 1; changed = True; continue
        if t1 in CLOSE and closes > opens:
            j -= 1; changed = True; continue
    return i, j


def _pretty(latex: str) -> str:
    latex = re.sub(r"\s+", " ", latex).strip()
    latex = re.sub(r"([(\[]) ", r"\1", latex)
    latex = re.sub(r" ([)\],])", r"\1", latex)
    latex = re.sub(r"(\\\{) ", r"\1 ", latex)
    return latex


def _zone(items, a, b, main) -> Zone:
    sel = [it for it in items[a:b] if not it.get("merged")]
    latex = " ".join(it["tok"] for it in sel if it["tok"])
    latex = _pretty(latex)
    syms, structs, unsup = [], set(), []
    for it in sel:
        syms.extend(it["syms"]); structs |= it["structs"]; unsup.extend(it["unsup"])
    if any(s.macro == "" or (not s.mapped) for s in syms):
        latex = re.sub(r"\s+", " ", latex)
    text = "".join(it["text"] for it in sel)
    x0 = min(it["ch"].x0 for it in sel); x1 = max(it["ch"].x1 for it in sel)
    y0 = min(it["ch"].y0 for it in sel); y1 = max(it["ch"].y1 for it in sel)
    z = Zone(sel, latex, text, syms, sorted(structs), sorted(set(unsup)), (x0, y0, x1, y1), {it["ch"].blk for it in sel})
    arrows = [k for k, it in enumerate(sel) if it["ch"].cls == "arrow" and it["tok"].split("{")[0].split("[")[0] in
              {"\\to", "\\longrightarrow", "\\xrightarrow", "\\xleftarrow", "\\Rightarrow", "\\mapsto", "\\longmapsto", "\\hookrightarrow",
               "\\twoheadrightarrow", "\\rightarrow", "\\leftarrow", "\\longleftarrow", "\\Longrightarrow", "\\Longleftrightarrow"}]
    if len(arrows) >= 2:
        nodes, edges, prev = [], [], 0
        for n, k in enumerate(arrows):
            nodes.append(" ".join(it["tok"] for it in sel[prev:k] if it["tok"]))
            prev = k + 1
        nodes.append(" ".join(it["tok"] for it in sel[prev:] if it["tok"]))
        for n, k in enumerate(arrows):
            m = re.match(r"(\\x\w+arrow)(?:\[(.*?)\])?\{(.*)\}$", sel[k]["tok"])
            edges.append(dict(src=n, dst=n + 1, arrow=sel[k]["tok"] if not m else m.group(1), label_over=(m.group(3) if m else None),
                              label_under=(m.group(2) if m else None)))
        z.chain = dict(nodes=nodes, edges=edges)
    return z


def _join_text(out, main) -> str:
    s, prev_x1 = "", None
    for e in out:
        if e[0] == "text":
            it = e[1]
            if it.get("merged"):
                continue
            x0, x1, piece = it["x0"], it["x1"], it["text"]
        else:
            it, z, last = e[1], e[2], e[3]
            x0, x1, piece = it["x0"], last["x1"], "$" + z.latex + "$"
        if prev_x1 is not None and x0 - prev_x1 > 0.2 * main:
            s += " "
        s += piece
        prev_x1 = x1
    return s


# ----------------------------------------------------------------------------- document driver
def extract_math(pdf_path, document_id: str, source_blocks: list[SourceBlock], logical: list[LogicalBlock],
                 pages: list[PageInfo]) -> list[MathExpression]:
    """Reconstruct LaTeX per logical block; fills LogicalBlock.latex_text / expression_ids. Born-digital pages only."""
    doc = pymupdf.open(pdf_path)
    route = {p.page_index: p.route for p in pages}
    exprs: list[MathExpression] = []
    cache: dict[int, tuple[list[Ch], list]] = {}
    for lb in logical:
        if route.get(lb.page_index) is not PageRoute.BORN_DIGITAL:
            continue
        if lb.page_index not in cache:
            pg = doc[lb.page_index]
            work = []
            for c in page_chars(pg):
                w = _copy(c); w.blk = c.blk
                map_char(w); w.kind = _kind(w); work.append(w)
            bars = page_bars(pg)
            fl: set = set()
            ub, bars_left = _underbrace(work, bars, fl, 0)         # structures that span several parser blocks
            frs = _bars(work, bars_left, fl, 0)
            live = [c for c in work if not c.dead]
            for sy in ub + frs:
                ref = getattr(sy, "cy_bar", None)
                ybase = (ref + 0.25 * 10.9) if ref is not None else sy.oy
                near = [c for c in live if not c.synthetic and c.size >= 9 and abs(c.oy - ybase) <= 0.45 * c.size]
                left = [c for c in near if c.x1 <= sy.x0 + 2]
                pick = max(left, key=lambda c: c.x1) if left else (min(near, key=lambda c: abs(c.x0 - sy.x1)) if near else None)
                if pick is not None:
                    sy.blk = pick.blk
            cache[lb.page_index] = (live + ub + frs, bars)
        chars, bars = cache[lb.page_index]
        ords = {int(m[-4:]) for m in lb.member_block_ids}
        sub = [c for c in chars if c.blk in ords]
        a = analyze(sub, [])
        lb.latex_text = a.latex_text or None
        blk_unsup = sorted({c.unsup for c in sub if c.unsup})   # tall delimiters, brace pieces, unmapped glyphs anywhere in the block
        for z in a.zones:
            eid = f"{document_id}-X{len(exprs):05d}"
            unsup = sorted(set(z.unsupported) | set(blk_unsup))
            partial = bool(unsup) or any(not s.mapped for s in z.symbols)
            exprs.append(MathExpression(
                document_id=document_id, expression_id=eid, page_index=lb.page_index,
                bbox=tuple(round(v, 3) for v in z.bbox),
                source_block_ids=[m for m in lb.member_block_ids if int(m[-4:]) in z.block_ordinals] or lb.member_block_ids,
                logical_id=lb.logical_id, text=z.text, latex=z.latex, symbols=z.symbols, structures=z.structures,
                unsupported=unsup, status="partial" if partial else "complete", chain=z.chain,
                needs_review=partial, warnings=(["contains unmapped or unsupported glyphs; send crop to OCR/VLM"] if partial else [])))
            lb.expression_ids.append(eid)
    return exprs


def latex_for_chars(chars: list[Ch], bars: list) -> str:
    """Mixed text/math LaTeX for an arbitrary char set (used for figure node labels)."""
    return analyze([_copy(c) for c in chars], bars).latex_text


def build_inventory(expressions: list[MathExpression]) -> dict:
    """Document-level symbol inventory: counts by macro and class, plus every unmapped glyph (for VLM/OCR review)."""
    from collections import Counter
    by_macro, by_cls, unmapped, structs = Counter(), Counter(), Counter(), Counter()
    for e in expressions:
        structs.update(e.structures)
        for sy in e.symbols:
            if sy.mapped and sy.macro:
                by_macro[sy.macro] += 1; by_cls[sy.cls or "?"] += 1
            elif not sy.mapped:
                unmapped[f"{sy.font} U+{sy.code:04X}"] += 1
    return dict(n_expressions=len(expressions), n_partial=sum(e.status == "partial" for e in expressions),
                symbols_by_macro=dict(by_macro.most_common()), symbols_by_class=dict(by_cls.most_common()),
                structures=dict(structs.most_common()), unmapped_glyphs=dict(unmapped.most_common()))