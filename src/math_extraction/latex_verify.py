"""Optional deterministic double-check: compile each reconstructed expression and extract it again.

For every complete expression the LaTeX is typeset (one page each, one pdflatex run), the compiled page is read
back with the same glyph extractor, and the result is compared with the original LaTeX. This catches invalid
macros, brace errors and structure that does not survive typesetting. It is a SELF-CONSISTENCY and VALIDITY check,
not independent ground truth: a systematic misreading that typesets back to itself would still pass.
Needs `pdflatex` (skipped cleanly if absent). Statuses: match | mismatch | compile_failed | skipped_partial | unavailable.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pymupdf

from .evaluation import norm_latex
from .math_symbols import latex_for_chars, page_bars, page_chars
from .schemas import MathExpression

HEADER = (r"\documentclass[11pt]{article}\usepackage[margin=1in]{geometry}\usepackage{amsmath,amssymb}"
          r"\pagestyle{empty}\begin{document}")


def verify_expressions(expressions: list[MathExpression], timeout: int = 180) -> dict:
    todo = [e for e in expressions if e.status == "complete" and e.latex.strip()]
    for e in expressions:
        if e not in todo:
            e.roundtrip = dict(status="skipped_partial")
    if not shutil.which("pdflatex"):
        for e in todo:
            e.roundtrip = dict(status="unavailable", reason="pdflatex not installed")
        return dict(checked=0, status="unavailable")
    body = "\n".join(rf"\noindent${e.latex}$\newpage" for e in todo)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "v.tex").write_text(HEADER + body + r"\end{document}")
        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "v.tex"], cwd=d, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        pdf = Path(d) / "v.pdf"
        if not pdf.exists():
            for e in todo:
                e.roundtrip = dict(status="compile_failed")
            return dict(checked=len(todo), status="compile_failed")
        doc = pymupdf.open(pdf)
        counts = {"match": 0, "mismatch": 0, "compile_failed": 0}
        for i, e in enumerate(todo):
            if i >= len(doc):
                e.roundtrip = dict(status="compile_failed"); counts["compile_failed"] += 1; continue
            pg = doc[i]
            back = latex_for_chars(page_chars(pg), page_bars(pg)).replace("\n", " ").strip()
            back = back[1:-1] if back.startswith("$") and back.endswith("$") and back.count("$") == 2 else back
            ok = norm_latex(back) == norm_latex(e.latex)
            e.roundtrip = dict(status="match" if ok else "mismatch", re_extracted=None if ok else back)
            if not ok:
                e.needs_review = True; e.warnings.append("round-trip mismatch: LaTeX did not typeset back to itself")
            counts["match" if ok else "mismatch"] += 1
        return dict(checked=len(todo), status="ok", **counts)