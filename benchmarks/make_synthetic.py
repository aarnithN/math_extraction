"""Generate synthetic math PDFs with KNOWN ground truth (compiled from LaTeX we wrote ourselves).

Variants: single column, two column, scanned (image-only; no text layer). Ground truth lives in
<name>.gold.json. These are synthetic: they test mechanics (parsing, order, coordinates, segmentation
rules), not real-world robustness.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

CLAIM = "Since the sequence is bounded, we may choose a convergent subsequence."
FILL = ("Observe that every index either is a peak or is not a peak, and that the set of indices after a given "
        "index is again infinite, so the argument can be repeated without loss of generality. ")

ITEMS = [  # (type, label, tex body, inner_claims, formulas)
    ("section", None, "1 Preliminaries", [], []),
    ("definition", "Definition 1.", r"A sequence $(a_n)$ of real numbers is bounded if there is $M>0$ such that $|a_n|\le M$ for all $n$.", [], [r"|a_n|\le M"]),
    ("definition", "Definition 2.", r"A subsequence of $(a_n)$ is a sequence $(a_{n_k})$ where $n_1<n_2<\cdots$ are positive integers.", [], [r"n_1<n_2<\cdots"]),
    ("remark", "Remark 1.", r"Boundedness is used below only through the bound $M$.", [], []),
    ("lemma", "Lemma 1.", r"Every sequence of real numbers has a monotone subsequence.", [], []),
    ("proof", "Proof.", r"Call an index $n$ a peak if $a_n\ge a_m$ for all $m>n$. If there are infinitely many peaks, list them as $n_1<n_2<\cdots$; then $a_{n_1}\ge a_{n_2}\ge\cdots$ is a monotone subsequence. " + FILL * 6 + r"Otherwise there is a last peak $N$, and starting from $m_1=N+1$ we can choose indices $m_1<m_2<\cdots$ with"
     + r"\[ a_{m_1}<a_{m_2}<a_{m_3}<\cdots \]" + r"which is again monotone. " + FILL * 6 + "This proves the lemma.", [], [r"a_{m_1}<a_{m_2}<a_{m_3}<\cdots"]),
    ("theorem", "Theorem 2 (Bolzano--Weierstrass).", r"Every bounded sequence of real numbers has a convergent subsequence.", [], []),
    ("proof", "Proof.", r"Let $(a_n)$ be bounded. By Lemma 1 it has a monotone subsequence. " + CLAIM + r" Indeed, a bounded monotone sequence converges, and so the subsequence converges to some $L$ with $|L|\le M$.", [CLAIM], [r"|L|\le M"]),
    ("example", "Example 1.", r"The sequence $a_n=(-1)^n$ is bounded and has the convergent subsequence $a_{2k}=1$.", [], []),
    ("exercise", "Exercise 1.", r"Show that the hypothesis of boundedness cannot be dropped.", [], []),
    ("notation", "Notation 1.", r"We write $(a_{n_k})$ for a subsequence.", [], []),
]


def plain(tex: str) -> str:
    t = re.sub(r"\\\[.*?\\\]", " ", tex, flags=re.S)
    t = re.sub(r"\$.*?\$", " ", t)
    t = re.sub(r"\\[A-Za-z]+", " ", t)
    return " ".join(t.replace("--", "-").split())


def build_tex(two_col: bool) -> str:
    body = []
    for typ, label, tex, *_ in ITEMS:
        if typ == "section":
            body.append(rf"\section*{{{tex}}}")
        else:
            body.append(rf"\par\medskip\noindent\textbf{{{label}}} {tex}")
    opt = "twocolumn," if two_col else ""
    return (r"\documentclass[10pt,%sletterpaper]{article}" % opt + "\n"
            r"\usepackage[paperwidth=6in,paperheight=5in,margin=0.5in]{geometry}\usepackage{amsmath,amssymb}"
            r"\setlength{\parindent}{0pt}\pagestyle{plain}\begin{document}" + "\n".join(body) + r"\end{document}")


def compile_tex(tex: str, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.tex").write_text(tex)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "a.tex"], cwd=d, capture_output=True, text=True)
        if not (Path(d) / "a.pdf").exists():
            raise RuntimeError(r.stdout[-2000:])
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "a.tex"], cwd=d, capture_output=True)
        dest.write_bytes((Path(d) / "a.pdf").read_bytes())


def gold(name: str, has_text_layer: bool) -> dict:
    segs, n = [], 0
    for typ, label, tex, claims, formulas in ITEMS:
        if typ == "section":
            continue
        segs.append(dict(id=f"g{n}", type=typ, label=label, is_explicitly_labeled=True,
                         text_plain=label + " " + plain(tex), inner_claims=[plain(c) for c in claims], formulas=formulas))
        n += 1
    return dict(name=name, has_text_layer=has_text_layer, segments=segs,
                notes="Synthetic. Types are those a human would assign; inner_claims are unlabeled sentences inside a proof.")


def rasterize(src: Path, dest: Path, dpi: int = 110) -> None:
    out = pymupdf.open()
    for page in pymupdf.open(src):
        pix = page.get_pixmap(dpi=dpi)
        np = out.new_page(width=page.rect.width, height=page.rect.height)
        np.insert_image(np.rect, pixmap=pix)
    out.save(dest)


def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, two in (("single_column", False), ("two_column", True)):
        compile_tex(build_tex(two), outdir / f"{name}.pdf")
        (outdir / f"{name}.gold.json").write_text(json.dumps(gold(name, True), indent=1))
    rasterize(outdir / "single_column.pdf", outdir / "scanned.pdf")
    (outdir / "scanned.gold.json").write_text(json.dumps(gold("scanned", False), indent=1))
    print("wrote", sorted(p.name for p in outdir.iterdir()))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "synthetic")
