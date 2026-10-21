import os
from pathlib import Path

import pymupdf
import pytest

REAL_PDF = Path(os.environ.get("MATH_TEST_PDF", "/mnt/user-data/uploads/lecture_notes_redacted__1_.pdf"))


@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory) -> Path:
    """3 pages: normal, rotated 90, blank. Known text at known positions."""
    p = tmp_path_factory.mktemp("pdf") / "syn.pdf"
    d = pymupdf.open()
    a = d.new_page(width=612, height=792)
    a.insert_text((72, 100), "Theorem 1. Every bounded sequence has a convergent subsequence.", fontsize=11)
    a.insert_text((72, 300), "Proof. Since the sequence is bounded, pick a subsequence.", fontsize=11)
    b = d.new_page(width=612, height=792)
    b.insert_text((100, 200), "Rotated page text", fontsize=12)
    b.set_rotation(90)
    d.new_page(width=612, height=792)
    d.save(p)
    return p


SYN = Path(__file__).resolve().parents[1] / "benchmarks" / "synthetic"


@pytest.fixture(scope="session")
def syn_single(): return SYN / "single_column.pdf"


@pytest.fixture(scope="session")
def syn_scanned(): return SYN / "scanned.pdf"


@pytest.fixture(scope="session")
def real_run(tmp_path_factory):
    if not REAL_PDF.exists():
        pytest.skip("real test PDF not present")
    from math_extraction.pipeline import run_pipeline
    return run_pipeline(REAL_PDF, tmp_path_factory.mktemp("real"))


@pytest.fixture(scope="session")
def syn_run(tmp_path_factory, syn_single):
    from math_extraction.pipeline import run_pipeline
    return run_pipeline(syn_single, tmp_path_factory.mktemp("syn"))
