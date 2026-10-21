import json

import pytest

from math_extraction.pdf_parser import parse_pdf, write_outputs
from math_extraction.schemas import PageRoute, SourceBlock
from math_extraction.validation import validate

from conftest import REAL_PDF


def test_synthetic_text_and_routes(synthetic_pdf, tmp_path):
    meta, pages, blocks = parse_pdf(synthetic_pdf, tmp_path, dpi=72)
    assert [p.route for p in pages] == [PageRoute.BORN_DIGITAL, PageRoute.BORN_DIGITAL, PageRoute.EMPTY]
    text = " ".join(b.normalized_text for b in blocks if b.block_type == "text")
    assert "bounded sequence has a convergent subsequence" in text
    assert all(b.math_latex is None and b.math_latex_status == "unavailable" for b in blocks)


def test_ids_stable_across_runs(synthetic_pdf, tmp_path):
    _, _, b1 = parse_pdf(synthetic_pdf, tmp_path / "a", dpi=72)
    _, _, b2 = parse_pdf(synthetic_pdf, tmp_path / "b", dpi=72)
    assert [b.block_id for b in b1] == [b.block_id for b in b2]
    assert [b.bbox for b in b1] == [b.bbox for b in b2]


def test_jsonl_roundtrip_and_validation(synthetic_pdf, tmp_path):
    meta, pages, blocks = parse_pdf(synthetic_pdf, tmp_path, dpi=72)
    root = write_outputs(tmp_path, meta, pages, blocks)
    loaded = [SourceBlock.model_validate_json(l) for l in (root / "blocks.jsonl").read_text().splitlines()]
    assert loaded == blocks
    assert not [i for i in validate(synthetic_pdf, root, pages, blocks) if i["severity"] == "error"]


def test_validation_detects_dropped_block(synthetic_pdf, tmp_path):
    meta, pages, blocks = parse_pdf(synthetic_pdf, tmp_path, dpi=72)
    root = write_outputs(tmp_path, meta, pages, blocks)
    issues = validate(synthetic_pdf, root, pages, blocks[1:])
    codes = {i["code"] for i in issues}
    assert "text_coverage" in codes and "reading_order" in codes


@pytest.mark.skipif(not REAL_PDF.exists(), reason="real test PDF not present")
def test_real_pdf_routes_and_coverage(tmp_path):
    meta, pages, blocks = parse_pdf(REAL_PDF, tmp_path, dpi=72)
    routes = [p.route for p in pages]
    assert routes[:12] == [PageRoute.BORN_DIGITAL] * 12
    assert routes[12:] == [PageRoute.EMBEDDED_OCR_LAYER] * 6
    root = write_outputs(tmp_path, meta, pages, blocks)
    assert not [i for i in validate(REAL_PDF, root, pages, blocks) if i["severity"] == "error"]
    ocr = [b for b in blocks if b.page_index >= 12]
    assert ocr and all(b.needs_review and b.provenance.value == "embedded_ocr" for b in ocr)
