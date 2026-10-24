import pymupdf
import pytest

from math_extraction.block_merge import build_logical_blocks
from math_extraction.layout import reorder_columns
from math_extraction.pdf_parser import parse_pdf


def test_real_split_equation_is_merged(real_run):
    hit = [l for l in real_run["logical"] if "yh2h−1" in l.normalized_text and "h′ ∈yH" in l.normalized_text]
    assert hit and len(hit[0].member_block_ids) == 2 and hit[0].merge_reason


def test_no_merging_on_ocr_pages(real_run):
    assert all(len(l.member_block_ids) == 1 for l in real_run["logical"] if l.page_index >= 12)


def test_merge_preserves_all_text(real_run):
    src = sorted("".join("".join((b.raw_text or "").split()) for b in real_run["blocks"]))
    log = sorted("".join("".join(l.raw_text.split()) for l in real_run["logical"]))
    assert src == log   # same characters: nothing dropped or duplicated (order may differ when fragments are merged)


def test_every_source_block_in_exactly_one_logical_block(real_run):
    ids = [m for l in real_run["logical"] for m in l.member_block_ids]
    assert sorted(ids) == sorted(b.block_id for b in real_run["blocks"])


@pytest.fixture()
def scrambled_two_column(tmp_path):
    d = pymupdf.open(); p = d.new_page(width=612, height=792)
    left = [f"LEFT{i} " + "alpha beta gamma delta " * 3 for i in range(4)]
    right = [f"RIGHT{i} " + "alpha beta gamma delta " * 3 for i in range(4)]
    for i, t in enumerate(right):          # right column written FIRST -> wrong native order
        p.insert_textbox(pymupdf.Rect(330, 80 + i * 150, 560, 80 + i * 150 + 100), t, fontsize=11)
    for i, t in enumerate(left):
        p.insert_textbox(pymupdf.Rect(50, 80 + i * 150, 280, 80 + i * 150 + 100), t, fontsize=11)
    f = tmp_path / "two.pdf"; d.save(f); return f


def test_column_reorder_fixes_scrambled_order(scrambled_two_column, tmp_path):
    meta, pages, blocks = parse_pdf(scrambled_two_column, tmp_path, dpi=50)
    logical = build_logical_blocks(meta["document_id"], blocks, pages)
    native = [l.normalized_text.split()[0] for l in logical]
    assert native[0].startswith("RIGHT")        # confirms the input really is mis-ordered
    out, twocol = reorder_columns(logical, pages)
    assert twocol == [0]
    assert [l.normalized_text.split()[0] for l in out] == [f"LEFT{i}" for i in range(4)] + [f"RIGHT{i}" for i in range(4)]
    assert [l.reading_order for l in out] == list(range(8))


def test_single_column_untouched(syn_run):
    assert syn_run["two_column_pages"] == []


def test_real_run_never_reorders_ocr_layer_pages(real_run):
    assert not [p for p in real_run["two_column_pages"] if p >= 12]