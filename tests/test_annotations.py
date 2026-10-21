import pymupdf

from math_extraction.annotations import read_annotations
from math_extraction.pdf_parser import parse_pdf


def test_reads_highlight_note_freetext_ink(tmp_path):
    d = pymupdf.open(); p = d.new_page()
    p.insert_text((72, 100), "Every bounded sequence has a convergent subsequence.", fontsize=12)
    hit = p.search_for("convergent subsequence")[0]
    h = p.add_highlight_annot(hit); h.set_info(content="key step", title="alice"); h.update()
    n = p.add_text_annot((300, 300), "why is this true?"); n.set_info(title="bob"); n.update()
    p.add_freetext_annot(pymupdf.Rect(72, 400, 300, 440), "margin remark", fontsize=10)
    p.add_ink_annot([[(100, 500), (120, 520), (150, 510)]])
    f = tmp_path / "ann.pdf"; d.save(f)
    meta, pages, blocks = parse_pdf(f, tmp_path / "o", dpi=50)
    anns = read_annotations(pymupdf.open(f), meta["document_id"], blocks)
    by = {a.subtype: a for a in anns}
    assert {"Highlight", "Text", "FreeText", "Ink"} <= set(by)
    assert by["Highlight"].highlighted_text == "convergent subsequence"
    assert by["Highlight"].content == "key step" and by["Highlight"].author == "alice"
    assert by["Highlight"].overlapping_block_ids          # linked to the text block it sits on
    assert by["Text"].content == "why is this true?"
    assert by["Ink"].n_ink_points == 3
    assert all(a.provenance.value == "user_annotation" for a in anns)
    # annotations never alter source blocks
    assert not any("key step" in (b.raw_text or "") for b in blocks)


def test_no_annotations_gives_empty(real_run):
    assert real_run["annotations"] == []
