import pymupdf
import pytest

from math_extraction.geometry import page_to_render_rect, render_to_page_rect
from math_extraction.pdf_parser import parse_pdf


@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_roundtrip(synthetic_pdf, rot):
    d = pymupdf.open(synthetic_pdf)
    page = d[0]
    page.set_rotation(rot)
    bbox = (72.0, 90.0, 300.0, 110.0)
    px = page_to_render_rect(bbox, page, 150)
    back = render_to_page_rect(px, page, 150)
    assert back == pytest.approx(bbox, abs=1e-3)


def test_render_rect_matches_pixmap_text(synthetic_pdf):
    """Crop the render at the mapped bbox and check ink is present there, on a rotated page."""
    d = pymupdf.open(synthetic_pdf)
    page = d[1]
    assert page.rotation == 90
    words = page.get_text("words")
    x0, y0, x1, y1, *_ = words[0]
    pix = page.get_pixmap(dpi=100)
    px = page_to_render_rect((x0, y0, x1, y1), page, 100)
    assert 0 <= px[0] < px[2] <= pix.width and 0 <= px[1] < px[3] <= pix.height
    clip = pymupdf.IRect(*[int(v) for v in px])
    dark = sum(1 for yy in range(clip.y0, clip.y1) for xx in range(clip.x0, clip.x1)
               if sum(pix.pixel(xx, yy)[:3]) < 300)
    assert dark > 5, "mapped bbox contains no ink: coordinate mapping is wrong"


def test_blocks_inside_page(synthetic_pdf, tmp_path):
    _, pages, blocks = parse_pdf(synthetic_pdf, tmp_path, dpi=72)
    for b in blocks:
        pg = pages[b.page_index]
        assert 0 <= b.bbox[0] < b.bbox[2] <= pg.width + 2
        assert 0 <= b.bbox[1] < b.bbox[3] <= pg.height + 2
