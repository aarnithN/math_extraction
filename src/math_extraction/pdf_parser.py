"""PyMuPDF extraction: raw dumps + normalized SourceBlocks with coordinates."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf

from . import __version__
from .document_classifier import classify_page, is_math_font
from .geometry import page_to_render_rect
from .schemas import PageInfo, PageRoute, Provenance, SourceBlock

PARSER_NAME = "pymupdf"
PUA_RE = re.compile("[\ue000-\uf8ff\ufffd]")


def doc_id_for(pdf_path: Path) -> tuple[str, str]:
    h = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return f"doc_{h[:12]}", h


def _strip_binary(obj):
    """Raw dict minus image bytes so it is JSON-serializable without loss of structure."""
    if isinstance(obj, dict):
        return {k: (f"<{len(v)} bytes>" if k == "image" and isinstance(v, (bytes, bytearray)) else _strip_binary(v))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_binary(x) for x in obj]
    return obj


def parse_pdf(pdf_path: Path, out_dir: Path, dpi: int = 150,
              crop_flagged: bool = True) -> tuple[dict, list[PageInfo], list[SourceBlock]]:
    pdf_path = Path(pdf_path)
    doc_id, sha = doc_id_for(pdf_path)
    root = out_dir / doc_id
    for sub in ("raw", "renders", "crops"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    version = pymupdf.VersionBind
    size_groups: dict[tuple[int, int], int] = {}
    pages: list[PageInfo] = []
    blocks: list[SourceBlock] = []
    order = 0

    for page in doc:
        pn = page.number
        td = page.get_text("dict")
        (root / "raw" / f"page_{pn:04d}.json").write_text(json.dumps(_strip_binary(td)), encoding="utf-8")

        route, reasons, stats = classify_page(page, td)
        key = (round(page.rect.width), round(page.rect.height))
        sg = size_groups.setdefault(key, len(size_groups))
        label = page.get_label() or None

        render_rel = f"renders/page_{pn:04d}.png"
        page.get_pixmap(dpi=dpi).save(root / render_rel)
        pages.append(PageInfo(document_id=doc_id, page_index=pn, printed_page_number=label,
                              width=page.rect.width, height=page.rect.height, rotation=page.rotation,
                              route=route, route_reasons=reasons, size_group=sg,
                              render_path=render_rel, render_dpi=dpi, **stats))

        for ordinal, b in enumerate(td["blocks"]):
            bbox = tuple(round(v, 3) for v in b["bbox"])
            bid = f"{doc_id}-p{pn:04d}-b{ordinal:04d}"
            warnings: list[str] = []
            common = dict(document_id=doc_id, block_id=bid, page_index=pn, printed_page_number=label,
                          reading_order=order, bbox=bbox, parser_name=PARSER_NAME, parser_version=version)
            order += 1
            if b["type"] == 1:
                blocks.append(SourceBlock(block_type="image", provenance=Provenance.EXTRACTED_TEXT,
                                          ocr_status="not_applicable", needs_review=True,
                                          warnings=["image block; content not transcribed in Phase 1"], **common))
                continue

            spans = [s for l in b["lines"] for s in l["spans"]]
            raw = "\n".join("".join(s["text"] for s in l["spans"]) for l in b["lines"])
            fonts = sorted({s["font"] for s in spans})
            has_math = any(is_math_font(s["font"]) for s in spans)
            if route is PageRoute.EMBEDDED_OCR_LAYER:
                prov, ocr, review = Provenance.EMBEDDED_OCR, "embedded_unverified", True
                warnings.append("text comes from an embedded OCR layer on a handwritten/image page; "
                                "treat as unreliable and compare to the crop")
            else:
                prov, ocr, review = Provenance.EXTRACTED_TEXT, "not_applicable", False
                if has_math:
                    warnings.append("contains math-font glyphs; Unicode text is lossy for "
                                    "super/subscripts, fractions and stacked layout; math_latex unavailable")
            if PUA_RE.search(raw):
                warnings.append("contains private-use or replacement characters")
                review = True
            flags = [s["flags"] for s in spans]
            blk = SourceBlock(
                block_type="text", raw_text=raw, normalized_text=" ".join(raw.split()),
                fonts=fonts, max_font_size=max((s["size"] for s in spans), default=None),
                baseline=round(sorted(s["origin"][1] for s in spans)[len(spans) // 2], 3) if spans else None,
                has_math_font=has_math, has_bold=any(f & 16 for f in flags) or any("Bold" in s["font"] for s in spans),
                has_italic=any(f & 2 for f in flags) or any("Italic" in s["font"] for s in spans),
                provenance=prov, ocr_status=ocr, needs_review=review, warnings=warnings, **common)
            if crop_flagged and (blk.needs_review or has_math):
                rel = f"crops/{bid}.png"
                px = page_to_render_rect(bbox, page, dpi)
                clip = pymupdf.Rect(bbox)  # clip is in unrotated page space
                pad = pymupdf.Rect(clip.x0 - 4, clip.y0 - 4, clip.x1 + 4, clip.y1 + 4) & page.rect
                page.get_pixmap(dpi=dpi, clip=pad).save(root / rel)
                blk.source_image_ref, blk.crop_dpi = rel, dpi
            blocks.append(blk)

    meta = dict(document_id=doc_id, source_file=pdf_path.name, sha256=sha, n_pages=len(doc),
                pymupdf_version=version, pipeline_version=__version__, render_dpi=dpi,
                metadata=doc.metadata, coordinate_system="PDF points, unrotated page space, origin top-left")
    return meta, pages, blocks


def write_outputs(out_dir: Path, meta: dict, pages: list[PageInfo], blocks: list[SourceBlock]) -> Path:
    root = out_dir / meta["document_id"]
    (root / "document.json").write_text(
        json.dumps({**meta, "pages": [p.model_dump(mode="json") for p in pages]}, indent=2), encoding="utf-8")
    with (root / "blocks.jsonl").open("w", encoding="utf-8") as f:
        for b in blocks:
            f.write(b.model_dump_json() + "\n")
    return root