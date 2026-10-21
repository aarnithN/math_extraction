"""End-to-end orchestration: parse -> annotations -> logical blocks -> columns -> segments -> export."""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from .annotations import read_annotations
from .block_merge import build_logical_blocks
from .export import add_explicit_references, export_math_chunks
from .layout import reorder_columns
from .llm import LLMClient, combine, llm_segment, omission_pass
from .pdf_parser import parse_pdf, write_outputs
from .segmentation import segment_rules
from .taxonomy import load_taxonomy
from .validation import validate


def _jsonl(path: Path, items) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in items:
            f.write(i.model_dump_json() + "\n")


def run_pipeline(pdf: Path, out: Path, dpi: int = 100, client: LLMClient | None = None,
                 taxonomy_path: str | None = None, reorder: bool = True) -> dict:
    taxonomy = load_taxonomy(taxonomy_path)
    meta, pages, blocks = parse_pdf(pdf, out, dpi=dpi)
    root = write_outputs(out, meta, pages, blocks)
    issues = validate(pdf, root, pages, blocks)
    (root / "validation.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")

    anns = read_annotations(pymupdf.open(pdf), meta["document_id"], blocks)
    _jsonl(root / "annotations.jsonl", anns)

    logical = build_logical_blocks(meta["document_id"], blocks, pages)
    twocol: list[int] = []
    if reorder:
        logical, twocol = reorder_columns(logical, pages)
    _jsonl(root / "logical_blocks.jsonl", logical)

    segs, skipped = segment_rules(meta["document_id"], logical, pages)
    rejected: list[dict] = []
    if client is not None:
        skip = {s["page_index"] for s in skipped}
        first, rej1 = llm_segment(meta["document_id"], logical, client, taxonomy, skip)
        merged = combine(segs, first)
        extra, rej2 = omission_pass(meta["document_id"], logical, merged, client, taxonomy, skip)
        segs = combine(merged, extra)
        rejected = rej1 + rej2
    add_explicit_references(segs)
    _jsonl(root / "segments.jsonl", segs)
    (root / "rejected_llm_segments.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    chunks = export_math_chunks(meta, segs, taxonomy, skipped, anns)
    (root / "math_chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    return dict(root=root, meta=meta, pages=pages, blocks=blocks, logical=logical, segments=segs, skipped=skipped,
                issues=issues, annotations=anns, two_column_pages=twocol, rejected=rejected)
