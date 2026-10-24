"""Explicit-reference extraction and the filtered 'math chunks' export for the graph builder."""
from __future__ import annotations

import re

from .schemas import PdfAnnotation, Reference, Segment

REF_RE = re.compile(r"\b(Definition|Lemma|Theorem|Proposition|Corollary|Remark|Example|Exercise|Notation)\s+(\d+)\b")
NAMED_RE = re.compile(r"\b((?:First|Second|Third|1st|2nd|3rd)\s+Isomorphism\s+Theorem|[A-Z][a-z]+['’]s\s+[Tt]heorem)")
LABEL_KEY_RE = re.compile(r"^\s*(\w+)\s+(\d+)")


def _canon_name(s: str) -> list[str]:
    s = s.lower().replace("1st", "first").replace("2nd", "second").replace("3rd", "third")
    return [w for w in re.findall(r"[a-z]+", s) if w not in ("theorem", "s")]


def add_explicit_references(segments: list[Segment]) -> None:
    """Attach explicit_reference entries (never inferred dependencies; those need the LLM and stay separate)."""
    by_key: dict[tuple[int | None, str], list[Segment]] = {}
    for s in segments:
        m = LABEL_KEY_RE.match(s.explicit_label or "")
        if m:
            by_key.setdefault((s.source_run, f"{m.group(1).lower()} {m.group(2)}"), []).append(s)
    for s in segments:
        own = (s.explicit_label or "").lower()
        text = s.exact_source_text
        # skip the label itself at the start of the segment
        body = text[len(s.explicit_label):] if s.explicit_label and text.startswith(s.explicit_label) else text
        for m in REF_RE.finditer(body):
            key = f"{m.group(1).lower()} {m.group(2)}"
            if key in own[:len(key) + 1]:
                continue
            cands = [t for t in by_key.get((s.source_run, key), []) if t.segment_id != s.segment_id]
            ev = body[max(0, m.start() - 30): m.end() + 30].strip()
            s.references.append(Reference(kind="explicit_reference", target_text=m.group(0),
                                          target_segment_id=cands[0].segment_id if len(cands) == 1 else None,
                                          resolved=len(cands) == 1, candidates=[c.segment_id for c in cands],
                                          evidence_quote=ev))
        for m in NAMED_RE.finditer(body):
            words = _canon_name(m.group(1))
            cands = [t for t in segments if t.explicit_label and t.segment_id != s.segment_id
                     and all(w in " ".join(_canon_name(t.explicit_label)) for w in words) and words]
            ev = body[max(0, m.start() - 30): m.end() + 30].strip()
            same = [c for c in cands if c.source_run == s.source_run]
            cands = same or cands   # prefer the same pasted/merged source document
            s.references.append(Reference(kind="explicit_reference", target_text=m.group(1),
                                          target_segment_id=cands[0].segment_id if len(cands) == 1 else None,
                                          resolved=len(cands) == 1, candidates=[c.segment_id for c in cands],
                                          evidence_quote=ev))


def attach_figures(segments: list[Segment], figures: list) -> None:
    """A figure belongs to the segment whose region on that page vertically contains it, else the nearest preceding one."""
    for f in figures:
        cy = (f.bbox[1] + f.bbox[3]) / 2
        cands = [(s, r) for s in segments for r in s.page_regions if r.page_index == f.page_index]
        inside = [s for s, r in cands if r.bbox[1] - 2 <= cy <= r.bbox[3] + 2]
        if inside:
            inside[0].figure_ids.append(f.figure_id); continue
        before = [(r.bbox[3], s) for s, r in cands if r.bbox[3] <= cy]
        if before:
            max(before, key=lambda t: t[0])[1].figure_ids.append(f.figure_id)
        else:
            prev = [s for s in segments if s.page_regions and s.page_regions[-1].page_index < f.page_index]
            if prev:
                prev[-1].figure_ids.append(f.figure_id)


def export_math_chunks(meta: dict, segments: list[Segment], taxonomy: dict, skipped_pages: list[dict],
                       annotations: list[PdfAnnotation], figures: list | None = None) -> dict:
    chunks = []
    for s in segments:
        if not taxonomy.get(s.segment_type, {"is_math": True})["is_math"]:
            continue
        chunks.append(dict(
            id=s.segment_id, type=s.segment_type, label=s.explicit_label,
            is_explicitly_labeled=s.is_explicitly_labeled, role_is_inferred=s.role_is_inferred,
            text=s.exact_source_text, text_latex=s.text_latex, expression_ids=s.expression_ids, figure_ids=s.figure_ids,
            section=s.section_title, source_run=s.source_run,
            pages=sorted({r.page_index + 1 for r in s.page_regions}),
            page_regions=[r.model_dump() for r in s.page_regions], source_block_ids=s.source_block_ids,
            parent=s.parent_segment_id, references=[r.model_dump() for r in s.references],
            provenance=s.provenance.value, extraction_method=s.extraction_method,
            needs_review=s.needs_review, warnings=s.warnings))
    return dict(document_id=meta["document_id"], source_file=meta["source_file"],
                coordinate_system=meta["coordinate_system"], page_numbering="pages are 1-based here; regions use page_index (0-based)",
                chunks=chunks, skipped_pages=skipped_pages,
                figures=[f.model_dump(mode="json") for f in (figures or [])],
                annotations=[a.model_dump(mode="json") for a in annotations],
                notes=["Chunk text is exact source text; no paraphrase or correction.",
                       "type/label reflect rule- or model-based interpretation; check provenance and role_is_inferred.",
                       "references are explicit mentions only; no inferred dependencies are included.",
                       "figures are located and structured deterministically from vector graphics; check structure_status/needs_review."])