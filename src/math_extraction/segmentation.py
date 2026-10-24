"""Deterministic (rule-based) segmentation: explicit labels, section structure, proof continuation.

This is the offline baseline. It cannot find unlabeled claims inside paragraphs; that is the job of the
LLM stage (llm.py). Everything it produces is tagged provenance=rule_based.
"""
from __future__ import annotations

import re

from .schemas import LogicalBlock, PageInfo, PageRegion, PageRoute, Provenance, Segment

LABEL_RE = re.compile(
    r"^\s*(?P<word>Definition|Lemma|Theorem|Proposition|Probosition|Corollary|Claim|Conjecture|Axiom|Remark|"
    r"Examples|Example|Exercise|Notation|Note|Terminology|Properties|Observation|Proof|Idea|Slogan|Want to prove)\b"
    r"(?P<num>\s+\d+)?(?P<paren>\s*\([^)]*\))?\s*[.:]?")
# label word -> (segment_type, inferred_mapping_note or None)
LABEL_MAP = {
    "definition": ("definition", None), "lemma": ("lemma", None), "theorem": ("theorem", None),
    "proposition": ("proposition", None), "probosition": ("proposition", "label spelled 'Probosition' in source"),
    "corollary": ("corollary", None), "claim": ("claim", None), "conjecture": ("conjecture", None),
    "axiom": ("axiom", None), "remark": ("remark", None), "example": ("example", None),
    "examples": ("example", None), "exercise": ("exercise", None), "notation": ("notation", None),
    "note": ("remark", "label 'Note' mapped to remark"), "terminology": ("notation", "label 'Terminology' mapped to notation"),
    "properties": ("proposition", "label 'Properties' mapped to proposition"),
    "observation": ("observation", None), "proof": ("proof", None),
    "idea": ("remark", "label 'Idea' mapped to remark"), "slogan": ("remark", "label 'Slogan' mapped to remark"),
    "want to prove": ("proof", "label 'Want to prove' mapped to proof (goal-directed passage)"),
}
HEADING_RE = re.compile(r"^\d+(\.\d+)*\s+[A-Z][^.]{0,80}$")
MONTH = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
BOILER_RES = [re.compile(p) for p in (
    r"^\d{1,3}$", r"^MATH \d{3}$", r"^Lecture( Notes)?\b.{0,60}(Fall|Spring|\d{4}|" + MONTH + r").*$",
    r"^" + MONTH + r"\b.{0,12}\d{0,4}.*$")]


def is_boilerplate(b: LogicalBlock) -> bool:
    t = b.normalized_text
    return len(t) < 70 and any(r.match(t) for r in BOILER_RES)


def is_heading(b: LogicalBlock) -> bool:
    return b.has_bold and bool(HEADING_RE.match(b.normalized_text))


def _is_header_for_run(b: LogicalBlock) -> bool:
    t = b.normalized_text
    return bool(re.match(r"^MATH \d{3}$", t) or re.match(r"^Lecture\b", t))


def match_label(text: str):
    m = LABEL_RE.match(text)
    if not m:
        return None
    word = m.group("word").lower()
    stype, note = LABEL_MAP[word]
    return stype, note, m.group(0).strip(), word


def segment_rules(document_id: str, blocks: list[LogicalBlock], pages: list[PageInfo]
                  ) -> tuple[list[Segment], list[dict]]:
    pinfo = {p.page_index: p for p in pages}
    skipped: dict[int, str] = {}
    segs: list[Segment] = []
    cur: dict | None = None
    section: str | None = None
    run = 0
    run_header: str | None = None
    prev_group = None
    seen_pages: set[int] = set()

    def close():
        nonlocal cur
        if not cur:
            return
        bl: list[LogicalBlock] = cur["blocks"]
        by_page: dict[int, list[LogicalBlock]] = {}
        for b in bl:
            by_page.setdefault(b.page_index, []).append(b)
        regions = [PageRegion(page_index=pg, bbox=(min(x.bbox[0] for x in v), min(x.bbox[1] for x in v),
                                                   max(x.bbox[2] for x in v), max(x.bbox[3] for x in v)))
                   for pg, v in sorted(by_page.items())]
        warns = list(cur["warnings"])
        if len(by_page) > 1:
            warns.append(f"spans pages {sorted(by_page)}")
        if any(b.merge_reason for b in bl):
            warns.append("contains blocks merged from split fragments")
        segs.append(Segment(
            document_id=document_id, segment_id=f"{document_id}-S{len(segs):04d}", segment_type=cur["type"],
            source_block_ids=[m for b in bl for m in b.member_block_ids], logical_block_ids=[b.logical_id for b in bl],
            exact_source_text="\n".join(b.normalized_text for b in bl), page_regions=regions,
            text_latex="\n".join((b.latex_text or b.normalized_text) for b in bl),
            expression_ids=[e for b in bl for e in b.expression_ids],
            document_order=len(segs), section_title=cur["section"], source_run=cur["run"],
            explicit_label=cur["label"], is_explicitly_labeled=cur["labeled"], role_is_inferred=cur["inferred"],
            provenance=Provenance.RULE_BASED, extraction_method="rule_based",
            needs_review=any(b.needs_review for b in bl) or cur["type"] == "unknown", warnings=warns))
        cur = None

    for b in blocks:
        pi = pinfo[b.page_index]
        if pi.route is not PageRoute.BORN_DIGITAL:
            skipped[b.page_index] = (f"route={pi.route.value}: text layer not trusted; "
                                     "needs OCR/vision transcription before segmentation")
            continue
        if b.page_index not in seen_pages:
            seen_pages.add(b.page_index)
            if prev_group is not None and pi.size_group != prev_group:
                close(); run += 1; run_header = None
            prev_group = pi.size_group
        if _is_header_for_run(b):
            if b.normalized_text != run_header:
                if run_header is not None:
                    close(); run += 1
                run_header = b.normalized_text
            continue
        if b.figure_id:
            continue                       # label/node text of a figure: not prose
        if is_boilerplate(b):
            continue
        if is_heading(b):
            close(); section = b.normalized_text
            continue
        lab = match_label(b.normalized_text)
        if lab:
            close()
            stype, note, label_text, word = lab
            cur = dict(type=stype, label=label_text, labeled=word not in ("want to prove",),
                       inferred=(word == "want to prove"), blocks=[b], section=section, run=run,
                       warnings=[note] if note else [])
        elif cur:
            cur["blocks"].append(b)
        else:
            wordy = len(re.findall(r"[A-Za-z]{4,}", b.normalized_text)) >= 2
            if section and re.search(r"\bExamples?\b", section):
                stype, why = "example", f"typed from section heading '{section}'"
            elif b.has_math_font and len(b.normalized_text) < 60 and not wordy:
                stype, why = "equation", "unlabeled short math-font block"
            elif b.has_math_font:
                stype, why = "mathematical_passage", "unlabeled text with math; role guessed by rule"
            else:
                stype, why = "unknown", "unlabeled prose outside any labeled object"
            cur = dict(type=stype, label=None, labeled=False, inferred=True, blocks=[b], section=section, run=run,
                       warnings=[why])
    close()
    return segs, [{"page_index": p, "reason": r} for p, r in sorted(skipped.items())]