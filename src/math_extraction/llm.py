"""LLM stage: provider-agnostic client, prompt, chunking, quote verification, omission pass.

The model never supplies coordinates, page numbers or source text. It returns logical-block ids + a quote;
`verify_segments` accepts a segment only if the quote is actually found in the cited blocks.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Protocol

from .schemas import LogicalBlock, PageRegion, Provenance, Segment, SpanRef
from .segmentation import is_boilerplate, is_heading

SYSTEM_PROMPT = """You annotate mathematical documents for a knowledge-graph pipeline.
You are given ordered text blocks, each with an id. Identify every mathematically meaningful segment.

HARD RULES
- Do not verify or correct the mathematics. Do not rewrite or paraphrase the source.
- Do not invent claims, lemmas or steps that are not written. Return only source-grounded segments.
- Each segment must cite block ids from the input and give `quote`: an EXACT contiguous substring copied
  from the cited blocks' text. No ellipses, no edits.
- Labels are not required: extract unlabeled definitions, claims, and proof steps written in ordinary prose.
- Distinguish a written label from your own interpretation: set is_explicitly_labeled=true only if the label
  text (e.g. "Lemma 2.") appears in the quote; otherwise set role_is_inferred=true.
- Do not assume every sentence is a proof step; do not split one logical step into fragments.
- If unsure of the type use "unknown" or "mathematical_passage" and list alternatives.
- Blocks marked [CONTEXT ONLY] must not be the only support for a segment.
Return ONLY JSON: {"segments":[{"segment_type":str,"block_ids":[str],"quote":str,"explicit_label":str|null,
"is_explicitly_labeled":bool,"role_is_inferred":bool,"alternatives":[str],"uncertainty":str|null}]}
Allowed segment_type values: %s"""


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Real client. Requires `pip install anthropic` and ANTHROPIC_API_KEY. UNTESTED in this repo's CI:
    the test-suite uses ScriptedClient so no network or key is needed."""
    def __init__(self, model: str | None = None, max_tokens: int = 8000):
        import anthropic  # lazy: optional dependency
        self._c = anthropic.Anthropic()
        self.model = model or os.environ.get("MATH_EXTRACT_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        r = self._c.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                    messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class ScriptedClient:
    """Test double: returns canned responses in order (last one repeats)."""
    def __init__(self, responses: list[str]):
        self.responses, self.calls = responses, []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).split())


def _compact_map(s: str) -> tuple[str, list[int]]:
    chars, idx = [], []
    for i, ch in enumerate(s):
        if not ch.isspace():
            chars.append(ch); idx.append(i)
    return "".join(chars), idx


def verify_segments(raw: str, blocks: list[LogicalBlock], taxonomy: dict, document_id: str,
                    method: str = "llm", required_any_of: set[str] | None = None
                    ) -> tuple[list[Segment], list[dict]]:
    """Accept only segments whose quote is found verbatim (modulo whitespace/NFKC) in the cited blocks."""
    rejected: list[dict] = []
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        items = json.loads(text)["segments"]
        assert isinstance(items, list)
    except Exception as e:
        return [], [{"reason": f"unparseable model output: {e}", "item": raw[:200]}]
    by_id = {b.logical_id: b for b in blocks}
    out: list[Segment] = []
    for it in items:
        try:
            ids = list(it["block_ids"]); quote = norm(str(it["quote"])); stype = str(it["segment_type"])
        except Exception as e:
            rejected.append({"reason": f"missing field: {e}", "item": it}); continue
        if not quote or not ids:
            rejected.append({"reason": "empty quote or block_ids", "item": it}); continue
        missing = [i for i in ids if i not in by_id]
        if missing:
            rejected.append({"reason": f"unknown block ids {missing}", "item": it}); continue
        if required_any_of is not None and not (set(ids) & required_any_of):
            rejected.append({"reason": "cites only context blocks", "item": it}); continue
        cited = sorted((by_id[i] for i in ids), key=lambda b: b.reading_order)
        pieces = [norm(b.normalized_text) for b in cited]
        joined = " ".join(pieces)
        warns: list[str] = []
        pos = joined.find(quote)
        if pos < 0:
            cj, cmap = _compact_map(joined); cq = "".join(quote.split())
            cp = cj.find(cq)
            if cp < 0:
                rejected.append({"reason": "quote not found in cited blocks (possible fabrication)", "item": it}); continue
            pos = cmap[cp]; end = cmap[cp + len(cq) - 1] + 1
            quote_real = joined[pos:end]
            warns.append("quote matched only after ignoring whitespace")
        else:
            end = pos + len(quote); quote_real = joined[pos:end]
        spans, off = [], 0
        for b, piece in zip(cited, pieces):
            s, e = max(pos, off), min(end, off + len(piece))
            if s < e:
                spans.append(SpanRef(logical_id=b.logical_id, start=s - off, end=e - off))
            off += len(piece) + 1
        if stype not in taxonomy:
            warns.append(f"type '{stype}' not in taxonomy; stored as unknown")
            alts = list(it.get("alternatives") or []) + [stype]; stype = "unknown"
        else:
            alts = list(it.get("alternatives") or [])
        label = it.get("explicit_label")
        labeled = bool(it.get("is_explicitly_labeled"))
        if labeled and (not label or norm(str(label)).lower() not in quote_real.lower()):
            labeled = False
            warns.append("model claimed an explicit label that is not in the quoted text; downgraded to inferred")
        if it.get("uncertainty"):
            warns.append(f"model uncertainty: {it['uncertainty']}")
        by_page: dict[int, list[LogicalBlock]] = {}
        for b in cited:
            by_page.setdefault(b.page_index, []).append(b)
        regions = [PageRegion(page_index=pg, bbox=(min(x.bbox[0] for x in v), min(x.bbox[1] for x in v),
                                                   max(x.bbox[2] for x in v), max(x.bbox[3] for x in v)))
                   for pg, v in sorted(by_page.items())]
        out.append(Segment(
            document_id=document_id, segment_id="tmp", segment_type=stype,
            source_block_ids=[m for b in cited for m in b.member_block_ids], logical_block_ids=[b.logical_id for b in cited],
            source_spans=spans, exact_source_text=quote_real, page_regions=regions, document_order=0,
            explicit_label=label if labeled else None, is_explicitly_labeled=labeled,
            role_is_inferred=bool(it.get("role_is_inferred")) or not labeled, alternatives=alts,
            provenance=Provenance.LLM_PROPOSED, extraction_method=method, interpretation_status="proposed",
            needs_review=bool(warns) or stype == "unknown", warnings=warns))
    return out, rejected


def _render(blocks: list[LogicalBlock], context_only: set[str] = frozenset()) -> str:
    return "\n".join(f"[{b.logical_id}] (page {b.page_index + 1}){' [CONTEXT ONLY]' if b.logical_id in context_only else ''} "
                     f"{b.normalized_text}" for b in blocks)


def _trusted(blocks: list[LogicalBlock], skip_pages: set[int]) -> list[LogicalBlock]:
    return [b for b in blocks if b.page_index not in skip_pages and not is_boilerplate(b)]


def llm_segment(document_id: str, blocks: list[LogicalBlock], client: LLMClient, taxonomy: dict,
                skip_pages: set[int] = frozenset(), window: int = 30, overlap: int = 6
                ) -> tuple[list[Segment], list[dict]]:
    """Overlapping windows in reading order; duplicates from overlaps are removed."""
    usable = _trusted(blocks, set(skip_pages))
    system = SYSTEM_PROMPT % ", ".join(sorted(t for t, v in taxonomy.items() if v["is_math"]))
    acc: list[Segment] = []; rej: list[dict] = []
    step = max(1, window - overlap)
    for start in range(0, max(len(usable), 1), step):
        chunk = usable[start:start + window]
        if not chunk:
            break
        heading = next((b.normalized_text for b in reversed(blocks[:blocks.index(chunk[0])]) if is_heading(b)), None)
        user = f"Current section: {heading or '(unknown)'}\n\nBLOCKS:\n{_render(chunk)}"
        segs, r = verify_segments(client.complete(system, user), chunk, taxonomy, document_id)
        acc.extend(segs); rej.extend(r)
        if start + window >= len(usable):
            break
    return _dedupe(acc), rej


def _dedupe(segs: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segs:
        key = (s.segment_type, norm(s.exact_source_text))
        if any((o.segment_type, norm(o.exact_source_text)) == key for o in out):
            continue
        out.append(s)
    return out


def find_uncovered(blocks: list[LogicalBlock], segments: list[Segment], skip_pages: set[int] = frozenset()
                   ) -> list[LogicalBlock]:
    covered = {i for s in segments for i in s.logical_block_ids}
    return [b for b in _trusted(blocks, set(skip_pages)) if b.logical_id not in covered and not is_heading(b)]


def omission_pass(document_id: str, blocks: list[LogicalBlock], segments: list[Segment], client: LLMClient,
                  taxonomy: dict, skip_pages: set[int] = frozenset(), context: int = 2
                  ) -> tuple[list[Segment], list[dict]]:
    """Ask the model about blocks no segment covers. Suggestions must cite at least one uncovered block."""
    unc = find_uncovered(blocks, segments, skip_pages)
    if not unc:
        return [], []
    order = {b.logical_id: i for i, b in enumerate(blocks)}
    want: set[int] = set()
    for b in unc:
        i = order[b.logical_id]
        want.update(range(max(0, i - context), min(len(blocks), i + context + 1)))
    shown = [blocks[i] for i in sorted(want) if blocks[i].page_index not in skip_pages]
    targets = {b.logical_id for b in unc}
    system = SYSTEM_PROMPT % ", ".join(sorted(t for t, v in taxonomy.items() if v["is_math"]))
    user = ("These blocks were NOT covered by any earlier segment. Review blocks without the CONTEXT ONLY tag and "
            "return segments for any mathematical content they hold (unlabeled assertions, equations in prose, "
            "informal definitions). Return an empty list if there is none.\n\nBLOCKS:\n"
            + _render(shown, context_only={b.logical_id for b in shown} - targets))
    return verify_segments(client.complete(system, user), shown, taxonomy, document_id,
                           method="llm_omission_pass", required_any_of=targets)


def combine(rule: list[Segment], extra: list[Segment]) -> list[Segment]:
    """Keep rule-based structure; add LLM segments that are not duplicates; link inner segments to parents."""
    allseg = list(rule)
    for s in extra:
        if any(o.segment_type == s.segment_type and set(o.logical_block_ids) == set(s.logical_block_ids)
               and norm(o.exact_source_text) == norm(s.exact_source_text) for o in allseg):
            continue
        allseg.append(s)
    pos = {}
    for r in rule:
        for i in r.logical_block_ids:
            pos.setdefault(i, 0)
    def first_order(s: Segment):
        return (s.page_regions[0].page_index, s.page_regions[0].bbox[1], s.page_regions[0].bbox[0])
    allseg.sort(key=first_order)
    for n, s in enumerate(allseg):
        s.document_order = n; s.segment_id = f"{s.document_id}-S{n:04d}"
    for s in allseg:
        if s.parent_segment_id or s.extraction_method == "rule_based":
            continue
        cand = [t for t in allseg if t is not s and set(s.logical_block_ids) <= set(t.logical_block_ids)
                and len(t.exact_source_text) > len(s.exact_source_text)]
        if cand:
            s.parent_segment_id = min(cand, key=lambda t: len(t.exact_source_text)).segment_id
    return allseg
