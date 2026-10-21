"""Scoring against gold annotations.

Gold format (synthetic or hand-made):
  {"segments":[{"id","type","text_plain" | "logical_block_ids", "inner_claims":[...], "formulas":[latex,...]}]}
Gold text is aligned to parsed logical blocks with a token matcher that does NOT depend on block order,
so a reading-order bug cannot hide itself. Report each metric separately; do not average them.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

TOK = re.compile(r"[a-z]{2,}")
alnum = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def _toks(s: str) -> list[str]:
    return TOK.findall(s.lower())


def align_gold_to_blocks(gold: dict, blocks: list[dict]) -> tuple[dict[str, set[str]], dict[str, float]]:
    """Returns ({gold_id: set(logical_id)}, {logical_id: gold stream position})."""
    stream: list[str] = []; owner: list[str] = []
    for g in gold["segments"]:
        t = _toks(g.get("text_plain", ""))
        stream += t; owner += [g["id"]] * len(t)
    sets: dict[str, set[str]] = {g["id"]: set() for g in gold["segments"]}
    pos: dict[str, float] = {}
    assigned: list[tuple[int, str]] = []
    for i, b in enumerate(blocks):
        bt = _toks(b["normalized_text"])
        if len(bt) < 2 or not stream:
            continue
        sm = SequenceMatcher(None, stream, bt, autojunk=False)
        m = sm.find_longest_match(0, len(stream), 0, len(bt))
        matched = sum(x.size for x in sm.get_matching_blocks())
        if m.size >= 2 and matched >= 0.5 * len(bt):
            gid = owner[m.a]
            sets[gid].add(b["logical_id"]); pos[b["logical_id"]] = m.a + m.size / 2
            assigned.append((i, gid))
    # fill equation-only blocks sandwiched between blocks of the same gold segment
    amap = dict(assigned)
    for i, b in enumerate(blocks):
        if b["logical_id"] in pos or re.fullmatch(r"\d{1,3}", b["normalized_text"]):
            continue   # already assigned, or a bare page number (boilerplate, never part of an object)
        prev = next((amap[j] for j in range(i - 1, -1, -1) if j in amap), None)
        nxt = next((amap[j] for j in range(i + 1, len(blocks)) if j in amap), None)
        if prev is not None and prev == nxt:
            sets[prev].add(b["logical_id"])
    return sets, pos


def _jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(gold: dict, blocks: list[dict], segments: list[dict]) -> dict:
    gold_sets, pos = align_gold_to_blocks(gold, blocks)
    bmap = {b["logical_id"]: b for b in blocks}
    top = [s for s in segments if not s.get("parent_segment_id")]
    pred_sets = [(s, set(s["logical_block_ids"])) for s in top]
    res: dict = dict(n_gold=len(gold["segments"]), n_pred_top_level=len(top), n_blocks=len(blocks))

    matched_pairs, exact, type_ok, loc_ok, ious = 0, 0, 0, 0, []
    used = set()
    for g in gold["segments"]:
        gs = gold_sets[g["id"]]
        if not gs:
            continue
        best = max(((s, _jacc(gs, ps)) for s, ps in pred_sets), key=lambda x: x[1], default=(None, 0))
        if best[0] is not None and best[1] >= 0.5:
            s = best[0]; used.add(s["segment_id"]); matched_pairs += 1
            exact += best[1] == 1.0
            type_ok += s["segment_type"] == g["type"]
            gpages = {bmap[i]["page_index"] for i in gs}
            loc_ok += {r["page_index"] for r in s["page_regions"]} == gpages
            for pg in gpages:
                gb = _union([bmap[i]["bbox"] for i in gs if bmap[i]["page_index"] == pg])
                pr = [r["bbox"] for r in s["page_regions"] if r["page_index"] == pg]
                ious.append(_iou(gb, _union(pr)) if pr else 0.0)
    n_aligned = sum(1 for g in gold["segments"] if gold_sets[g["id"]])
    res.update(
        gold_segments_aligned_to_blocks=n_aligned,
        segment_recall=matched_pairs / len(gold["segments"]) if gold["segments"] else None,
        segment_precision=len(used) / len(top) if top else None,
        type_accuracy_on_matched=type_ok / matched_pairs if matched_pairs else None,
        exact_boundary_rate_on_matched=exact / matched_pairs if matched_pairs else None,
        page_set_accuracy_on_matched=loc_ok / matched_pairs if matched_pairs else None,
        mean_region_iou_on_matched=sum(ious) / len(ious) if ious else None,
    )
    # unlabeled claims: a predicted segment must contain the claim and be about its size (isolated, not the whole proof)
    claims = [c for g in gold["segments"] for c in g.get("inner_claims", [])]
    hit = 0
    for c in claims:
        ca = alnum(c)
        hit += any(ca in alnum(s["exact_source_text"]) and len(alnum(s["exact_source_text"])) <= 1.2 * len(ca) + 10
                   for s in segments)
    res["unlabeled_claim_recall"] = hit / len(claims) if claims else None
    res["n_unlabeled_claims"] = len(claims)
    # reading order: among gold-aligned blocks in the given order, share of adjacent pairs that go forward
    seq = [pos[b["logical_id"]] for b in blocks if b["logical_id"] in pos]
    res["reading_order_accuracy"] = (sum(1 for a, b in zip(seq, seq[1:]) if b >= a) / (len(seq) - 1)) if len(seq) > 1 else None
    # formula fidelity: only measurable if latex was extracted
    latex = [b.get("math_latex") for b in blocks if b.get("math_latex")]
    gf = [f for g in gold["segments"] for f in g.get("formulas", [])]
    if latex and gf:
        sc = [max(SequenceMatcher(None, f, l).ratio() for l in latex) for f in gf]
        res["formula_fidelity"] = sum(sc) / len(sc)
    else:
        res["formula_fidelity"] = None
        res["formula_fidelity_note"] = "no LaTeX extracted (math_latex is empty in this pipeline version)"
    res["type_confusions"] = []
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="math-eval")
    ap.add_argument("doc_dir", type=Path, help="pipeline output folder (out/doc_xxx)")
    ap.add_argument("gold", type=Path)
    a = ap.parse_args(argv)
    blocks = [json.loads(l) for l in (a.doc_dir / "logical_blocks.jsonl").read_text().splitlines()]
    segs = [json.loads(l) for l in (a.doc_dir / "segments.jsonl").read_text().splitlines()]
    r = evaluate(json.loads(a.gold.read_text()), blocks, segs)
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
