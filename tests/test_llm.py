import json

import pytest

from math_extraction.evaluation import evaluate
from math_extraction.llm import ScriptedClient, combine, find_uncovered, llm_segment, omission_pass, verify_segments
from math_extraction.taxonomy import DEFAULT_TAXONOMY as TAX

from conftest import SYN

CLAIM = "Since the sequence is bounded, we may choose a convergent subsequence."


@pytest.fixture()
def blocks(syn_run):
    return syn_run["logical"]


@pytest.fixture()
def proof_block(blocks):
    return next(b for b in blocks if CLAIM in b.normalized_text)


def resp(*items): return json.dumps({"segments": list(items)})


def item(ids, quote, stype="proof_step", **kw):
    return dict(segment_type=stype, block_ids=ids, quote=quote, explicit_label=None, is_explicitly_labeled=False,
                role_is_inferred=True, alternatives=[], uncertainty=None, **kw)


def test_exact_quote_accepted_with_correct_spans(blocks, proof_block):
    segs, rej = verify_segments(resp(item([proof_block.logical_id], CLAIM)), blocks, TAX, "d")
    assert len(segs) == 1 and not rej
    sp = segs[0].source_spans[0]
    assert proof_block.normalized_text[sp.start:sp.end] == CLAIM
    assert segs[0].provenance.value == "llm_proposed" and segs[0].role_is_inferred


def test_fabricated_quote_rejected(blocks, proof_block):
    segs, rej = verify_segments(resp(item([proof_block.logical_id], "Hence every sequence converges.")), blocks, TAX, "d")
    assert not segs and "fabrication" in rej[0]["reason"]


def test_one_word_changed_is_rejected(blocks, proof_block):
    segs, rej = verify_segments(resp(item([proof_block.logical_id], CLAIM.replace("bounded", "unbounded"))), blocks, TAX, "d")
    assert not segs and rej


def test_whitespace_difference_accepted_with_warning(blocks, proof_block):
    q = CLAIM.replace("we may", "we  may\n")
    segs, _ = verify_segments(resp(item([proof_block.logical_id], q)), blocks, TAX, "d")
    assert len(segs) == 1 and segs[0].exact_source_text == CLAIM


def test_unknown_block_id_rejected(blocks):
    segs, rej = verify_segments(resp(item(["nope"], "x y")), blocks, TAX, "d")
    assert not segs and "unknown block ids" in rej[0]["reason"]


def test_bad_json_rejected_and_fenced_json_ok(blocks, proof_block):
    assert verify_segments("not json", blocks, TAX, "d")[0] == []
    fenced = "```json\n" + resp(item([proof_block.logical_id], CLAIM)) + "\n```"
    assert len(verify_segments(fenced, blocks, TAX, "d")[0]) == 1


def test_model_cannot_set_coordinates_or_text(blocks, proof_block):
    evil = item([proof_block.logical_id], CLAIM, page_regions=[{"page_index": 99, "bbox": [0, 0, 1, 1]}],
                exact_source_text="something invented", bbox=[1, 2, 3, 4])
    s = verify_segments(resp(evil), blocks, TAX, "d")[0][0]
    assert s.exact_source_text == CLAIM and s.page_regions[0].page_index == proof_block.page_index
    assert tuple(s.page_regions[0].bbox) == tuple(proof_block.bbox)


def test_false_explicit_label_downgraded(blocks, proof_block):
    it = item([proof_block.logical_id], CLAIM, "lemma"); it.update(explicit_label="Lemma 7.", is_explicitly_labeled=True)
    s = verify_segments(resp(it), blocks, TAX, "d")[0][0]
    assert not s.is_explicitly_labeled and s.explicit_label is None and any("downgraded" in w for w in s.warnings)


def test_type_outside_taxonomy_kept_as_unknown(blocks, proof_block):
    s = verify_segments(resp(item([proof_block.logical_id], CLAIM, "banana")), blocks, TAX, "d")[0][0]
    assert s.segment_type == "unknown" and "banana" in s.alternatives and s.needs_review


def test_llm_claim_gets_parent_and_improves_claim_recall(syn_run, blocks, proof_block):
    gold = json.loads((SYN / "single_column.gold.json").read_text())
    dump = lambda segs: [s.model_dump(mode="json") for s in segs]
    L = [b.model_dump(mode="json") for b in blocks]
    before = evaluate(gold, L, dump(syn_run["segments"]))["unlabeled_claim_recall"]
    client = ScriptedClient([resp(item([proof_block.logical_id], CLAIM),
                                  item([proof_block.logical_id], "This sentence is not in the document."))])
    first, rej = llm_segment("d", blocks, client, TAX)
    assert len(first) == 1 and len(rej) == 1            # fabricated one rejected, real one kept
    merged = combine(syn_run["segments"], first)
    claim = next(s for s in merged if s.extraction_method == "llm")
    parent = next(s for s in merged if s.segment_id == claim.parent_segment_id)
    assert parent.segment_type == "proof"
    after = evaluate(gold, L, dump([s for s in merged]))["unlabeled_claim_recall"]
    assert before == 0.0 and after == 1.0


def test_omission_pass_requires_uncovered_block(syn_run, blocks):
    segs = syn_run["segments"]
    dropped = next(s for s in segs if s.segment_type == "remark")
    kept = [s for s in segs if s is not dropped]
    unc = find_uncovered(blocks, kept)
    assert [b.logical_id for b in unc] == dropped.logical_block_ids
    target = unc[0]
    other = next(b for b in blocks if b.logical_id not in dropped.logical_block_ids and b.normalized_text.startswith("Lemma 1"))
    q_ok = target.normalized_text
    good = ScriptedClient([resp(item([target.logical_id], q_ok, "remark"))])
    s, _ = omission_pass("d", blocks, kept, good, TAX)
    assert len(s) == 1 and s[0].extraction_method == "llm_omission_pass"
    ctx_only = ScriptedClient([resp(item([other.logical_id], other.normalized_text, "lemma"))])
    s2, r2 = omission_pass("d", blocks, kept, ctx_only, TAX)
    assert not s2 and "context" in r2[0]["reason"]


def test_pipeline_with_llm_client_end_to_end(tmp_path, syn_single, proof_block):
    from math_extraction.pipeline import run_pipeline
    c = ScriptedClient([resp(item(["ignored"], "x"))])      # references a block id that does not exist
    r = run_pipeline(syn_single, tmp_path, client=c)
    assert r["rejected"] and all(s.extraction_method != "llm" for s in r["segments"])   # bad output never leaks in
