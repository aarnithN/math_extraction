import json
from pathlib import Path

from conftest import SYN


def test_synthetic_types_match_gold_in_order(syn_run):
    gold = json.loads((SYN / "single_column.gold.json").read_text())
    assert [s.segment_type for s in syn_run["segments"]] == [g["type"] for g in gold["segments"]]


def test_proof_crossing_page_break_is_one_segment(syn_run):
    proof = next(s for s in syn_run["segments"] if s.segment_type == "proof")
    assert len({r.page_index for r in proof.page_regions}) >= 2
    assert any("spans pages" in w for w in proof.warnings)


def test_page_numbers_not_in_segments(syn_run):
    assert all(not s.exact_source_text.strip().isdigit() for s in syn_run["segments"])


def test_rule_segmenter_cannot_isolate_unlabeled_claim(syn_run):
    """Documents the known gap the LLM stage must fill: no segment is just the unlabeled sentence."""
    claim = "Since the sequence is bounded"
    assert not [s for s in syn_run["segments"] if s.exact_source_text.startswith(claim)]


def test_scanned_pages_skipped_not_silently_dropped(tmp_path, syn_scanned):
    from math_extraction.pipeline import run_pipeline
    r = run_pipeline(syn_scanned, tmp_path)
    assert r["segments"] == []
    assert [s["page_index"] for s in r["skipped"]] == [0, 1, 2]
    assert all(p.route.value == "scanned_no_text" for p in r["pages"])


def test_real_pages_and_runs(real_run):
    assert [s["page_index"] for s in real_run["skipped"]] == list(range(12, 18))
    assert {s.source_run for s in real_run["segments"]} == {0, 1, 2, 3}


def test_real_labels_and_inferred_roles(real_run):
    S = real_run["segments"]
    prob = next(s for s in S if s.explicit_label and s.explicit_label.startswith("Probosition"))
    assert prob.segment_type == "proposition" and any("Probosition" in w for w in prob.warnings)
    want = next(s for s in S if s.explicit_label and s.explicit_label.startswith("Want to prove"))
    assert want.segment_type == "proof" and want.role_is_inferred and not want.is_explicitly_labeled


def test_real_explicit_references_resolve_within_source(real_run):
    S = {s.segment_id: s for s in real_run["segments"]}
    rm2 = next(s for s in S.values() if s.explicit_label == "Remark 2." and s.source_run == 0)
    refs = {r.target_text: r for r in rm2.references}
    assert refs["Definition 1"].resolved and S[refs["Definition 1"].target_segment_id].source_run == 0
    assert refs["Proposition 1"].resolved
    assert all(r.kind == "explicit_reference" for s in S.values() for r in s.references)


def test_real_no_ocr_page_text_in_math_chunks(real_run):
    chunks = json.loads((Path(real_run["root"]) / "math_chunks.json").read_text())
    assert all(max(c["pages"]) <= 12 for c in chunks["chunks"])
    assert [s["page_index"] for s in chunks["skipped_pages"]] == list(range(12, 18))
