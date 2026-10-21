import json

from math_extraction.evaluation import evaluate

from conftest import SYN


def _dump(r):
    return [b.model_dump(mode="json") for b in r["logical"]], [s.model_dump(mode="json") for s in r["segments"]]


def test_baseline_scores_on_synthetic(syn_run):
    gold = json.loads((SYN / "single_column.gold.json").read_text())
    e = evaluate(gold, *_dump(syn_run))
    assert e["segment_recall"] == 1.0 and e["segment_precision"] == 1.0
    assert e["type_accuracy_on_matched"] == 1.0 and e["exact_boundary_rate_on_matched"] == 1.0
    assert e["reading_order_accuracy"] == 1.0 and e["page_set_accuracy_on_matched"] == 1.0
    assert e["unlabeled_claim_recall"] == 0.0 and e["formula_fidelity"] is None


def test_scorer_detects_dropped_and_mistyped_segments(syn_run):
    gold = json.loads((SYN / "single_column.gold.json").read_text())
    L, S = _dump(syn_run)
    e = evaluate(gold, L, S[1:])
    assert e["segment_recall"] == 0.9
    S2 = [dict(s) for s in S]; S2[0]["segment_type"] = "lemma"
    assert evaluate(gold, L, S2)["type_accuracy_on_matched"] == 0.9


def test_scorer_detects_bad_reading_order(syn_run):
    gold = json.loads((SYN / "single_column.gold.json").read_text())
    L, S = _dump(syn_run)
    assert evaluate(gold, list(reversed(L)), S)["reading_order_accuracy"] < 0.3
