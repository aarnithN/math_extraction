"""Configurable segment taxonomy. `is_math=False` types are excluded from the math-chunk export."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TAXONOMY: dict[str, dict] = {
    "theorem": {"is_math": True, "desc": "Stated theorem"},
    "lemma": {"is_math": True, "desc": "Stated lemma"},
    "proposition": {"is_math": True, "desc": "Stated proposition or property"},
    "corollary": {"is_math": True, "desc": "Stated corollary"},
    "definition": {"is_math": True, "desc": "Definition, formal or informal"},
    "axiom": {"is_math": True, "desc": "Axiom or standing assumption"},
    "assumption": {"is_math": True, "desc": "Assumption"},
    "conjecture": {"is_math": True, "desc": "Conjecture or hypothesis"},
    "claim": {"is_math": True, "desc": "Claim or subclaim, labeled or not"},
    "proof": {"is_math": True, "desc": "A proof or proof segment"},
    "proof_step": {"is_math": True, "desc": "A distinct step inside a proof (model annotation)"},
    "derivation": {"is_math": True, "desc": "Chain of computations"},
    "equation": {"is_math": True, "desc": "Standalone equation or inequality"},
    "example": {"is_math": True, "desc": "Example"},
    "counterexample": {"is_math": True, "desc": "Counterexample / non-example"},
    "remark": {"is_math": True, "desc": "Remark, note, observation, idea"},
    "observation": {"is_math": True, "desc": "Observation"},
    "exercise": {"is_math": True, "desc": "Exercise left to the reader"},
    "notation": {"is_math": True, "desc": "Notation, convention or terminology"},
    "mathematical_passage": {"is_math": True, "desc": "Mathematical text that fits no other type"},
    "unknown": {"is_math": True, "desc": "Unclear; kept, flagged for review"},
    "section_heading": {"is_math": False, "desc": "Heading (context only)"},
    "boilerplate": {"is_math": False, "desc": "Dates, page numbers, running headers"},
}


def load_taxonomy(path: str | Path | None = None) -> dict[str, dict]:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return DEFAULT_TAXONOMY
