# math-extraction

Extracts source-grounded mathematical segments (definitions, theorems, proofs, examples, ...) from PDFs
for a downstream knowledge-graph builder. Every segment points back to parser blocks, page regions and exact text.

## Install and run (zsh: quote the extras)

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e '.[dev]'
    math-extract path/to/notes.pdf --out out --overlay      # rule-based, fully local
    math-extract path/to/notes.pdf --out out --llm          # + LLM pass (see Privacy)
    math-eval out/doc_xxxx benchmarks/synthetic/single_column.gold.json
    pytest

Main output: `out/doc_<id>/math_chunks.json` (the filtered math-only export). Other files in that folder:
`blocks.jsonl` (every parser block), `logical_blocks.jsonl` (split fragments merged), `segments.jsonl`,
`rejected_llm_segments.json`, `annotations.jsonl`, `validation.json`, `document.json`, `raw/`, `renders/`, `crops/`.

## Pipeline

1. **Inspect and route each page** (PyMuPDF): born-digital / embedded-OCR-layer / scanned-no-text / empty.
2. **Extract blocks** with coordinates, fonts and raw dumps (Layer A: `SourceBlock`, never modified).
3. **Read user annotations** (highlights, comments, free text, ink) as separate objects.
4. **Merge split fragments** (e.g. an equation broken into pieces) into `LogicalBlock`s; reorder two-column pages.
5. **Rule-based segmentation**: explicit labels, section structure, proofs continuing across pages.
6. **Optional LLM pass**: overlapping windows -> JSON with block ids + exact quotes -> `verify_segments`
   rejects anything not found in the cited blocks; a second pass reviews uncovered blocks the same way.
7. **Explicit references** ("by Definition 1") resolved within their source document.
8. **Export** math-only chunks, with skipped pages and annotations listed, not hidden.

LLM never supplies text, page numbers or coordinates; those come from the parser layer.

## Status (honest)

| Area | State |
|---|---|
| Parsing, coordinates, routing, validation | Working; tested on 1 real PDF + synthetic PDFs |
| Split-equation merging | Working on the real file (all merges inspected); lossy text, no LaTeX |
| Rule segmentation + references | Working on the real file; rule-based only sees labels |
| LLM stage | **Verification, prompt, chunking, omission pass tested with scripted responses only.** `AnthropicClient` is untested; model quality is unmeasured |
| Handwritten / scanned pages | **Not extracted.** Detected and listed in `skipped_pages`; no OCR route yet |
| LaTeX recovery (`math_latex`) | Not implemented |
| Annotations | Tested on a PDF generated here; real files from other editors untested |
| Column ordering | Unit-tested on a scrambled synthetic page; real two-column papers untested |
| App (`app.py`) | Local skeleton; X-User-Id is NOT authentication; in-process jobs |
| Evaluation | Scripts work; benchmark is synthetic. No hand-annotated real gold set yet |

## Privacy

The default pipeline is local. `--llm` (or `MATH_EXTRACT_USE_LLM=1` for the API) sends block TEXT to the
configured provider. Do not use it on private notes without the owner's consent.

## Known limitations

- Unlabeled claims inside paragraphs are found only by the LLM stage.
- Nested objects (a Remark inside a Proof) are flat in the rule baseline.
- Types come from label words (`Idea`, `Note`, `Properties` are mapped; see `segmentation.py`); mapped labels carry warnings.
- Source documents pasted together are separated by header/page-size heuristics (`source_run`).
- Redacted regions leave no trace in the output.
