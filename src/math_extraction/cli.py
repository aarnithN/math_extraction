from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .overlay import write_overlays
from .pipeline import run_pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="math-extract")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--overlay", action="store_true", help="write page images with block boxes")
    ap.add_argument("--llm", action="store_true",
                    help="add the LLM segmentation + omission pass (needs `pip install anthropic` and ANTHROPIC_API_KEY; "
                         "sends block TEXT to the provider)")
    ap.add_argument("--no-column-reorder", action="store_true")
    a = ap.parse_args(argv)

    client = None
    if a.llm:
        from .llm import AnthropicClient
        client = AnthropicClient()
    r = run_pipeline(a.pdf, a.out, dpi=a.dpi, client=client, reorder=not a.no_column_reorder)
    if a.overlay:
        write_overlays(a.pdf, r["root"], r["pages"], r["blocks"])
    segs = r["segments"]
    print(f"document_id={r['meta']['document_id']}  pages={r['meta']['n_pages']}  source_blocks={len(r['blocks'])}"
          f"  logical_blocks={len(r['logical'])}")
    print("routes:", dict(Counter(p.route.value for p in r["pages"])))
    print("segments:", len(segs), dict(Counter(s.segment_type for s in segs)))
    print("skipped pages (need OCR/vision):", [s["page_index"] + 1 for s in r["skipped"]])
    print("annotations:", len(r["annotations"]), " two-column pages:", r["two_column_pages"])
    print("validation:", dict(Counter((i['severity'], i['code']) for i in r["issues"])))
    print("main output:", r["root"] / "math_chunks.json")
    return 1 if any(i["severity"] == "error" for i in r["issues"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
