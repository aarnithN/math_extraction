"""Minimal upload API (skeleton, local use). NOT production-ready:
  * X-User-Id is a placeholder, NOT authentication. Add real auth before exposing this anywhere.
  * Jobs run in FastAPI background tasks (in-process). Use a real queue for load/persistence.
  * No LLM calls are made unless MATH_EXTRACT_USE_LLM=1 and credentials are configured; page text would
    then leave the machine. Say so to users.
Files are stored under data/<user>/<doc_uuid>/; the client filename is never used in a path.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile

from .pipeline import run_pipeline

MAX_BYTES = int(os.environ.get("MATH_EXTRACT_MAX_BYTES", 50 * 1024 * 1024))
USER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DOC_RE = re.compile(r"^[0-9a-f]{32}$")


def create_app(data_dir: str | Path = "data") -> FastAPI:
    base = Path(data_dir)
    app = FastAPI(title="math-extraction API (skeleton)")

    def user_dir(user: str | None) -> Path:
        if not user or not USER_RE.match(user):
            raise HTTPException(400, "missing or invalid X-User-Id")
        return base / user

    def doc_dir(user: str | None, doc_id: str) -> Path:
        d = user_dir(user) / doc_id
        if not DOC_RE.match(doc_id) or not d.is_dir():
            raise HTTPException(404, "document not found")
        return d

    def status_write(d: Path, **kw) -> None:
        p = d / "status.json"
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur.update(kw)
        p.write_text(json.dumps(cur))

    def process(d: Path) -> None:
        status_write(d, state="running")
        try:
            client = None
            if os.environ.get("MATH_EXTRACT_USE_LLM") == "1":
                from .llm import AnthropicClient
                client = AnthropicClient()
            r = run_pipeline(d / "input.pdf", d / "out", client=client)
            status_write(d, state="done", document_id=r["meta"]["document_id"], n_pages=r["meta"]["n_pages"],
                         n_segments=len(r["segments"]), skipped_pages=[s["page_index"] + 1 for s in r["skipped"]],
                         results_dir=str(r["root"].relative_to(d)))
        except Exception as e:  # report, don't crash the worker
            status_write(d, state="failed", error=f"{type(e).__name__}: {e}")

    @app.post("/documents", status_code=202)
    async def upload(bg: BackgroundTasks, file: UploadFile = File(...), x_user_id: str | None = Header(None)):
        ud = user_dir(x_user_id)
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "file too large")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "not a PDF")
        doc_id = uuid.uuid4().hex
        d = ud / doc_id
        d.mkdir(parents=True)
        (d / "input.pdf").write_bytes(data)
        status_write(d, state="queued", original_filename=os.path.basename(file.filename or ""))
        bg.add_task(process, d)
        return {"document_id": doc_id, "state": "queued"}

    @app.get("/documents/{doc_id}")
    def status(doc_id: str, x_user_id: str | None = Header(None)):
        return json.loads((doc_dir(x_user_id, doc_id) / "status.json").read_text())

    def result_file(doc_id: str, user: str | None, name: str):
        d = doc_dir(user, doc_id)
        st = json.loads((d / "status.json").read_text())
        if st.get("state") != "done":
            raise HTTPException(409, f"not ready: {st.get('state')}")
        return d / st["results_dir"] / name

    @app.get("/documents/{doc_id}/math-chunks")
    def chunks(doc_id: str, x_user_id: str | None = Header(None)):
        return json.loads(result_file(doc_id, x_user_id, "math_chunks.json").read_text())

    @app.get("/documents/{doc_id}/blocks")
    def blocks(doc_id: str, x_user_id: str | None = Header(None)):
        p = result_file(doc_id, x_user_id, "blocks.jsonl")
        return [json.loads(l) for l in p.read_text().splitlines()]

    return app
