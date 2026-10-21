import pytest
from fastapi.testclient import TestClient

from math_extraction.app import create_app


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(tmp_path / "data"))


def up(client, path, user="alice", name="notes.pdf"):
    return client.post("/documents", files={"file": (name, open(path, "rb"), "application/pdf")},
                       headers={"X-User-Id": user} if user else {})


def test_upload_process_and_fetch(client, syn_single):
    r = up(client, syn_single); assert r.status_code == 202
    doc = r.json()["document_id"]
    st = client.get(f"/documents/{doc}", headers={"X-User-Id": "alice"}).json()
    assert st["state"] == "done" and st["n_segments"] == 10
    ch = client.get(f"/documents/{doc}/math-chunks", headers={"X-User-Id": "alice"}).json()
    assert ch["chunks"][0]["label"] == "Definition 1."
    assert client.get(f"/documents/{doc}/blocks", headers={"X-User-Id": "alice"}).json()


def test_other_user_cannot_read(client, syn_single):
    doc = up(client, syn_single).json()["document_id"]
    assert client.get(f"/documents/{doc}", headers={"X-User-Id": "mallory"}).status_code == 404


def test_rejects_non_pdf_missing_user_bad_ids_and_traversal(client, syn_single, tmp_path):
    bad = tmp_path / "x.pdf"; bad.write_bytes(b"hello")
    assert up(client, bad).status_code == 415
    assert up(client, syn_single, user=None).status_code == 400
    assert up(client, syn_single, user="../etc").status_code == 400
    assert client.get("/documents/..%2F..", headers={"X-User-Id": "alice"}).status_code == 404
    r = up(client, syn_single, name="../../evil.pdf")       # hostile filename must not influence paths
    assert r.status_code == 202
    assert not (tmp_path / "evil.pdf").exists()


def test_corrupt_pdf_reports_failed_state(client, tmp_path):
    bad = tmp_path / "c.pdf"; bad.write_bytes(b"%PDF-1.4\ngarbage")
    doc = up(client, bad).json()["document_id"]
    st = client.get(f"/documents/{doc}", headers={"X-User-Id": "alice"}).json()
    assert st["state"] == "failed" and st["error"]
    assert client.get(f"/documents/{doc}/math-chunks", headers={"X-User-Id": "alice"}).status_code == 409
