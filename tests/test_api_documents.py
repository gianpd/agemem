"""Tests for the API documents endpoint and health check."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body


# ── POST /api/v1/documents/ingest ────────────────────────────────────────────

class TestIngestEndpoint:

    def test_file_not_found_returns_404(self, client):
        resp = client.post(
            "/api/v1/documents/ingest",
            json={"path": "/nonexistent/file.md"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_unsupported_file_type_returns_422(self, client, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        resp = client.post(
            "/api/v1/documents/ingest",
            json={"path": str(txt_file)},
        )
        assert resp.status_code == 422
        assert "unsupported" in resp.json()["detail"].lower()

    @patch("api.v1.documents.ingest_document")
    def test_successful_md_ingest(self, mock_ingest, client, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test document")
        mock_ingest.return_value = "Successfully ingested markdown. doc_id: test_abc123"

        resp = client.post(
            "/api/v1/documents/ingest",
            json={"path": str(md_file)},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["doc_id"] == "test_abc123"
        mock_ingest.assert_called_once_with(
            path=str(md_file),
            doc_type="document",
            labels="edilizia",
        )

    @patch("api.v1.documents.ingest_document")
    def test_successful_pdf_ingest(self, mock_ingest, client, tmp_path):
        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        mock_ingest.return_value = "Successfully ingested PDF.\n\ndoc_id: report_xyz789"

        resp = client.post(
            "/api/v1/documents/ingest",
            json={"path": str(pdf_file), "doc_type": "contract", "labels": "legal"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        mock_ingest.assert_called_once_with(
            path=str(pdf_file),
            doc_type="contract",
            labels="legal",
        )

    @patch("api.v1.documents.ingest_document")
    def test_ingest_error_returns_500(self, mock_ingest, client, tmp_path):
        md_file = tmp_path / "broken.md"
        md_file.write_text("content")
        mock_ingest.return_value = "Error: something went wrong"

        resp = client.post(
            "/api/v1/documents/ingest",
            json={"path": str(md_file)},
        )
        assert resp.status_code == 500
        assert "error" in resp.json()["detail"].lower()

    def test_missing_path_returns_422(self, client):
        resp = client.post(
            "/api/v1/documents/ingest",
            json={},
        )
        assert resp.status_code == 422


# ── POST /api/v1/retrieval/upload ─────────────────────────────────────────────

class TestUploadEndpoint:

    def test_missing_user_id_header_returns_401(self, client):
        resp = client.post(
            "/api/v1/retrieval/upload",
            files={"file": ("test.md", b"# Test content")},
        )
        assert resp.status_code == 401
        assert "X-Agemem-User-ID" in resp.json()["detail"]

    def test_invalid_user_id_returns_400(self, client):
        resp = client.post(
            "/api/v1/retrieval/upload",
            headers={"X-Agemem-User-ID": "user@invalid"},
            files={"file": ("test.md", b"# Test content")},
        )
        assert resp.status_code == 400
        assert "Invalid" in resp.json()["detail"]

    def test_unsupported_file_type_returns_422(self, client):
        resp = client.post(
            "/api/v1/retrieval/upload",
            headers={"X-Agemem-User-ID": "test_user"},
            files={"file": ("test.txt", b"plain text")},
        )
        assert resp.status_code == 422
        assert "unsupported" in resp.json()["detail"].lower()

    @patch("api.v1.retrieval.ingest_document_to_corpus")
    def test_successful_md_upload(self, mock_ingest, client, tmp_path):
        mock_ingest.return_value = "Successfully ingested markdown. doc_id: test_abc123"

        resp = client.post(
            "/api/v1/retrieval/upload",
            headers={"X-Agemem-User-ID": "test_user"},
            files={"file": ("document.md", b"# Test document\n\nSome content here.")},
            data={"doc_type": "report", "labels": "generic"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["doc_id"] == "test_abc123"
        assert body["filename"] == "document.md"

    @patch("api.v1.retrieval.ingest_document_to_corpus")
    def test_upload_error_returns_500(self, mock_ingest, client):
        mock_ingest.return_value = "Error: ingestion failed"

        resp = client.post(
            "/api/v1/retrieval/upload",
            headers={"X-Agemem-User-ID": "test_user"},
            files={"file": ("test.md", b"content")},
        )
        assert resp.status_code == 500
        assert "Error" in resp.json()["detail"]


# ── POST /api/v1/retrieval/query ──────────────────────────────────────────────

class TestQueryEndpoint:

    def test_missing_user_id_header_returns_401(self, client):
        resp = client.post(
            "/api/v1/retrieval/query",
            json={"query": "test search"},
        )
        assert resp.status_code == 401
        assert "X-Agemem-User-ID" in resp.json()["detail"]

    def test_invalid_user_id_returns_400(self, client):
        resp = client.post(
            "/api/v1/retrieval/query",
            headers={"X-Agemem-User-ID": "user@invalid"},
            json={"query": "test search"},
        )
        assert resp.status_code == 400
        assert "Invalid" in resp.json()["detail"]

    def test_invalid_search_type_returns_422(self, client):
        resp = client.post(
            "/api/v1/retrieval/query",
            headers={"X-Agemem-User-ID": "test_user"},
            json={"query": "test", "search_type": "invalid_type"},
        )
        assert resp.status_code == 422

    def test_empty_user_corpus_returns_empty_results(self, client, tmp_path, monkeypatch):
        # Patch USER_CORPUS_BASE to use tmp_path
        monkeypatch.setattr("api.v1.retrieval.USER_CORPUS_BASE", tmp_path / "users")

        resp = client.post(
            "/api/v1/retrieval/query",
            headers={"X-Agemem-User-ID": "new_user"},
            json={"query": "test search", "max_results": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"] == []
        assert body["total"] == 0
        assert body["query"] == "test search"

    def test_missing_query_returns_422(self, client):
        resp = client.post(
            "/api/v1/retrieval/query",
            headers={"X-Agemem-User-ID": "test_user"},
            json={},
        )
        assert resp.status_code == 422

    @patch("api.v1.retrieval.search_corpus_structured")
    def test_successful_keyword_query(self, mock_search, client, tmp_path, monkeypatch):
        # Patch USER_CORPUS_BASE and create a file in the corpus
        corpus_base = tmp_path / "users"
        test_user_corpus = corpus_base / "test_user"
        test_user_corpus.mkdir(parents=True, exist_ok=True)
        # Create a dummy document so the corpus check passes
        (test_user_corpus / "dummy.md").write_text("# Dummy")

        monkeypatch.setattr("api.v1.retrieval.USER_CORPUS_BASE", corpus_base)

        mock_search.return_value = [
            {
                "doc_id": "doc_001",
                "title": "Test Document",
                "snippet": "This is a test snippet...",
                "score": 0.85,
                "source_type": "keyword",
            }
        ]

        resp = client.post(
            "/api/v1/retrieval/query",
            headers={"X-Agemem-User-ID": "test_user"},
            json={"query": "test query", "max_results": 10, "search_type": "keyword"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["results"][0]["doc_id"] == "doc_001"
        assert body["search_type"] == "keyword"

    @patch("api.v1.retrieval.search_corpus_structured")
    def test_successful_hybrid_query(self, mock_search, client, tmp_path, monkeypatch):
        # Patch USER_CORPUS_BASE and create a file in the corpus
        corpus_base = tmp_path / "users"
        test_user_corpus = corpus_base / "test_user"
        test_user_corpus.mkdir(parents=True, exist_ok=True)
        # Create a dummy document so the corpus check passes
        (test_user_corpus / "dummy.md").write_text("# Dummy")

        monkeypatch.setattr("api.v1.retrieval.USER_CORPUS_BASE", corpus_base)
        mock_search.return_value = [
            {
                "doc_id": "doc_001",
                "title": "Test Document",
                "snippet": "This is a test snippet...",
                "score": 0.75,
                "source_type": "keyword",
            },
            {
                "doc_id": "doc_002",
                "title": "Another Document",
                "snippet": "More content here...",
                "score": 0.65,
                "source_type": "semantic",
            },
        ]

        resp = client.post(
            "/api/v1/retrieval/query",
            headers={"X-Agemem-User-ID": "test_user"},
            json={"query": "test query", "max_results": 5, "search_type": "hybrid"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["search_type"] == "hybrid"
