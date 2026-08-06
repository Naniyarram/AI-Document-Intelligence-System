# tests/test_api.py
import sys
import os
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app, get_pipeline

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}


def test_list_documents_empty():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert "documents" in response.json()


def test_upload_unsupported_format():
    files = {"file": ("test.xyz", b"hello world", "text/plain")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_query_no_documents():
    response = client.post(
        "/api/v1/query",
        json={"question": "hello?", "document_name": "non_existent.pdf", "mode": "qa"}
    )
    assert response.status_code == 404
