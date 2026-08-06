# ruff: noqa: E402
# ─────────────────────────────────────────────────────────
# tests/conftest.py — Shared pytest fixtures
# ─────────────────────────────────────────────────────────

import sys
import os
import tempfile
from unittest.mock import MagicMock

# ── Mock sentence_transformers to bypass PyTorch WinError 1114 DLL failures ──
class MockSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, texts, *args, **kwargs):
        import numpy as np
        if isinstance(texts, str):
            return np.zeros(384)
        return np.zeros((len(texts), 384))

class MockCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass
    def predict(self, pairs, *args, **kwargs):
        import numpy as np
        return np.zeros(len(pairs))

# Inject mock module
import types
mock_st_module = types.ModuleType("sentence_transformers")
mock_st_module.SentenceTransformer = MockSentenceTransformer
mock_st_module.CrossEncoder = MockCrossEncoder
sys.modules["sentence_transformers"] = mock_st_module

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import patch
from src.chunking.semantic_chunker import TextChunk


@pytest.fixture
def sample_text():
    """A realistic ~500 word document about a consulting agreement."""
    return (
        "CONSULTING AGREEMENT\n\n"
        "This Consulting Agreement (the 'Agreement') is entered into as of January 15, 2024, "
        "by and between Acme Corporation, a Delaware corporation with offices at 742 Evergreen "
        "Terrace, Springfield, IL 62704 ('Client'), and Jane Smith Consulting LLC, a California "
        "limited liability company ('Consultant').\n\n"
        "1. SCOPE OF SERVICES\n\n"
        "The Consultant agrees to provide strategic advisory services related to digital "
        "transformation, including but not limited to: (a) assessment of current technology "
        "infrastructure; (b) development of a three-year technology roadmap; (c) vendor "
        "evaluation and selection support; and (d) change management recommendations. The "
        "Consultant shall deliver a comprehensive written report within 90 days of the "
        "Effective Date.\n\n"
        "2. COMPENSATION\n\n"
        "Client shall pay Consultant a fixed fee of $84,200.00 for all services described "
        "herein. Payment shall be made in three installments: (i) $25,000 upon execution of "
        "this Agreement; (ii) $30,000 upon delivery of the interim report at Day 45; and "
        "(iii) $29,200 upon delivery of the final report. Late payments shall accrue interest "
        "at a rate of 1.5% per month.\n\n"
        "3. TERM AND TERMINATION\n\n"
        "This Agreement shall commence on the Effective Date and continue for a period of "
        "six (6) months unless earlier terminated. Either party may terminate this Agreement "
        "upon thirty (30) days' prior written notice to the other party. In the event of "
        "termination, the Consultant shall be compensated for all services performed up to "
        "the date of termination, calculated on a pro-rata basis.\n\n"
        "4. CONFIDENTIALITY\n\n"
        "Each party agrees to maintain the confidentiality of all proprietary information "
        "disclosed by the other party during the term of this Agreement. This obligation "
        "shall survive termination of this Agreement for a period of two (2) years. "
        "Confidential information includes, but is not limited to, trade secrets, customer "
        "lists, financial data, and business strategies.\n\n"
        "5. GOVERNING LAW\n\n"
        "This Agreement shall be governed by and construed in accordance with the laws of "
        "the State of Delaware, without regard to its conflict of laws provisions. Any "
        "disputes arising under this Agreement shall be resolved through binding arbitration "
        "in Wilmington, Delaware.\n\n"
        "IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first "
        "written above.\n\n"
        "Signed: John Doe, CEO, Acme Corporation\n"
        "Signed: Jane Smith, Managing Partner, Jane Smith Consulting LLC\n"
        "Contact: support@acme-corp.com | billing@janesmithconsulting.com\n"
        "Invoice Reference: INV-2024-0341\n"
    )


@pytest.fixture
def sample_chunks(sample_text):
    """A list of TextChunks built from sample_text, simulating what the chunker produces."""
    # Split sample_text into rough paragraphs, make a chunk for each
    sections = [s.strip() for s in sample_text.split("\n\n") if s.strip()]
    chunks = []
    for i, section in enumerate(sections):
        chunks.append(TextChunk(
            text=section,
            source_file="test_contract.pdf",
            page_number=0,
            section_title=f"Section {i}",
            chunk_index=i,
            content_type="text",
        ))
    return chunks


@pytest.fixture
def mock_pipeline():
    """
    A DocumentPipeline with heavy external deps mocked out.
    The embedder, LLM client, and sentence-transformers model are all faked
    so tests don't need GPU/network access.
    """
    with patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("chromadb.PersistentClient") as mock_chroma, \
         patch("openai.OpenAI") as mock_openai:

        # Fake embedding model — returns 384-dim zeros
        fake_model = MagicMock()
        fake_model.encode.return_value = __import__("numpy").zeros((1, 384))
        mock_st.return_value = fake_model

        # Fake chromadb client
        fake_client = MagicMock()
        fake_client.list_collections.return_value = []
        mock_chroma.return_value = fake_client

        # Fake OpenAI client
        fake_openai_client = MagicMock()
        mock_openai.return_value = fake_openai_client

        from pipeline import DocumentPipeline
        pipeline = DocumentPipeline()

        # Attach mocks in case tests need to assert on them
        pipeline._mock_st = mock_st
        pipeline._mock_chroma = mock_chroma
        pipeline._mock_openai = mock_openai

        yield pipeline


@pytest.fixture
def tmp_txt_file():
    """Create a temp .txt file, yield its path, clean up after."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("Section one content.\n\nSection two content.\n\nSection three.")
        path = f.name

    yield path

    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def tmp_csv_file():
    """Create a temp .csv file, yield its path, clean up after."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("name,amount,date\n")
        f.write("Widget A,1500.00,2024-01-15\n")
        f.write("Widget B,2300.50,2024-02-20\n")
        f.write("Widget C,875.25,2024-03-10\n")
        path = f.name

    yield path

    try:
        os.unlink(path)
    except OSError:
        pass
