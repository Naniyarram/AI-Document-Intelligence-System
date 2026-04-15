
# tests/test_pipeline.py

# Unit tests for the document intelligence pipeline.
# Run with: pytest tests/ -v


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import tempfile


#Test: SemanticChunker 
class TestSemanticChunker:

    def setup_method(self):
        from src.chunking.semantic_chunker import SemanticChunker
        self.chunker = SemanticChunker()

    def test_chunk_basic_text(self):
        """Test that basic text is chunked into expected pieces."""
        from src.ingestion.document_loader import DocumentPage
        page = DocumentPage(
            source_file="test.txt",
            page_number=0,
            text="This is a test sentence. " * 100,  # long enough to create multiple chunks
            content_type="text"
        )
        chunks = self.chunker.chunk([page])
        assert len(chunks) > 0
        assert all(chunk.text for chunk in chunks)
        assert all(chunk.source_file == "test.txt" for chunk in chunks)

    def test_table_not_split(self):
        """Tables should be kept as a single chunk, not split."""
        from src.ingestion.document_loader import DocumentPage
        table_text = "Name | Age | City\n" + ("John | 30 | NYC\n" * 50)
        page = DocumentPage(
            source_file="test.xlsx",
            page_number=0,
            text=table_text,
            content_type="table"
        )
        chunks = self.chunker.chunk([page])
        # Table should be ONE chunk
        assert len(chunks) == 1
        assert chunks[0].content_type == "table"

    def test_empty_pages_skipped(self):
        """Empty pages should not create chunks."""
        from src.ingestion.document_loader import DocumentPage
        pages = [
            DocumentPage(source_file="test.txt", page_number=0, text="", content_type="text"),
            DocumentPage(source_file="test.txt", page_number=1, text="   ", content_type="text"),
        ]
        chunks = self.chunker.chunk(pages)
        assert len(chunks) == 0

    def test_chunk_has_required_fields(self):
        """Each chunk must have all required metadata fields."""
        from src.ingestion.document_loader import DocumentPage
        from src.chunking.semantic_chunker import TextChunk
        page = DocumentPage(
            source_file="contract.pdf",
            page_number=2,
            section_title="Payment Terms",
            text="Payment shall be due within 30 days of invoice. Late fees apply.",
            content_type="text"
        )
        chunks = self.chunker.chunk([page])
        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.source_file == "contract.pdf"
        assert chunk.page_number == 2
        assert chunk.section_title == "Payment Terms"


#  Test: EntityExtractor 
class TestEntityExtractor:

    def setup_method(self):
        from src.extraction.entity_extractor import EntityExtractor
        self.extractor = EntityExtractor()

    def test_extract_emails(self):
        text = "Contact us at support@example.com or sales@company.org"
        entities = self.extractor.extract(text)
        assert "email_addresses" in entities
        assert len(entities["email_addresses"]) == 2

    def test_extract_invoice_numbers(self):
        text = "Invoice INV-2024-0341 and PO#9876 are overdue."
        entities = self.extractor.extract(text)
        assert "invoice_numbers" in entities
        assert len(entities["invoice_numbers"]) >= 1

    def test_extract_money(self):
        text = "The total is $84,200.00 and the deposit is $5,000"
        entities = self.extractor.extract(text)
        assert "money_amounts" in entities
        assert len(entities["money_amounts"]) >= 1

    def test_empty_text_returns_empty(self):
        entities = self.extractor.extract("")
        assert entities == {}


# Test: DocumentLoader 
class TestDocumentLoader:

    def setup_method(self):
        from src.ingestion.document_loader import DocumentLoader
        self.loader = DocumentLoader()

    def test_load_txt(self):
        """Test loading a simple .txt file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Section one content.\n\nSection two content.\n\nSection three.")
            tmp_path = f.name

        try:
            pages = self.loader.load(tmp_path)
            assert len(pages) > 0
            assert all(p.text for p in pages)
            assert all(p.source_file.endswith(".txt") for p in pages)
        finally:
            os.unlink(tmp_path)

    def test_unsupported_format_raises(self):
        """Unsupported file types should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            self.loader.load("document.xyz")

    def test_looks_like_table(self):
        """Test the table detection heuristic."""
        table_text = "Name | Age | City\nJohn | 30 | NYC\nJane | 25 | LA"
        assert self.loader._looks_like_table(table_text) is True

        plain_text = "This is a normal paragraph with no tables at all."
        assert self.loader._looks_like_table(plain_text) is False


# Test: Config
class TestConfig:

    def test_config_has_required_fields(self):
        from config import Config
        assert hasattr(Config, "EMBEDDING_MODEL")
        assert hasattr(Config, "RERANKER_MODEL")
        assert hasattr(Config, "CHUNK_SIZE")
        assert hasattr(Config, "CHUNK_OVERLAP")
        assert Config.CHUNK_SIZE > 0
        assert Config.CHUNK_OVERLAP > 0
        assert Config.CHUNK_OVERLAP < Config.CHUNK_SIZE

    def test_prefers_hf_api_key_alias(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "")
        monkeypatch.setenv("HF_API_KEY", "hf_alias_token")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-router-token")

        import config
        from importlib import reload

        reload(config)
        assert config.Config.get_api_key() == "hf_alias_token"
        assert config.Config.get_backend_name() == "HuggingFace"

    def test_uses_backend_safe_default_models(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "")
        monkeypatch.setenv("HF_API_KEY", "hf_alias_token")
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        monkeypatch.setenv("LLM_MODEL", "")
        monkeypatch.setenv("VLM_MODEL", "")

        import config
        from importlib import reload

        reload(config)
        assert config.Config.get_llm_model() == config.Config.DEFAULT_HF_LLM
        assert config.Config.get_vlm_model() == config.Config.DEFAULT_HF_VLM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
