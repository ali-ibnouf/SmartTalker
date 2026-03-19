"""Tests for SmartTalker Knowledge Base engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# KnowledgeBaseEngine Tests
# =============================================================================


class TestKBEngine:
    """Tests for KnowledgeBaseEngine."""

    def test_init(self, config):
        """Engine initializes without loading."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        engine = KnowledgeBaseEngine(config)
        assert not engine.is_loaded

    @pytest.mark.asyncio
    async def test_load_creates_collection(self, config, tmp_path):
        """load() initializes ChromaDB and creates collection."""
        pytest.importorskip("chromadb")
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        config.kb_storage_dir = tmp_path / "kb"
        engine = KnowledgeBaseEngine(config)

        mock_chroma = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            engine.load()

            assert engine.is_loaded
            mock_client.get_or_create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_unload(self, config):
        """unload() resets state."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        engine = KnowledgeBaseEngine(config)
        engine._loaded = True
        await engine.unload()
        assert not engine.is_loaded

    @pytest.mark.asyncio
    async def test_ingest_text(self, config, tmp_path):
        """ingest_text stores text chunks in ChromaDB."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        config.kb_storage_dir = tmp_path / "kb"
        engine = KnowledgeBaseEngine(config)
        engine._loaded = True

        mock_collection = MagicMock()
        mock_collection.add = MagicMock()
        engine._collection = mock_collection

        # Mock embedding
        engine._get_embeddings_batch = AsyncMock(return_value=[[0.1] * 384])

        doc = await engine.ingest_text(
            text="This is test content for knowledge base.",
            source_name="test_source",
        )

        assert doc.doc_id is not None
        assert doc.filename == "test_source"
        assert doc.chunk_count >= 1
        mock_collection.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_document_missing_file(self, config, tmp_path):
        """ingest_document with missing file raises error."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        from src.utils.exceptions import KnowledgeBaseError
        config.kb_storage_dir = tmp_path / "kb"
        engine = KnowledgeBaseEngine(config)
        engine._loaded = True

        with pytest.raises(KnowledgeBaseError, match="not found"):
            await engine.ingest_document("/nonexistent/file.txt", doc_type="txt")

    def test_chunk_text_basic(self, config):
        """_chunk_text splits text into overlapping chunks."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        config.kb_chunk_size = 100
        config.kb_chunk_overlap = 20
        engine = KnowledgeBaseEngine(config)

        # Use small chunk size for testing
        text = "word " * 200  # ~1000 chars
        chunks = engine._chunk_text(text)
        assert len(chunks) > 1
        # Each chunk should be <= chunk_size (in chars)
        for chunk in chunks:
            assert len(chunk) <= 120  # Some tolerance for stripping

    def test_chunk_text_short(self, config):
        """Short text returns single chunk."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        config.kb_chunk_size = 500
        config.kb_chunk_overlap = 50
        engine = KnowledgeBaseEngine(config)

        chunks = engine._chunk_text("Short text.")
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_chunk_text_empty(self, config):
        """Empty text returns empty list."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        config.kb_chunk_size = 500
        config.kb_chunk_overlap = 50
        engine = KnowledgeBaseEngine(config)

        chunks = engine._chunk_text("")
        assert chunks == []

    def test_chunk_overlap(self, config):
        """Chunks have overlapping content."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        config.kb_chunk_size = 100
        config.kb_chunk_overlap = 30
        engine = KnowledgeBaseEngine(config)

        text = " ".join([f"word{i}" for i in range(100)])
        chunks = engine._chunk_text(text)

        if len(chunks) >= 2:
            # Check there's some overlap between consecutive chunks
            words_chunk1 = set(chunks[0].split())
            words_chunk2 = set(chunks[1].split())
            overlap = words_chunk1 & words_chunk2
            assert len(overlap) > 0

    @pytest.mark.asyncio
    async def test_search_empty_collection(self, config, tmp_path):
        """Search on empty collection returns empty results."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        config.kb_storage_dir = tmp_path / "kb"
        engine = KnowledgeBaseEngine(config)
        engine._loaded = True

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        engine._collection = mock_collection

        result = await engine.search("test query")
        assert result.chunks == []
        assert result.top_similarity == 0.0

    @pytest.mark.asyncio
    async def test_query_no_results(self, config, tmp_path):
        """Query with no relevant results returns low confidence."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        config.kb_storage_dir = tmp_path / "kb"
        engine = KnowledgeBaseEngine(config)
        engine._loaded = True

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        engine._collection = mock_collection

        result = await engine.query("test question")
        assert result.confidence == 0.0
        assert not result.has_answer

    def test_list_documents_empty(self, config, tmp_path):
        """list_documents returns empty when no docs ingested."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        config.kb_storage_dir = tmp_path / "kb"
        engine = KnowledgeBaseEngine(config)
        engine._loaded = True
        engine._docs_index = {}

        docs = engine.list_documents()
        assert docs == []

    def test_delete_document_not_found(self, config, tmp_path):
        """delete_document returns False for unknown doc."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        config.kb_storage_dir = tmp_path / "kb"
        engine = KnowledgeBaseEngine(config)
        engine._loaded = True
        engine._docs_index = {}
        engine._collection = MagicMock()

        result = engine.delete_document("nonexistent-id")
        assert result is False

    def test_parse_txt(self, config, tmp_path):
        """TXT parser reads file content."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        engine = KnowledgeBaseEngine(config)

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello World\nLine two.", encoding="utf-8")

        text = engine._parse_txt(str(txt_file))
        assert "Hello World" in text
        assert "Line two" in text

    def test_parse_csv(self, config, tmp_path):
        """CSV parser reads row content."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        engine = KnowledgeBaseEngine(config)

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")

        text = engine._parse_csv(str(csv_file))
        assert "Alice" in text
        assert "Bob" in text

    def test_parse_json(self, config, tmp_path):
        """JSON parser reads and stringifies content."""
        from src.pipeline.knowledge_base import KnowledgeBaseEngine
        engine = KnowledgeBaseEngine(config)

        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps({"key": "value", "nested": {"a": 1}}), encoding="utf-8")

        text = engine._parse_json(str(json_file))
        assert "key" in text
        assert "value" in text
