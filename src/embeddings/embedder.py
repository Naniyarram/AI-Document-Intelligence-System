# ─────────────────────────────────────────────────────────
# src/embeddings/embedder.py
#
# STAGE 4 — Embedding & Vector Store
#
# ROOT CAUSE FIX for the onnxruntime DLL error:
#
#   ChromaDB ships with a built-in embedding function that
#   uses ONNX Runtime to run a MiniLM model internally.
#   On some Windows setups this DLL fails to load.
#
#   THE FIX: We bypass ChromaDB's built-in embedder entirely.
#   Instead we:
#     1. Compute embeddings ourselves using sentence-transformers
#     2. Pass the pre-computed vectors directly to ChromaDB
#
#   ChromaDB never needs to load onnxruntime at all.
#   This is actually the BETTER approach anyway — we control
#   exactly which model is used and can swap it easily.
#
# Model: all-MiniLM-L6-v2 (sentence-transformers)
#   - Runs fully locally, no API needed
#   - 384-dimensional vectors
#   - ~80 MB download, cached after first use
#
# Storage: ChromaDB (PersistentClient)
#   - Saves to disk → survives app restarts
#   - No server, no Docker, just a folder
# ─────────────────────────────────────────────────────────

import os
import re
import hashlib
import numpy as np
from typing import List
from loguru import logger
from config import Config
from src.chunking.semantic_chunker import TextChunk


class LocalHashEmbeddingModel:
    """
    Fully local fallback embedder.

    It creates deterministic fixed-size vectors from token hashes so the
    project can still index and retrieve documents when model downloads
    from Hugging Face are blocked.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(
        self,
        texts,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
    ):
        vectors = [self._encode_text(text) for text in texts]
        matrix = np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=np.float32)
        return matrix if convert_to_numpy else matrix.tolist()

    def _encode_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"\b\w+\b", (text or "").lower())

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


class Embedder:
    """
    Embeds TextChunks and stores them in ChromaDB.

    Key design: we pass pre-computed embeddings to ChromaDB.
    This completely bypasses ChromaDB's ONNX dependency.

    Usage:
        embedder = Embedder()
        embedder.index(chunks, collection_name="contract_pdf")
        results = embedder.dense_search("termination clause", "contract_pdf")
    """

    def __init__(self):
        # Load our own embedding model first
        self.model = self._load_embedding_model()
        # Then init ChromaDB — safe because we won't use its embedder
        self.chroma_client = self._init_chroma()
        logger.info("Embedder ready (custom embeddings, bypassing ChromaDB ONNX)")

    def _load_embedding_model(self):
        """
        Load sentence-transformers embedding model.
        Downloads ~80 MB on first run, then cached locally.
        No API key or internet connection needed after download.
        """
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            try:
                return SentenceTransformer(Config.EMBEDDING_MODEL, local_files_only=True)
            except TypeError:
                return SentenceTransformer(Config.EMBEDDING_MODEL)
        except Exception as e:
            logger.warning(
                f"Could not load embedding model '{Config.EMBEDDING_MODEL}': {e}\n"
                "Using local hash-based embeddings instead."
            )
            return LocalHashEmbeddingModel(dim=384)

    def _init_chroma(self):
        """
        Initialize ChromaDB with persistent on-disk storage.

        IMPORTANT: We set the default embedding function to None here
        (via embedding_function=None when creating collections).
        This tells ChromaDB "I will supply my own vectors — don't load ONNX."
        """
        import chromadb
        from chromadb.config import Settings

        os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)

        # anonymized_telemetry=False stops ChromaDB from phoning home
        client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB initialized at: {Config.CHROMA_DB_PATH}")
        return client

    def index(self, chunks: List[TextChunk], collection_name: str) -> int:
        """
        Embed all chunks and store in ChromaDB.

        Steps:
          1. Compute embeddings with sentence-transformers
          2. Create a ChromaDB collection (embedding_function=None)
          3. Add vectors + texts + metadata to the collection

        Args:
            chunks:          List of TextChunk objects to index
            collection_name: Usually the uploaded filename

        Returns:
            Number of chunks successfully indexed
        """
        if not chunks:
            logger.warning("No chunks to index — skipping")
            return 0

        clean_name = self._clean_collection_name(collection_name)

        # Delete the old collection if it exists
        # (prevents duplicates when re-uploading the same file)
        try:
            self.chroma_client.delete_collection(clean_name)
            logger.debug(f"Deleted existing collection: {clean_name}")
        except Exception:
            pass  # Didn't exist — that's fine

        # Create a NEW collection with embedding_function=None
        # This is the KEY LINE that prevents the ONNX error.
        # It tells ChromaDB: "Don't use your built-in embedder.
        # I will provide my own vectors."
        collection = self.chroma_client.create_collection(
            name=clean_name,
            embedding_function=None,      # ← disables ONNX / built-in embedder
            metadata={"hnsw:space": "cosine"},  # cosine similarity for text
        )

        # Compute embeddings using our sentence-transformers model
        texts = [chunk.text for chunk in chunks]
        logger.info(f"Embedding {len(texts)} chunks with {Config.EMBEDDING_MODEL}...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )

        # Prepare IDs and metadata for ChromaDB
        ids       = [f"{clean_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [chunk.to_metadata_dict() for chunk in chunks]

        # Insert in batches of 100 (prevents memory issues on large docs)
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),  # list of lists
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
            logger.debug(f"Indexed batch {start}–{end}")

        logger.info(f"✅ Indexed {len(chunks)} chunks → collection '{clean_name}'")
        return len(chunks)

    def dense_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = None,
    ) -> List[dict]:
        """
        Semantic vector search.

        Finds chunks that are *semantically similar* to the query,
        even if they don't share the same keywords.

        We embed the query ourselves (same model as indexing)
        and pass the vector directly to ChromaDB's query method.

        Returns:
            List of dicts: {text, score, metadata, retrieval_method}
        """
        top_k      = top_k or Config.DENSE_TOP_K
        clean_name = self._clean_collection_name(collection_name)

        # Get the collection (returns error if not found)
        try:
            collection = self.chroma_client.get_collection(
                name=clean_name,
                embedding_function=None,   # again — don't trigger ONNX
            )
        except Exception as e:
            logger.warning(f"Collection '{clean_name}' not found: {e}")
            return []

        # Embed the query with our model
        query_vector = self.model.encode([query])[0].tolist()

        count = collection.count()
        if count == 0:
            return []

        # Query ChromaDB with our pre-computed vector
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            include=["documents", "distances", "metadatas"],
        )

        # Format results
        formatted = []
        if results and results["documents"] and results["documents"][0]:
            for text, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to a 0–1 similarity score
                similarity = 1.0 - (dist / 2.0)
                formatted.append({
                    "text": text,
                    "score": round(similarity, 4),
                    "metadata": meta,
                    "retrieval_method": "dense",
                })

        return formatted

    def get_collection_names(self) -> List[str]:
        """List all indexed document collections."""
        return [col.name for col in self.chroma_client.list_collections()]

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a document has already been indexed."""
        clean_name = self._clean_collection_name(collection_name)
        return clean_name in self.get_collection_names()

    def delete_collection(self, collection_name: str):
        """Remove a document's index from the vector store."""
        clean_name = self._clean_collection_name(collection_name)
        try:
            self.chroma_client.delete_collection(clean_name)
            logger.info(f"Deleted collection: {clean_name}")
        except Exception as e:
            logger.warning(f"Could not delete '{clean_name}': {e}")

    def _clean_collection_name(self, name: str) -> str:
        """
        ChromaDB collection names must be:
          - 3 to 63 characters
          - Alphanumeric, underscores, or hyphens only
          - Start and end with alphanumeric character

        This method converts any filename into a valid name.
        e.g. "My Contract (2024).pdf" → "my-contract--2024-"
        """
        import re
        clean = os.path.splitext(name)[0]              # remove extension
        clean = re.sub(r"[^a-zA-Z0-9_-]", "-", clean) # replace bad chars
        clean = clean[:63].strip("-")                   # enforce length limit
        if len(clean) < 3:
            clean = clean + "doc"                       # ensure min length
        return clean.lower()
