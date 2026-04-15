
# src/retrieval/hybrid_retriever.py

# STAGE 5 — Hybrid Retrieval: BM25 + Dense + Reranker


from typing import List
from loguru import logger
from config import Config


class HybridRetriever:
    """
    Three-stage retrieval pipeline.

    Stage 1 — BM25 sparse search (keyword matching)
    Stage 2 — Dense vector search (semantic similarity)
    Stage 3 — Cross-encoder reranker (precise relevance scoring)

    If BM25 index is unavailable (empty chunks), falls back to dense-only.
    If reranker is unavailable, falls back to score-sorted results.
    """

    def __init__(self, embedder, chunks: list):
        self.embedder = embedder
        self.chunks   = chunks or []

        self.bm25_index = self._build_bm25(self.chunks)
        self.reranker   = self._load_reranker()

    # BM25 Index Builder
 
    def _build_bm25(self, chunks: list):
        """
        Build BM25 index.
        Returns None (safely) if chunks is empty or all chunks are blank.
        Never raises ZeroDivisionError.
        """
        # Guard 1: empty list
        if not chunks:
            logger.warning("BM25: no chunks — using dense-only retrieval")
            return None

        try:
            from rank_bm25 import BM25Okapi

            # Tokenise every chunk text
            tokenized = [chunk.text.lower().split() for chunk in chunks if chunk.text]

            # Guard 2: filter out empty token lists
            # (blank chunk text produces [] which causes ZeroDivisionError)
            tokenized = [tokens for tokens in tokenized if tokens]

            # Guard 3: still nothing after filtering
            if not tokenized:
                logger.warning("BM25: all chunks empty after tokenising — using dense-only")
                return None

            index = BM25Okapi(tokenized)
            logger.info(f"BM25 index ready: {len(tokenized)} documents")
            return index

        except ImportError:
            logger.warning("rank-bm25 not installed — BM25 disabled")
            return None
        except ZeroDivisionError:
            # Explicit catch as final safety net
            logger.warning("BM25 ZeroDivisionError caught — using dense-only retrieval")
            return None
        except Exception as e:
            logger.warning(f"BM25 build error: {e} — using dense-only retrieval")
            return None

    # Reranker Loader
   
    def _load_reranker(self):
        """Load cross-encoder reranker. Returns None if unavailable."""
        try:
            from sentence_transformers import CrossEncoder
            try:
                model = CrossEncoder(Config.RERANKER_MODEL, local_files_only=True)
            except TypeError:
                model = CrossEncoder(Config.RERANKER_MODEL)
            logger.info(f"Reranker ready: {Config.RERANKER_MODEL}")
            return model
        except Exception as e:
            logger.warning(f"Reranker unavailable ({e}) — will rank by score only")
            return None

    # Main Retrieve Method

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = None,
    ) -> List[dict]:
        """
        Full 3-stage retrieval.
        Returns top_k most relevant chunks for the query.
        Each result: {text, score, metadata, retrieval_method}
        """
        top_k = top_k or Config.RERANKER_TOP_K

        # Stage 1: BM25 (returns [] if index is None)
        bm25_results = self._bm25_search(query, top_k=Config.BM25_TOP_K)
        logger.debug(f"BM25: {len(bm25_results)} candidates")

        # Stage 2: Dense vector search
        dense_results = self.embedder.dense_search(
            query=query,
            collection_name=collection_name,
            top_k=Config.DENSE_TOP_K,
        )
        logger.debug(f"Dense: {len(dense_results)} candidates")

        # Merge + deduplicate
        all_candidates = self._merge(bm25_results, dense_results)
        logger.debug(f"Merged pool: {len(all_candidates)} unique candidates")

        if not all_candidates:
            return []

        # Stage 3: Rerank (or sort by score if reranker unavailable)
        if self.reranker and len(all_candidates) > top_k:
            results = self._rerank(query, all_candidates, top_k)
        else:
            results = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:top_k]

        return results

  
    # Stage 1: BM25 Search

    def _bm25_search(self, query: str, top_k: int) -> List[dict]:
        """Keyword search. Returns [] safely if BM25 index is None."""
        if not self.bm25_index or not self.chunks:
            return []

        tokenized_query = query.lower().split()
        if not tokenized_query:
            return []

        try:
            scores = self.bm25_index.get_scores(tokenized_query)
        except Exception as e:
            logger.warning(f"BM25 scoring error: {e}")
            return []

        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self.chunks) and float(scores[idx]) > 0:
                chunk = self.chunks[idx]
                results.append({
                    "text":             chunk.text,
                    "score":            float(scores[idx]),
                    "metadata":         chunk.to_metadata_dict(),
                    "retrieval_method": "bm25",
                })
        return results

  
    # Merge + Deduplicate

    def _merge(
        self,
        bm25_results: List[dict],
        dense_results: List[dict],
    ) -> List[dict]:
        """
        Merge BM25 and dense results, deduplicate by first 100 chars of text.
        Normalise BM25 scores to [0,1] range before merging.
        """
        seen   = set()
        merged = []

        # Normalise BM25 scores
        if bm25_results:
            max_score = max(r["score"] for r in bm25_results)
            if max_score > 0:
                for r in bm25_results:
                    r["score"] = round(r["score"] / max_score, 4)

        for result in bm25_results + dense_results:
            key = result["text"][:100]
            if key not in seen:
                seen.add(key)
                merged.append(result)

        return merged

    
    # Stage 3: Cross-Encoder Reranker

    def _rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int,
    ) -> List[dict]:
        """Rerank candidates using cross-encoder. Returns top_k results."""
        pairs  = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["reranker_score"] = float(scores[i])
            candidate["score"]          = float(scores[i])

        return sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)[:top_k]
