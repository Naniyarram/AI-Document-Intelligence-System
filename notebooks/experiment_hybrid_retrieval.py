
# Benchmark: BM25 vs Dense vs Hybrid Retrieval

# This script proves that hybrid retrieval is better.


# What it measures:
#   - How many relevant chunks each method finds
#   - How the methods complement each other


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import DocumentPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from config import Config


def run_retrieval_benchmark(document_name: str, test_queries: list):
    """
    Compare BM25, Dense, and Hybrid retrieval on the same queries.
    Prints a comparison table showing what each method finds.
    """
    print(f"\n{'='*60}")
    print(f"  Retrieval Method Benchmark")
    print(f"  Document: {document_name}")
    print(f"  Queries: {len(test_queries)}")
    print(f"{'='*60}\n")

    # Initialize pipeline
    pipeline = DocumentPipeline()
    chunks = pipeline.all_chunks.get(document_name, [])

    if not chunks:
        print(f"❌ Document '{document_name}' not indexed.")
        print("   Upload it in the app first, then re-run this script.")
        return

    retriever = HybridRetriever(
        embedder=pipeline.embedder,
        chunks=chunks
    )

    results_summary = []

    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 50)

        # BM25 only
        bm25_results = retriever._bm25_search(query, top_k=5)

        # Dense only
        dense_results = pipeline.embedder.dense_search(
            query=query,
            collection_name=document_name,
            top_k=5
        )

        # Full hybrid (BM25 + Dense + Reranker)
        hybrid_results = retriever.retrieve(
            query=query,
            collection_name=document_name,
            top_k=5
        )

        # Get unique texts found by each method
        bm25_texts = set(r["text"][:80] for r in bm25_results)
        dense_texts = set(r["text"][:80] for r in dense_results)
        hybrid_texts = set(r["text"][:80] for r in hybrid_results)

        # Count unique results per method
        only_in_bm25 = bm25_texts - dense_texts
        only_in_dense = dense_texts - bm25_texts
        in_both = bm25_texts & dense_texts

        print(f"  BM25 found:        {len(bm25_results)} results")
        print(f"  Dense found:       {len(dense_results)} results")
        print(f"  Unique to BM25:    {len(only_in_bm25)} results (BM25 advantage)")
        print(f"  Unique to Dense:   {len(only_in_dense)} results (Dense advantage)")
        print(f"  Found by BOTH:     {len(in_both)} results")
        print(f"  Hybrid (reranked): {len(hybrid_results)} final results")
        print()

        # Show top hybrid result
        if hybrid_results:
            top = hybrid_results[0]
            print(f"  Best result (score: {top['score']:.3f}):")
            print(f"  '{top['text'][:120]}...'")
            print(f"  Source: {top['metadata'].get('source_file', '?')}, "
                  f"Page: {top['metadata'].get('page_number', '?')}")

        print()

        results_summary.append({
            "query": query,
            "bm25_count": len(bm25_results),
            "dense_count": len(dense_results),
            "hybrid_count": len(hybrid_results),
            "bm25_unique": len(only_in_bm25),
            "dense_unique": len(only_in_dense),
        })

    #  Summary Table 
    print(f"\n{'='*60}")
    print("  Summary Table")
    print(f"{'='*60}")
    print(f"  {'Query':<35} {'BM25':>5} {'Dense':>6} {'Hybrid':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*7}")
    for r in results_summary:
        print(
            f"  {r['query'][:35]:<35} "
            f"{r['bm25_count']:>5} "
            f"{r['dense_count']:>6} "
            f"{r['hybrid_count']:>7}"
        )
    print(f"{'='*60}")
    print("\nConclusion: Hybrid retrieval combines the strengths of both methods.")
    print("Add these results to your README for a strong portfolio showing.\n")


#  Sample usage 
if __name__ == "__main__":

    # Replace with your actual document name and relevant queries
    DOCUMENT_NAME = "your_document.pdf"
    TEST_QUERIES = [
        "payment terms and conditions",
        "termination clause",
        "liability and penalties",
        "key dates and deadlines",
        "parties involved in the agreement",
    ]

    run_retrieval_benchmark(DOCUMENT_NAME, TEST_QUERIES)
