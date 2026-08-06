import sys
import os
import time

# Ensure we can import the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import DocumentPipeline
from config import Config

def profile_pipeline():
    print("="*60)
    print("  RAG Pipeline Latency Profiler")
    print("="*60)

    start_total = time.time()
    
    # 1. Initialize Pipeline
    t0 = time.time()
    pipeline = DocumentPipeline()
    t1 = time.time()
    print(f"[Init] Pipeline Initialization: {t1-t0:.2f}s")
    
    # 2. Ingestion
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("CONSULTING AGREEMENT\n\nThis Consulting Agreement is entered into as of January 15, 2024, by and between Acme Corporation and Jane Smith Consulting LLC.\nClient shall pay Consultant a fixed fee of $84,200.00 for all services described herein.")
        doc_path = f.name

    try:
        t0 = time.time()
        summary = pipeline.index(doc_path)
        t1 = time.time()
        print(f"[Ingestion] Total Indexing time: {t1-t0:.2f}s")
        print(f"  - Pages: {summary['total_pages']}, Chunks: {summary['total_chunks']}")
    finally:
        os.unlink(doc_path)
    
    # 3. Query Execution (Non-Streaming)
    query = "What is the total fee amount?"
    t0 = time.time()
    result = pipeline.query(query, mode="qa")
    t1 = time.time()
    print(f"[Query (Sync)] Full pipeline end-to-end: {t1-t0:.2f}s")
    
    # 4. Query Execution (Streaming)
    t0 = time.time()
    stream = pipeline.query_stream(query, mode="qa")
    
    first_token_time = None
    for token in stream:
        if first_token_time is None:
            first_token_time = time.time()
    t1 = time.time()
    
    print(f"[Query (Stream)] First token latency: {first_token_time-t0:.2f}s")
    print(f"[Query (Stream)] Total stream completion latency (including metadata): {t1-t0:.2f}s")
    
    # Analyze the time diff between first token and total completion
    # If the difference is huge, it means metadata computation (faithfulness) is blocking.
    print("-"*60)
    print("  Detailed Breakdown Expected:")
    print(f"  Stream Generation Time (estimated): {first_token_time - t0:.2f}s")
    print(f"  Post-Stream Blocking Time (Faithfulness/Metadata): {t1 - first_token_time:.2f}s")
    
    print("="*60)

if __name__ == "__main__":
    profile_pipeline()
