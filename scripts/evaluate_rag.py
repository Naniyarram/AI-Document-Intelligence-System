#!/usr/bin/env python3
# ruff: noqa: E402
"""
RAG Pipeline Evaluation Harness
================================
Evaluates the RAG pipeline against 15 ground-truth Q&A pairs.

Tier 1 — Deterministic / Local (no LLM calls, runs offline):
  - Answer Precision   : token-level, set-intersection based
  - Answer Recall      : token-level, set-intersection based
  - Answer F1          : harmonic mean of precision and recall
  - Key Fact Recall    : exact substring match on critical facts
  - Context Relevance  : token overlap between retrieved chunks and ground truth

NOTE: These are lightweight lexical-overlap metrics, NOT standard semantic
benchmarks (BLEU/ROUGE/embedding-based). They serve as a reproducible baseline
suitable for a GenAI portfolio project evaluated on a free-tier API.

Tier 2 — LLM-as-a-Judge (Faithfulness, Answer Relevancy):
  NOT evaluated in this run. Requires the `ragas` + `datasets` packages
  (not in requirements.txt) and would consume 60+ free-tier API calls per run
  with no reliable rate-limit guarantee. Will be marked N/A in the output.

Output:
  - Console table with per-question and aggregate results
  - JSON artifact: artifacts/evaluation/rag_evaluation_results.json

Usage:
    python scripts/evaluate_rag.py
"""

# ── Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError) ─────────
import sys as _sys
import io as _io
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── PyTorch / Windows DLL pre-init ───────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST
    _ = _ST
except ImportError:
    pass

import sys
import os
import json
import time
import re
import tempfile
import statistics

# Add project root so pipeline imports resolve correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import DocumentPipeline  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
#  Sample Document
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_DOCUMENT = (
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

# ─────────────────────────────────────────────────────────────────────────────
#  15 Ground-Truth Evaluation Pairs
# ─────────────────────────────────────────────────────────────────────────────
EVALUATION_SET = [
    {
        "id": "Q01",
        "question": "What is the total fee amount for the consulting engagement?",
        "ground_truth": "The total fee is $84,200.00.",
        "key_facts": ["84,200", "$84,200"],
    },
    {
        "id": "Q02",
        "question": "What are the three payment milestones?",
        "ground_truth": "$25,000 upon execution, $30,000 at Day 45 interim report, $29,200 at final report.",
        "key_facts": ["25,000", "30,000", "29,200"],
    },
    {
        "id": "Q03",
        "question": "Who are the two parties in this agreement?",
        "ground_truth": "Acme Corporation and Jane Smith Consulting LLC.",
        "key_facts": ["Acme", "Jane Smith"],
    },
    {
        "id": "Q04",
        "question": "What is the late payment interest rate?",
        "ground_truth": "1.5% per month.",
        "key_facts": ["1.5%", "per month"],
    },
    {
        "id": "Q05",
        "question": "How can the agreement be terminated early?",
        "ground_truth": "Either party may terminate upon thirty days prior written notice.",
        "key_facts": ["thirty", "30", "written notice"],
    },
    {
        "id": "Q06",
        "question": "What happens to fees if the agreement is terminated early?",
        "ground_truth": "The consultant is compensated for services performed up to termination on a pro-rata basis.",
        "key_facts": ["pro-rata", "termination"],
    },
    {
        "id": "Q07",
        "question": "What state's laws govern this agreement?",
        "ground_truth": "The State of Delaware.",
        "key_facts": ["Delaware"],
    },
    {
        "id": "Q08",
        "question": "How long does the confidentiality obligation last after termination?",
        "ground_truth": "Two years after termination.",
        "key_facts": ["two", "2", "years"],
    },
    {
        "id": "Q09",
        "question": "What is the total duration of the agreement?",
        "ground_truth": "Six months from the effective date.",
        "key_facts": ["six", "6", "months"],
    },
    {
        "id": "Q10",
        "question": "What services does the consultant provide?",
        "ground_truth": (
            "Strategic advisory services for digital transformation including technology "
            "assessment, roadmap development, vendor evaluation, and change management."
        ),
        "key_facts": ["digital transformation", "technology", "roadmap", "vendor"],
    },
    {
        "id": "Q11",
        "question": "What is the invoice reference number?",
        "ground_truth": "INV-2024-0341.",
        "key_facts": ["INV-2024-0341"],
    },
    {
        "id": "Q12",
        "question": "Where will disputes be resolved?",
        "ground_truth": "Through binding arbitration in Wilmington, Delaware.",
        "key_facts": ["arbitration", "Wilmington"],
    },
    {
        "id": "Q13",
        "question": "When was the agreement signed?",
        "ground_truth": "January 15, 2024.",
        "key_facts": ["January", "2024"],
    },
    {
        "id": "Q14",
        "question": "What is Acme Corporation's address?",
        "ground_truth": "742 Evergreen Terrace, Springfield, IL 62704.",
        "key_facts": ["742", "Evergreen", "Springfield"],
    },
    {
        "id": "Q15",
        "question": "When must the final deliverable be submitted?",
        "ground_truth": "Within 90 days of the effective date.",
        "key_facts": ["90 days"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  Metric Functions (Tier 1 — Deterministic, zero LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    """Lowercase, remove punctuation, return set of tokens."""
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def compute_f1(prediction: str, ground_truth: str) -> dict:
    """
    Token-level Precision, Recall, F1.

    Method: set-intersection of lowercased, punctuation-stripped tokens.
    This is a lexical baseline — not equivalent to ROUGE-1 (which uses
    counts, not sets) or embedding-based semantic similarity.
    """
    pred_tokens  = _tokenize(prediction)
    truth_tokens = _tokenize(ground_truth)

    if not pred_tokens or not truth_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common    = pred_tokens & truth_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(truth_tokens)
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


def compute_key_fact_recall(answer: str, key_facts: list) -> float:
    """
    Exact substring match of critical facts in the generated answer.
    Fraction of key_facts found → [0.0, 1.0].
    """
    if not key_facts:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for fact in key_facts if fact.lower() in answer_lower)
    return round(found / len(key_facts), 4)


def compute_context_relevance(retrieved_texts: list, ground_truth: str) -> float:
    """
    Token overlap between the retrieved chunk texts and the ground-truth answer.
    Measures whether the retriever surfaced the relevant passage.
    """
    if not retrieved_texts:
        return 0.0
    combined     = " ".join(retrieved_texts)
    truth_tokens = _tokenize(ground_truth)
    ctx_tokens   = _tokenize(combined)
    if not truth_tokens:
        return 0.0
    overlap = truth_tokens & ctx_tokens
    return round(len(overlap) / len(truth_tokens), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Evaluation Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n" + "=" * 64)
    print("  RAG Pipeline Evaluation Harness")
    print("=" * 64)
    print(
        "\n  Metric methodology: Tier 1 Deterministic (lexical-overlap).\n"
        "  Faithfulness / Answer Relevancy (LLM-as-a-Judge) are NOT\n"
        "  evaluated — ragas is not installed and would require 60+\n"
        "  free-tier LLM calls with no rate-limit guarantee.\n"
    )

    # ── 1. Write sample document to a temp file ───────────────────────────
    tmp_dir  = tempfile.mkdtemp(prefix="rag_eval_")
    tmp_file = os.path.join(tmp_dir, "consulting_agreement.txt")
    with open(tmp_file, "w", encoding="utf-8") as fh:
        fh.write(SAMPLE_DOCUMENT)
    print(f"  ✓ Sample document written ({len(SAMPLE_DOCUMENT):,} chars)\n")

    # ── 2. Initialise pipeline ────────────────────────────────────────────
    print("  ⏳ Initialising pipeline...")
    pipeline = DocumentPipeline()
    print("  ✓ Pipeline ready\n")

    # ── 3. Index the document ─────────────────────────────────────────────
    collection_name = "consulting_agreement.txt"
    print("  ⏳ Indexing document...")
    idx = pipeline.index(tmp_file, original_filename=collection_name)
    print(
        f"  ✓ Indexed {idx['total_chunks']} chunks "
        f"in {idx['processing_time_sec']}s\n"
    )

    # ── 4. Evaluation loop ────────────────────────────────────────────────
    results       = []
    total         = len(EVALUATION_SET)
    failed_llm    = 0
    succeeded_llm = 0

    for i, item in enumerate(EVALUATION_SET, 1):
        qid = item["id"]
        q   = item["question"]
        gt  = item["ground_truth"]
        kf  = item["key_facts"]

        q_display = q[:55] + "…" if len(q) > 55 else q
        print(f"  [{i:>2}/{total}] {qid}: {q_display}")

        # ── LLM answer generation ─────────────────────────────────────
        answer          = ""
        llm_error       = None
        retrieved_texts = []

        for attempt in range(1, 4):
            try:
                response = pipeline.query(q, collection_name=collection_name)
                answer   = response.get("answer", "")

                # Detect LLM-level errors vs real answers
                error_phrases = [
                    "api authentication failed",
                    "rate limit",
                    "no documents have been indexed",
                    "couldn't find relevant",
                    "llm error",
                ]
                if any(p in answer.lower() for p in error_phrases):
                    raise RuntimeError(f"LLM returned error response: {answer[:120]}")

                succeeded_llm += 1
                break

            except Exception as exc:
                llm_error = str(exc)
                if attempt < 3:
                    wait = attempt * 4
                    print(f"         ⚠ Attempt {attempt} failed ({exc}). Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print("         ✗ All 3 attempts failed. Skipping LLM answer.")
                    failed_llm += 1
                    answer = ""

        # ── Re-retrieve context for Context Relevance metric ──────────
        # pipeline.query() doesn't return raw chunks directly.
        # We call the retriever explicitly to get the text for metric computation.
        try:
            from src.retrieval.hybrid_retriever import HybridRetriever
            chunks = pipeline.all_chunks.get(collection_name, [])
            retriever = HybridRetriever(embedder=pipeline.embedder, chunks=chunks)
            retrieved_chunks = retriever.retrieve(
                query=q,
                collection_name=collection_name,
            )
            retrieved_texts = [r["text"] for r in retrieved_chunks]
        except Exception as exc:
            print(f"         ⚠ Retrieval error for metrics: {exc}")
            retrieved_texts = []

        # ── Compute deterministic metrics ─────────────────────────────
        f1  = compute_f1(answer, gt)                        if answer else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        kfr = compute_key_fact_recall(answer, kf)           if answer else 0.0
        cr  = compute_context_relevance(retrieved_texts, gt)

        results.append({
            "id":               qid,
            "question":         q,
            "ground_truth":     gt,
            "answer":           answer,
            "llm_error":        llm_error,
            "retrieved_count":  len(retrieved_texts),
            "f1":               f1,
            "key_fact_recall":  kfr,
            "context_relevance": cr,
            # Tier 2 — NOT evaluated
            "faithfulness":     None,
            "answer_relevancy": None,
        })

        status = "✓" if not llm_error else "✗"
        print(
            f"         {status} P={f1['precision']:.2f}  R={f1['recall']:.2f}  "
            f"F1={f1['f1']:.2f}  Facts={kfr:.2f}  Ctx={cr:.2f}"
        )

        # Rate-limit buffer between questions
        time.sleep(3)

    # ── 5. Aggregate metrics ───────────────────────────────────────────────
    n = len(results)
    answered = [r for r in results if not r["llm_error"]]

    def _mean(vals):
        return round(statistics.mean(vals), 4) if vals else None

    def _std(vals):
        return round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0

    prec_vals = [r["f1"]["precision"]  for r in answered]
    rec_vals  = [r["f1"]["recall"]     for r in answered]
    f1_vals   = [r["f1"]["f1"]         for r in answered]
    kfr_vals  = [r["key_fact_recall"]  for r in answered]
    cr_vals   = [r["context_relevance"] for r in results]  # CR doesn't need LLM

    aggregate = {
        "answer_precision":          _mean(prec_vals),
        "answer_precision_std":      _std(prec_vals),
        "answer_recall":             _mean(rec_vals),
        "answer_recall_std":         _std(rec_vals),
        "answer_f1":                 _mean(f1_vals),
        "answer_f1_std":             _std(f1_vals),
        "key_fact_recall":           _mean(kfr_vals),
        "key_fact_recall_std":       _std(kfr_vals),
        "context_relevance":         _mean(cr_vals),
        "context_relevance_std":     _std(cr_vals),
        # Tier 2 — explicitly marked as not evaluated
        "faithfulness":              None,
        "answer_relevancy":          None,
    }

    # ── 6. Console report ──────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  EVALUATION RESULTS")
    print("=" * 64)
    print(f"  Total questions   : {n}")
    print(f"  LLM answers OK    : {succeeded_llm}")
    print(f"  LLM answers FAILED: {failed_llm}")
    print(f"  Context Relevance samples: {n} (retrieval-only, always runs)")
    print()
    print("  ┌─────────────────────────┬──────────┬──────────┐")
    print("  │ Metric                  │  Mean    │  StdDev  │")
    print("  ├─────────────────────────┼──────────┼──────────┤")

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A  "

    print(f"  │ Answer Precision        │ {_fmt(aggregate['answer_precision'])} │ {_fmt(aggregate['answer_precision_std'])} │")
    print(f"  │ Answer Recall           │ {_fmt(aggregate['answer_recall'])} │ {_fmt(aggregate['answer_recall_std'])} │")
    print(f"  │ Answer F1               │ {_fmt(aggregate['answer_f1'])} │ {_fmt(aggregate['answer_f1_std'])} │")
    print(f"  │ Key Fact Recall         │ {_fmt(aggregate['key_fact_recall'])} │ {_fmt(aggregate['key_fact_recall_std'])} │")
    print(f"  │ Context Relevance       │ {_fmt(aggregate['context_relevance'])} │ {_fmt(aggregate['context_relevance_std'])} │")
    print("  │ Faithfulness (LLM-J)    │   N/A — not evaluated          │")
    print("  │ Answer Relevancy (LLM-J)│   N/A — not evaluated          │")
    print("  └─────────────────────────┴──────────┴──────────┘")
    print()
    print("  Note: Precision/Recall/F1 are lexical token-overlap metrics,")
    print("  not ROUGE/BLEU or embedding-based semantic similarity.")
    print()

    # Per-question breakdown
    print("  Per-Question Breakdown:")
    for r in results:
        st = "✓" if not r["llm_error"] else "✗"
        print(
            f"    [{st}] {r['id']}: "
            f"F1={r['f1']['f1']:.2f}  Facts={r['key_fact_recall']:.2f}  "
            f"Ctx={r['context_relevance']:.2f}  | {r['question'][:48]}"
        )
    print("=" * 64)

    # ── 7. Save JSON artifact ──────────────────────────────────────────────
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    artifact_dir = os.path.join(project_root, "artifacts", "evaluation")
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_path = os.path.join(artifact_dir, "rag_evaluation_results.json")

    report = {
        "schema_version":       "1.0",
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "document":             collection_name,
        "model_backend":        "HuggingFace / OpenRouter (free tier)",
        "num_questions":        n,
        "llm_answers_succeeded": succeeded_llm,
        "llm_answers_failed":   failed_llm,
        "metric_methodology": {
            "answer_precision":   "lexical token-overlap set-intersection",
            "answer_recall":      "lexical token-overlap set-intersection",
            "answer_f1":          "harmonic mean of precision and recall",
            "key_fact_recall":    "exact substring match on critical facts",
            "context_relevance":  "token overlap between retrieved chunks and ground truth",
            "faithfulness":       "NOT_EVALUATED — ragas not installed",
            "answer_relevancy":   "NOT_EVALUATED — ragas not installed",
        },
        "aggregate_metrics":    aggregate,
        "per_question_results": results,
    }

    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\n  ✓ JSON artifact saved to:\n    {artifact_path}\n")

    # ── 8. Cleanup temp files ──────────────────────────────────────────────
    try:
        os.unlink(tmp_file)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"  Done. {n} questions evaluated ({succeeded_llm} answered, {failed_llm} failed).\n")
    return report


if __name__ == "__main__":
    run_evaluation()
