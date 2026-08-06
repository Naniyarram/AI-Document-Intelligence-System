#!/usr/bin/env python3
"""
RAG Pipeline Evaluation Harness
================================
Indexes a sample consulting agreement and evaluates the RAG pipeline
against 15 ground-truth question-answer pairs.

Metrics (deterministic — no LLM calls needed):
  - Answer F1:          Token-level precision, recall, F1
  - Key Fact Recall:    Fraction of critical facts present in the answer
  - Context Relevance:  Token overlap between retrieved chunks and ground truth

Optional (LLM-based — may fail under rate limits):
  - Faithfulness:       LLM-as-judge grounding check

Outputs:
  - Console table with per-question and aggregate results
  - JSON report: scripts/evaluation_report.json
  - Bar chart:   scripts/evaluation_chart.png

Usage:
    python scripts/evaluate_rag.py
"""

# ── PyTorch pre-initialisation (Windows DLL pattern) ─────
try:
    from sentence_transformers import SentenceTransformer
    _ = SentenceTransformer
except ImportError:
    pass

import sys
import os
import json
import time
import tempfile
import statistics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import DocumentPipeline  # noqa: E402

# ─────────────────────────────────────────────────────────
#  Sample Document (Consulting Agreement)
# ─────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────
#  Evaluation Set — 15 questions
# ─────────────────────────────────────────────────────────
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
        "ground_truth": "Strategic advisory services for digital transformation including technology assessment, roadmap development, vendor evaluation, and change management.",
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


# ─────────────────────────────────────────────────────────
#  Metric Functions (deterministic — no LLM calls)
# ─────────────────────────────────────────────────────────

def compute_f1(prediction: str, ground_truth: str) -> dict:
    """Compute token-level precision, recall, and F1."""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    if not pred_tokens or not truth_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = pred_tokens & truth_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_key_fact_recall(answer: str, key_facts: list) -> float:
    """Check what fraction of key facts appear in the answer."""
    if not key_facts:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for fact in key_facts if fact.lower() in answer_lower)
    return round(found / len(key_facts), 4)


def compute_context_relevance(retrieved_texts: list, ground_truth: str) -> float:
    """Token overlap between retrieved chunks and ground truth."""
    if not retrieved_texts:
        return 0.0
    combined = " ".join(retrieved_texts).lower()
    truth_tokens = set(ground_truth.lower().split())
    combined_tokens = set(combined.split())
    if not truth_tokens:
        return 0.0
    overlap = truth_tokens & combined_tokens
    return round(len(overlap) / len(truth_tokens), 4)


# ─────────────────────────────────────────────────────────
#  Chart Generation
# ─────────────────────────────────────────────────────────

def generate_chart(aggregate: dict, output_path: str):
    """Generate a bar chart of aggregate metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = {
            "Answer F1": aggregate["answer_f1"],
            "Key Fact\nRecall": aggregate["key_fact_recall"],
            "Context\nRelevance": aggregate["context_relevance"],
        }

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            metrics.keys(),
            metrics.values(),
            color=["#3b82f6", "#10b981", "#8b5cf6"],
            width=0.5,
            edgecolor="white",
            linewidth=1.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, metrics.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold",
            )

        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("RAG Pipeline Evaluation — Deterministic Metrics", fontsize=14, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  OK Chart saved to {output_path}")
    except ImportError:
        print("  WARN matplotlib not installed — skipping chart generation")
    except Exception as e:
        print(f"  WARN Chart generation failed: {e}")


# ─────────────────────────────────────────────────────────
#  Main Evaluation Loop
# ─────────────────────────────────────────────────────────

def run_evaluation():
    """Run the full evaluation pipeline."""
    print("\n" + "=" * 60)
    print("  RAG Pipeline Evaluation Harness")
    print("=" * 60)

    # 1. Write sample document to a temp file
    tmp_dir = tempfile.mkdtemp(prefix="rag_eval_")
    tmp_file = os.path.join(tmp_dir, "consulting_agreement.txt")
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(SAMPLE_DOCUMENT)
    print(f"\n  OK Sample document written ({len(SAMPLE_DOCUMENT)} chars)")

    # 2. Initialise pipeline
    print("  WAIT Initialising pipeline...")
    pipeline = DocumentPipeline()
    print("  OK Pipeline ready")

    # 3. Index the document
    print("  WAIT Indexing document...")
    collection_name = "consulting_agreement.txt"
    index_result = pipeline.index(tmp_file, original_filename=collection_name)
    print(f"  OK Indexed: {index_result['total_chunks']} chunks in {index_result['processing_time_sec']}s\n")

    # 4. Run evaluation queries
    results = []
    total = len(EVALUATION_SET)

    for i, item in enumerate(EVALUATION_SET, 1):
        q = item["question"]
        gt = item["ground_truth"]
        kf = item["key_facts"]
        qid = item["id"]

        q_display = q[:55] + "…" if len(q) > 55 else q
        print(f"  [{i:>2}/{total}] {qid}: {q_display}")

        try:
            response = pipeline.query(q, collection_name=collection_name)
        except Exception as e:
            print(f"         WARN Query failed: {e}")
            results.append({
                "id": qid,
                "question": q,
                "ground_truth": gt,
                "answer": "",
                "f1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "key_fact_recall": 0.0,
                "context_relevance": 0.0,
                "faithfulness": -1,
                "error": str(e),
            })
            time.sleep(5)
            continue

        answer = response.get("answer", "")
        retrieved_texts = response.get("retrieved_texts", [])
        faithfulness = response.get("faithfulness", {})
        faith_score = faithfulness.get("score", -1) if isinstance(faithfulness, dict) else -1

        f1 = compute_f1(answer, gt)
        kfr = compute_key_fact_recall(answer, kf)
        cr = compute_context_relevance(retrieved_texts, gt)

        results.append({
            "id": qid,
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "f1": f1,
            "key_fact_recall": kfr,
            "context_relevance": cr,
            "faithfulness": faith_score,
        })

        faith_display = f"{faith_score}" if faith_score >= 0 else "N/A"
        print(f"         F1={f1['f1']:.2f}  Facts={kfr:.2f}  Ctx={cr:.2f}  Faith={faith_display}")

        # Delay between queries to avoid rate limiting
        time.sleep(2)

    # ─── Aggregate Metrics ────────────────────────────────
    n = len(results)
    f1_scores = [r["f1"]["f1"] for r in results]
    kfr_scores = [r["key_fact_recall"] for r in results]
    cr_scores = [r["context_relevance"] for r in results]

    avg_f1 = round(statistics.mean(f1_scores), 4) if f1_scores else 0
    avg_precision = round(statistics.mean([r["f1"]["precision"] for r in results]), 4) if n else 0
    avg_recall = round(statistics.mean([r["f1"]["recall"] for r in results]), 4) if n else 0
    avg_kfr = round(statistics.mean(kfr_scores), 4) if kfr_scores else 0
    avg_cr = round(statistics.mean(cr_scores), 4) if cr_scores else 0

    std_f1 = round(statistics.stdev(f1_scores), 4) if len(f1_scores) > 1 else 0.0
    std_kfr = round(statistics.stdev(kfr_scores), 4) if len(kfr_scores) > 1 else 0.0
    std_cr = round(statistics.stdev(cr_scores), 4) if len(cr_scores) > 1 else 0.0

    valid_faith = [r["faithfulness"] for r in results if r["faithfulness"] >= 0]
    avg_faith = round(statistics.mean(valid_faith), 1) if valid_faith else -1

    # ─── Console Report ───────────────────────────────────
    print()
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Questions: {n}  |  Document: consulting_agreement.txt")
    print()
    print("  ---------------------------------------------")
    print("  | Metric                |  Mean   |  StdDev |")
    print("  ---------------------------------------------")
    print(f"  | Answer F1             |  {avg_f1:.4f} |  {std_f1:.4f} |")
    print(f"  | Key Fact Recall       |  {avg_kfr:.4f} |  {std_kfr:.4f} |")
    print(f"  | Context Relevance     |  {avg_cr:.4f} |  {std_cr:.4f} |")
    faith_str = f"{avg_faith:.0f}/100" if avg_faith >= 0 else "  N/A "
    print(f"  | Faithfulness (LLM)    | {faith_str:>7s} |    -    |")
    print("  ---------------------------------------------")
    print()

    # Per-question breakdown
    print("  Per-Question Breakdown:")
    for r in results:
        status = "[V]" if r["key_fact_recall"] >= 0.5 else "[X]"
        print(
            f"    {status} {r['id']}: F1={r['f1']['f1']:.2f}  "
            f"Facts={r['key_fact_recall']:.2f}  "
            f"Ctx={r['context_relevance']:.2f}  "
            f"| {r['question'][:50]}"
        )
    print("=" * 60)

    # ─── Save JSON Report ─────────────────────────────────
    aggregate = {
        "answer_f1": avg_f1,
        "answer_f1_std": std_f1,
        "answer_precision": avg_precision,
        "answer_recall": avg_recall,
        "key_fact_recall": avg_kfr,
        "key_fact_recall_std": std_kfr,
        "context_relevance": avg_cr,
        "context_relevance_std": std_cr,
        "faithfulness": avg_faith,
    }

    report = {
        "evaluation_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "document": "consulting_agreement.txt",
        "num_questions": n,
        "aggregate_metrics": aggregate,
        "per_question_results": results,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  OK JSON report saved to {report_path}")

    # ─── Generate Chart ───────────────────────────────────
    chart_path = os.path.join(script_dir, "evaluation_chart.png")
    generate_chart(aggregate, chart_path)

    # ─── Cleanup ──────────────────────────────────────────
    try:
        os.unlink(tmp_file)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"\n  Done. {n} questions evaluated.\n")
    return report


if __name__ == "__main__":
    run_evaluation()
