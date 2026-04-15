
# src/evaluation/ragas_evaluator.py



# What RAGAs measures:
#   - Faithfulness:      Is the answer grounded in the retrieved context?
#                        (Prevents hallucination — LLM should not make things up)
#   - Answer Relevance:  Does the answer actually address the question asked?
#   - Context Recall:    Did retrieval find the right chunks?
#   - Context Precision: Are the retrieved chunks actually relevant?





from typing import List, Dict, Optional
from loguru import logger


class RAGAsEvaluator:
    """
    Evaluates the RAG pipeline using the RAGAs framework.

    Requires: pip install ragas datasets

    Usage:
        evaluator = RAGAsEvaluator(pipeline)

        questions = [
            "What are the payment terms?",
            "Who is the contracting party?",
        ]
        ground_truths = [
            "Payment is due within 30 days.",
            "The contracting party is Acme Corp.",
        ]

        results = evaluator.evaluate(questions, ground_truths, "contract.pdf")
        evaluator.print_report(results)
    """

    def __init__(self, pipeline):
        """
        Args:
            pipeline: DocumentPipeline instance with a document already indexed
        """
        self.pipeline = pipeline
        self._ragas_available = self._check_ragas()

    def _check_ragas(self) -> bool:
        """Check if RAGAs is installed."""
        try:
            import ragas
            return True
        except ImportError:
            logger.warning(
                "RAGAs not installed.\n"
                "Install with: pip install ragas datasets\n"
                "Then re-run evaluation."
            )
            return False

    def evaluate(
        self,
        questions: List[str],
        ground_truths: List[str],
        document_name: str
    ) -> Dict:
        """
        Run full RAGAs evaluation on a set of questions.

        Args:
            questions:     List of test questions
            ground_truths: List of correct answers (one per question)
            document_name: Name of the indexed document to query

        Returns:
            Dict with metric scores and per-question breakdown
        """
        if not self._ragas_available:
            return {"error": "RAGAs not installed. Run: pip install ragas datasets"}

        if len(questions) != len(ground_truths):
            raise ValueError("questions and ground_truths must have the same length")

        logger.info(f"Running RAGAs evaluation: {len(questions)} questions...")

        #  Collect answers and contexts
        # We need to run every question through the pipeline
        # and collect: the answer, the retrieved contexts
        answers = []
        contexts_list = []

        for i, question in enumerate(questions):
            logger.info(f"  Evaluating question {i+1}/{len(questions)}: {question[:50]}...")

            # Run the query pipeline
            result = self.pipeline.query(
                question=question,
                collection_name=document_name,
                mode="qa"
            )

            answers.append(result["answer"])

            # Collect the raw retrieved chunk texts
            # (RAGAs needs the actual context, not just the answer)
            # We re-run retrieval to get the chunks
            from src.retrieval.hybrid_retriever import HybridRetriever
            chunks = self.pipeline.all_chunks.get(document_name, [])
            retriever = HybridRetriever(
                embedder=self.pipeline.embedder,
                chunks=chunks
            )
            retrieved = retriever.retrieve(
                query=question,
                collection_name=document_name
            )
            context_texts = [r["text"] for r in retrieved]
            contexts_list.append(context_texts)

        # Run RAGAs Metrics
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            )

            # RAGAs expects a Hugging Face Dataset with specific columns
            eval_data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts_list,
                "ground_truth": ground_truths,
            }
            dataset = Dataset.from_dict(eval_data)

            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                    context_precision,
                ],
            )

            # Convert to plain dict
            scores = result.to_pandas().mean().to_dict()

            return {
                "scores": {
                    "faithfulness": round(scores.get("faithfulness", 0), 3),
                    "answer_relevancy": round(scores.get("answer_relevancy", 0), 3),
                    "context_recall": round(scores.get("context_recall", 0), 3),
                    "context_precision": round(scores.get("context_precision", 0), 3),
                },
                "num_questions": len(questions),
                "per_question": [
                    {
                        "question": q,
                        "answer": a,
                        "ground_truth": g,
                        "context_count": len(c),
                    }
                    for q, a, g, c in zip(questions, answers, ground_truths, contexts_list)
                ]
            }

        except Exception as e:
            logger.error(f"RAGAs evaluation failed: {e}")
            return {
                "error": str(e),
                "answers_collected": len(answers)
            }

    def print_report(self, results: Dict):
        """Print a nicely formatted evaluation report to the console."""
        if "error" in results:
            print(f"\n❌ Evaluation Error: {results['error']}")
            return

        scores = results.get("scores", {})

        print("\n" + "═" * 50)
        print("  RAGAs Evaluation Report")
        print("═" * 50)
        print(f"  Questions evaluated: {results.get('num_questions', 0)}")
        print()
        print(f"  Faithfulness:       {scores.get('faithfulness', 0):.1%}")
        print(f"    → Are answers grounded in the document?")
        print()
        print(f"  Answer Relevancy:   {scores.get('answer_relevancy', 0):.1%}")
        print(f"    → Do answers address the question?")
        print()
        print(f"  Context Recall:     {scores.get('context_recall', 0):.1%}")
        print(f"    → Did retrieval find the right chunks?")
        print()
        print(f"  Context Precision:  {scores.get('context_precision', 0):.1%}")
        print(f"    → Are retrieved chunks relevant?")
        print("═" * 50)

        # Overall health check
        avg = sum(scores.values()) / len(scores) if scores else 0
        if avg >= 0.85:
            print(f"  Overall: ✅ Excellent ({avg:.1%})")
        elif avg >= 0.70:
            print(f"  Overall: ⚠️  Good ({avg:.1%}) — room to improve chunking/retrieval")
        else:
            print(f"  Overall: ❌ Needs work ({avg:.1%}) — check chunk size and retrieval")
        print("═" * 50 + "\n")
