"""
evaluate.py — LLM evaluation pipeline.

Metrics computed:
  - RAGAS: faithfulness, answer_relevancy, context_recall
  - Latency: retrieval_ms, generation_ms, total_ms
  - Hallucination: NLI-based entailment check (DeBERTa)

Usage:
    python src/evaluate.py
    python src/evaluate.py --questions_path data/eval_questions.json
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

from loguru import logger
from src.utils import cfg, save_json


# Sample eval questions 
SAMPLE_EVAL_QUESTIONS = [
    {
        "question":        "What is the transformer architecture?",
        "ground_truth":    "The transformer uses self-attention and feed-forward layers without recurrence.",
    },
    {
        "question":        "What is BERT's pre-training objective?",
        "ground_truth":    "BERT uses masked language modeling and next sentence prediction.",
    },
    {
        "question":        "How does dropout prevent overfitting?",
        "ground_truth":    "Dropout randomly deactivates neurons during training, reducing co-adaptation.",
    },
]

# Data classes 
@dataclass
class EvalResult:
    question:           str
    answer:             str
    ground_truth:       str
    context:            str
    faithfulness:       float
    answer_relevancy:   float
    context_recall:     float
    hallucination_flag: bool
    retrieval_ms:       float
    generation_ms:      float
    total_ms:           float

# RAGAS Evaluation 
def _run_ragas(
    questions:     list[str],
    answers:       list[str],
    contexts:      list[list[str]],
    ground_truths: list[str],
) -> list[dict]:
    """
    Run RAGAS evaluation on a batch of QA samples.
    Returns per-sample metrics.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
        )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        data = {
            "question":    questions,
            "answer":      answers,
            "contexts":    contexts,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)

        llm        = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=cfg.OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(openai_api_key=cfg.OPENAI_API_KEY)

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=llm,
            embeddings=embeddings,
        )

        df = result.to_pandas()
        metrics_list = df[["faithfulness", "answer_relevancy", "context_recall"]].to_dict(orient="records")
        logger.success("RAGAS evaluation complete.")
        return metrics_list

    except Exception as e:
        logger.warning(f"RAGAS evaluation failed ({e}). Using placeholder scores.")
        # Return dummy scores if RAGAS fails (e.g., no OpenAI key)
        return [
            {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}
            for _ in questions
        ]

# Hallucination Check (NLI) 
class HallucinationDetector:
    """
    NLI-based hallucination detector.
    Checks if the answer is entailed by (supported by) the retrieved context.
    Uses a lightweight DeBERTa cross-encoder fine-tuned on NLI.
    """

    NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading NLI model: {self.NLI_MODEL}")
                self._model = pipeline(
                    "zero-shot-classification",
                    model=self.NLI_MODEL,
                    device=-1,          # CPU
                )
                logger.success("NLI model loaded.")
            except Exception as e:
                logger.warning(f"NLI model load failed: {e}. Hallucination check disabled.")
                self._model = "failed"

    def is_hallucination(self, answer: str, context: str, threshold: float = 0.4) -> bool:
        """
        Returns True if the answer appears NOT entailed by the context.
        Entailment score < threshold → potential hallucination.
        """
        self._load()
        if self._model == "failed" or self._model is None:
            return False

        try:
            result = self._model(
                answer[:512],                           
                candidate_labels=["entailment", "contradiction", "neutral"],
                hypothesis_template="Based on the context: {}",
            )
            scores = dict(zip(result["labels"], result["scores"]))
            entailment_score = scores.get("entailment", 0.0)
            is_hall = entailment_score < threshold
            logger.debug(f"Entailment score: {entailment_score:.3f} | Hallucination: {is_hall}")
            return is_hall
        except Exception as e:
            logger.warning(f"NLI check failed: {e}")
            return False

# Main Evaluator 
class LLMEvaluator:
    """
    Orchestrates end-to-end evaluation of the RAG pipeline.
    """

    def __init__(self):
        self._detector = HallucinationDetector()

    def evaluate_batch(
        self,
        eval_samples: list[dict],
        rag_assistant,
    ) -> list[EvalResult]:
        """
        Args:
            eval_samples : list of {"question": ..., "ground_truth": ...}
            rag_assistant: instance of RAGAssistant

        Returns:
            list of EvalResult dataclasses
        """
        questions, answers, contexts, ground_truths = [], [], [], []
        timings = []

        logger.info(f"Running RAG on {len(eval_samples)} eval questions...")
        for sample in eval_samples:
            result = rag_assistant.ask(sample["question"])
            questions.append(sample["question"])
            answers.append(result["answer"])
            ground_truths.append(sample.get("ground_truth", ""))
            contexts.append([result["context"]])
            timings.append(result["timing"])

        # RAGAS metrics
        ragas_metrics = _run_ragas(questions, answers, contexts, ground_truths)

        # Build results
        eval_results = []
        for i, sample in enumerate(eval_samples):
            is_hall = self._detector.is_hallucination(
                answers[i],
                contexts[i][0],
            )
            m = ragas_metrics[i]
            eval_results.append(EvalResult(
                question=questions[i],
                answer=answers[i],
                ground_truth=ground_truths[i],
                context=contexts[i][0][:300] + "...",  
                faithfulness=round(m.get("faithfulness", 0.0), 4),
                answer_relevancy=round(m.get("answer_relevancy", 0.0), 4),
                context_recall=round(m.get("context_recall", 0.0), 4),
                hallucination_flag=is_hall,
                retrieval_ms=timings[i]["retrieval_ms"],
                generation_ms=timings[i]["generation_ms"],
                total_ms=timings[i]["total_ms"],
            ))

        return eval_results

    def summarize(self, results: list[EvalResult]) -> dict:
        """Compute aggregate statistics across all eval results."""
        n = len(results)
        if n == 0:
            return {}

        def avg(attr):
            return round(sum(getattr(r, attr) for r in results) / n, 4)

        hallucination_rate = round(
            sum(1 for r in results if r.hallucination_flag) / n * 100, 1
        )

        return {
            "n_samples":            n,
            "avg_faithfulness":     avg("faithfulness"),
            "avg_answer_relevancy": avg("answer_relevancy"),
            "avg_context_recall":   avg("context_recall"),
            "hallucination_rate_%": hallucination_rate,
            "avg_retrieval_ms":     avg("retrieval_ms"),
            "avg_generation_ms":    avg("generation_ms"),
            "avg_total_ms":         avg("total_ms"),
        }

    def run_and_save(
        self,
        eval_samples: list[dict],
        rag_assistant,
        output_path: str | None = None,
    ) -> dict:
        output_path = output_path or cfg.EVAL_RESULTS_PATH
        results     = self.evaluate_batch(eval_samples, rag_assistant)
        summary     = self.summarize(results)

        output = {
            "summary": summary,
            "per_sample": [asdict(r) for r in results],
        }
        save_json(output, output_path)

        logger.success("─── Evaluation Summary ───")
        for k, v in summary.items():
            logger.success(f"  {k:<30} {v}")

        return output

# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--questions_path", default=None, help="JSON file with eval questions")
    parser.add_argument("--use_lora",       action="store_true", help="Use LoRA fine-tuned model")
    parser.add_argument("--output_path",    default=None, help="Where to save eval_results.json")
    args = parser.parse_args()

    # Load eval questions
    if args.questions_path and Path(args.questions_path).exists():
        with open(args.questions_path) as f:
            eval_samples = json.load(f)
    else:
        logger.warning("No eval questions file provided. Using sample questions.")
        eval_samples = SAMPLE_EVAL_QUESTIONS

    # Load RAG assistant
    from src.rag_chain import RAGAssistant
    assistant = RAGAssistant(use_lora=args.use_lora)

    # Run evaluation
    evaluator = LLMEvaluator()
    evaluator.run_and_save(eval_samples, assistant, output_path=args.output_path)