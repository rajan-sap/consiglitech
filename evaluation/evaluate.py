"""
RAG Evaluation Orchestrator
============================

Runs the full evaluation suite against a generated eval dataset:
  1. Loads eval dataset (from dataset_generator.py output)
  2. Runs retriever on each question → collects contexts
  3. Runs generator on each question → collects answers + token usage
  4. Computes RAGAS metrics (context precision/recall, faithfulness, relevancy, correctness)
  5. Computes custom metrics (MRR, Hit Rate @k, Latency, Token Cost)
  6. Derives hallucination rate = 1 - faithfulness
  7. Prints formatted report and saves results to JSON

Usage:
    python -m evaluation.evaluate
"""

import json
import os
import time
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from llm_config import llm_client as openai_client, LLM_MODEL, truncate_context

from evaluation.config import (
    EVAL_CONFIG,
    EVAL_DATA_DIR,
    EVAL_DATASET_FILE,
    EVAL_RESULTS_FILE,
    HIT_RATE_K_VALUES,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    LATENCY_RUNS_PER_QUERY,
    RETRIEVAL_TOP_K,
)
from evaluation.metrics import (
    compute_avg_token_cost,
    compute_hit_rate_dataset,
    compute_mrr_dataset,
    extract_token_usage,
    measure_latency,
)
from retrieval.retriever import Retriever, retrieve_aggregated_context

load_dotenv()


# =============================================================================
# DATA LOADING
# =============================================================================


def load_eval_dataset(path: str = None) -> List[Dict]:
    """Load the evaluation dataset from JSON."""
    if path is None:
        path = os.path.join(EVAL_DATA_DIR, EVAL_DATASET_FILE)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Eval dataset not found at {path}. "
            "Run 'python -m evaluation.dataset_generator' first."
        )

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"  Loaded {len(dataset)} eval samples from {path}")
    return dataset


# =============================================================================
# RETRIEVAL STEP
# =============================================================================


def run_retrieval(
    dataset: List[Dict], retriever: Retriever, top_k: int = RETRIEVAL_TOP_K
) -> List[List[Dict]]:
    """
    Run retriever.search() on each question in the dataset.
    Returns a list of search results (one per question).
    """
    all_results = []
    for i, sample in enumerate(dataset):
        question = sample["question"]
        results = retriever.search(question, k=top_k)
        all_results.append(results)
        if (i + 1) % 5 == 0:
            print(f"    Retrieved {i+1}/{len(dataset)}")
    return all_results


# =============================================================================
# GENERATION STEP
# =============================================================================


def run_generation(
    dataset: List[Dict], retriever: Retriever
) -> List[Dict]:
    """
    Run the full RAG pipeline (retrieve + generate) on each question.
    Returns a list of dicts with 'answer', 'contexts', 'token_usage', 'latency_ms'.
    """
    results = []
    for i, sample in enumerate(dataset):
        question = sample["question"]

        try:
            start = time.perf_counter()

            # Retrieve context (truncate to fit model's context window)
            aggregated_context = retrieve_aggregated_context(question, retriever)
            aggregated_context = truncate_context(aggregated_context)

            # Generate answer
            prompt = (
                f"Given the following data as a ground of truth, answer the original "
                f"question as accurately as possible.\n"
                f"Original Question: {question}\n"
                f"Information:\n{aggregated_context}"
            )

            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions "
                        "using provided question and context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            answer = response.choices[0].message.content.strip()
            token_usage = extract_token_usage(response)

            results.append({
                "answer": answer,
                "contexts": aggregated_context,
                "token_usage": token_usage,
                "latency_ms": elapsed_ms,
            })
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"    [WARN] Generation failed for Q{i+1}: {e}")
            results.append({
                "answer": "[generation failed]",
                "contexts": "",
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_ms": elapsed_ms,
            })

        if (i + 1) % 5 == 0:
            print(f"    Generated {i+1}/{len(dataset)}")

    return results


# =============================================================================
# RAGAS EVALUATION
# =============================================================================


def run_ragas_evaluation(
    dataset: List[Dict],
    generation_results: List[Dict],
) -> Dict[str, float]:
    """
    Run RAGAS metrics on the evaluation results.

    Returns a dict of metric_name -> score.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics.collections import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_community.chat_models import ChatOpenAI
        from datasets import Dataset
    except ImportError as exc:
        print(f"\n  [WARN] RAGAS dependencies missing ({exc}). Skipping RAGAS metrics.")
        print("  Install with: pip install ragas datasets langchain-community")
        return {}

    # Build RAGAS-compatible dataset
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for sample, gen_result in zip(dataset, generation_results):
        ragas_data["question"].append(sample["question"])
        ragas_data["answer"].append(gen_result["answer"])

        # Contexts: split aggregated context back into chunks
        ctx_text = gen_result["contexts"]
        chunks = [c.strip() for c in ctx_text.split("\n\n") if c.strip()]
        ragas_data["contexts"].append(chunks if chunks else [""])

        ragas_data["ground_truth"].append(sample["ground_truth"])

    ragas_dataset = Dataset.from_dict(ragas_data)

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]

    # Configure LLM for RAGAS judge (use LM Studio via OpenAI-compatible endpoint)
    from llm_config import LLM_BASE_URL, LLM_API_KEY
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            openai_api_base=LLM_BASE_URL,
            openai_api_key=LLM_API_KEY or "not-needed",
        )
    )

    print("  Running RAGAS evaluation (this may take a few minutes)...")
    result = ragas_evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=judge_llm,
        raise_exceptions=False,
    )

    return dict(result)


# =============================================================================
# REPORT
# =============================================================================


def print_report(scores: Dict):
    """Print a formatted evaluation report."""
    print("\n")
    print("=" * 60)
    print("  RAG EVALUATION REPORT")
    print("=" * 60)

    # RAGAS metrics
    ragas_keys = [
        "context_precision", "context_recall", "faithfulness",
        "answer_relevancy", "answer_correctness",
    ]
    print("\n  RAGAS Metrics (LLM-as-Judge)")
    print("  " + "-" * 45)
    for key in ragas_keys:
        if key in scores:
            val = scores[key]
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"  {key:<25} {val:.4f}  {bar}")

    # Hallucination rate
    if "hallucination_rate" in scores:
        val = scores["hallucination_rate"]
        print(f"\n  {'hallucination_rate':<25} {val:.4f}  (1 - faithfulness)")

    # Retrieval metrics
    print("\n  Retrieval Metrics")
    print("  " + "-" * 45)
    if "mrr" in scores:
        print(f"  {'MRR':<25} {scores['mrr']:.4f}")
    for key in sorted(scores.keys()):
        if key.startswith("hit_rate"):
            print(f"  {key:<25} {scores[key]:.4f}")

    # Latency
    print("\n  Latency")
    print("  " + "-" * 45)
    for key in ["latency_p50_ms", "latency_p95_ms", "latency_mean_ms"]:
        if key in scores:
            print(f"  {key:<25} {scores[key]:.0f} ms")

    # Token cost
    print("\n  Token Cost (per query avg)")
    print("  " + "-" * 45)
    for key in ["avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens"]:
        if key in scores:
            print(f"  {key:<25} {scores[key]:.0f}")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================


def run_evaluation(dataset_path: str = None) -> Dict:
    """
    Execute the complete evaluation pipeline.

    Steps:
      1. Load eval dataset
      2. Run retrieval (for MRR / Hit Rate)
      3. Run generation (for answers, latency, token cost)
      4. Run RAGAS (for LLM-as-judge metrics)
      5. Compute custom metrics
      6. Aggregate and report

    Returns:
        Dict of all evaluation scores.
    """
    print("=" * 60)
    print("  RAG Evaluation Pipeline")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/5] Loading eval dataset...")
    dataset = load_eval_dataset(dataset_path)

    # 2. Initialize retriever
    print("\n[2/5] Initializing retriever...")
    retriever = Retriever()

    # 3. Run retrieval for MRR / Hit Rate
    print("\n[3/5] Running retrieval evaluation...")
    retrieval_results = run_retrieval(dataset, retriever, top_k=max(HIT_RATE_K_VALUES))

    # 4. Run full generation pipeline
    print("\n[4/5] Running generation evaluation...")
    generation_results = run_generation(dataset, retriever)

    # 5. Compute all metrics
    print("\n[5/5] Computing metrics...")

    scores = {}

    # --- RAGAS (skip for local models — too slow and requires many LLM calls) ---
    # To enable RAGAS, set SKIP_RAGAS = False (recommended with OpenAI API only)
    SKIP_RAGAS = True
    if not SKIP_RAGAS:
        ragas_scores = run_ragas_evaluation(dataset, generation_results)
        scores.update(ragas_scores)
    else:
        print("  [INFO] RAGAS skipped (SKIP_RAGAS=True). Enable for LLM-as-judge metrics.")

    # --- Hallucination rate ---
    if "faithfulness" in scores:
        scores["hallucination_rate"] = 1.0 - scores["faithfulness"]

    # --- MRR ---
    ground_truth_contexts = [
        sample["ground_truth_contexts"][0] if sample.get("ground_truth_contexts") else ""
        for sample in dataset
    ]
    scores["mrr"] = compute_mrr_dataset(retrieval_results, ground_truth_contexts)

    # --- Hit Rate @k ---
    for k in HIT_RATE_K_VALUES:
        scores[f"hit_rate_at_{k}"] = compute_hit_rate_dataset(
            retrieval_results, ground_truth_contexts, k=k
        )

    # --- Latency ---
    latencies = [r["latency_ms"] for r in generation_results]
    latency_arr = np.array(latencies)
    scores["latency_p50_ms"] = float(np.percentile(latency_arr, 50))
    scores["latency_p95_ms"] = float(np.percentile(latency_arr, 95))
    scores["latency_mean_ms"] = float(np.mean(latency_arr))

    # --- Token cost ---
    token_usages = [r["token_usage"] for r in generation_results]
    token_stats = compute_avg_token_cost(token_usages)
    scores.update(token_stats)

    # --- Report ---
    print_report(scores)

    # --- Save results ---
    results_path = os.path.join(EVAL_DATA_DIR, EVAL_RESULTS_FILE)
    os.makedirs(EVAL_DATA_DIR, exist_ok=True)

    # Save full results with per-sample details
    full_results = {
        "summary": scores,
        "per_sample": [
            {
                "question": dataset[i]["question"],
                "ground_truth": dataset[i]["ground_truth"],
                "generated_answer": generation_results[i]["answer"],
                "latency_ms": generation_results[i]["latency_ms"],
                "token_usage": generation_results[i]["token_usage"],
            }
            for i in range(len(dataset))
        ],
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {results_path}")
    return scores


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_evaluation()
