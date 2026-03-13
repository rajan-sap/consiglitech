"""
Evaluation Configuration
========================

Centralized settings for the RAG evaluation pipeline.
Edit values here to adjust eval behavior without touching other files.
"""

import os

# =============================================================================
# PATHS
# =============================================================================

# Directory where generated eval datasets and results are saved
EVAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "eval_data")

# Default eval dataset filename
EVAL_DATASET_FILE = "eval_dataset.json"

# Default results filename
EVAL_RESULTS_FILE = "eval_results.json"

# =============================================================================
# DATASET GENERATION
# =============================================================================

# How many QA pairs to generate per report (e.g. BMW 2021, Ford 2023, etc.)
QA_PAIRS_PER_REPORT = 10

# How many chunks to sample per report for QA generation
CHUNKS_TO_SAMPLE_PER_REPORT = 20

# Reports to evaluate — each (company, year) pair maps to one annual report
# Auto-discovered at runtime from ChromaDB; these are the known reports:
KNOWN_REPORTS = [
    ("BMW", "2021"),
    ("BMW", "2022"),
    ("BMW", "2023"),
    ("Ford", "2021"),
    ("Ford", "2022"),
    ("Ford", "2023"),
    ("Tesla", "2022"),
    ("Tesla", "2023"),
]

# LLM model used for synthetic QA generation
QA_GENERATION_MODEL = "gpt-4-1106-preview"

# =============================================================================
# RETRIEVAL EVALUATION
# =============================================================================

# Top-k values to evaluate for Hit Rate
HIT_RATE_K_VALUES = [1, 3, 5, 10]

# Default top-k for retrieval during eval
RETRIEVAL_TOP_K = 5

# =============================================================================
# RAGAS METRICS
# =============================================================================

# Which RAGAS metrics to compute (must match ragas metric names)
RAGAS_METRICS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]

# =============================================================================
# LLM-AS-JUDGE
# =============================================================================

# Model used by RAGAS for LLM-as-judge evaluations
JUDGE_MODEL = "gpt-4-1106-preview"

# Temperature for judge model (low = deterministic)
JUDGE_TEMPERATURE = 0.0

# =============================================================================
# LATENCY
# =============================================================================

# Number of runs per query for latency measurement
LATENCY_RUNS_PER_QUERY = 1  # Set higher for more accurate P50/P95 (costs more)

# =============================================================================
# AGGREGATE CONFIG DICT (for easy import)
# =============================================================================

EVAL_CONFIG = {
    "eval_data_dir": EVAL_DATA_DIR,
    "eval_dataset_file": EVAL_DATASET_FILE,
    "eval_results_file": EVAL_RESULTS_FILE,
    "qa_pairs_per_report": QA_PAIRS_PER_REPORT,
    "chunks_to_sample_per_report": CHUNKS_TO_SAMPLE_PER_REPORT,
    "known_reports": KNOWN_REPORTS,
    "qa_generation_model": QA_GENERATION_MODEL,
    "hit_rate_k_values": HIT_RATE_K_VALUES,
    "retrieval_top_k": RETRIEVAL_TOP_K,
    "ragas_metrics": RAGAS_METRICS,
    "judge_model": JUDGE_MODEL,
    "judge_temperature": JUDGE_TEMPERATURE,
    "latency_runs": LATENCY_RUNS_PER_QUERY,
}
