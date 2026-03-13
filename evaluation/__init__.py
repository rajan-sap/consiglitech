"""
RAG Evaluation Pipeline
=======================

This package provides end-to-end evaluation for the DocIntel RAG system.

Modules
-------
config.py              — Centralized evaluation settings (paths, metrics, model config).
dataset_generator.py   — Generates synthetic QA eval datasets from ingested document chunks.
metrics.py             — Custom metrics: MRR, Hit Rate @k, Latency, Token Cost.
evaluate.py            — Main orchestrator: runs RAGAS + custom metrics, prints report.

Usage
-----
    # Step 1: Generate an eval dataset from your vector store
    python -m evaluation.dataset_generator

    # Step 2: Run the full evaluation suite
    python -m evaluation.evaluate

Output is saved to evaluation/eval_data/
"""

from evaluation.config import EVAL_CONFIG
