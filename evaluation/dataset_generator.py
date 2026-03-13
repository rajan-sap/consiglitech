"""
Synthetic Evaluation Dataset Generator
=======================================

Generates QA pairs from ingested document chunks using GPT-4.
Each QA pair includes:
  - question: A factual question answerable from the chunk
  - ground_truth: The correct answer derived from the chunk
  - ground_truth_contexts: The source chunk(s) containing the answer
  - metadata: Company, document type, year

Usage:
    python -m evaluation.dataset_generator
"""

import json
import os
import random
import re
from typing import Dict, List

from evaluation.config import (
    CHUNKS_TO_SAMPLE_PER_REPORT,
    EVAL_DATA_DIR,
    EVAL_DATASET_FILE,
    KNOWN_REPORTS,
    QA_GENERATION_MODEL,
    QA_PAIRS_PER_REPORT,
)
from ingestion.constants import COMPANY_FOLDERS, VECTOR_DB_PATH, EMBEDDING_MODEL_NAME
from retrieval.retriever import Retriever
from llm_config import llm_client as client


# =============================================================================
# CHUNK SAMPLING
# =============================================================================


def sample_chunks_for_report(retriever: Retriever, company: str, year: str, n: int) -> List[Dict]:
    """
    Sample n chunks from the vector store filtered by company AND year.
    Uses diverse queries to get a spread of topics from the report.
    Returns a list of dicts with 'document' and 'metadata' keys.
    """
    broad_queries = [
        f"{company} {year} revenue and financial performance",
        f"{company} {year} annual report highlights and key figures",
        f"{company} {year} business operations strategy and outlook",
        f"{company} {year} risk factors and challenges",
        f"{company} {year} market share product segments",
        f"{company} {year} sustainability ESG initiatives",
        f"{company} {year} research development and innovation",
    ]
    all_results = []
    seen_docs = set()

    for query in broad_queries:
        results = retriever.search(
            query,
            k=n,
            metadata_filter={"company": company, "year": year},
        )
        for r in results:
            # Deduplicate by document content (first 100 chars)
            doc_key = r["document"][:100]
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                all_results.append(r)

    # Shuffle and return up to n chunks
    random.shuffle(all_results)
    return all_results[:n]


# =============================================================================
# QA GENERATION
# =============================================================================

QA_GENERATION_PROMPT = """You are an expert at creating evaluation datasets for RAG systems.

Given the following document chunk, generate {n} question-answer pairs.

Rules:
1. Each question must be answerable ONLY from the provided chunk.
2. Each answer must be factual, concise, and directly supported by the chunk.
3. Questions should be diverse: mix factual (what, how much), comparative, and analytical.
4. Include specific numbers, dates, or names when available in the chunk.
5. Return ONLY valid JSON — no markdown, no explanation.

Document chunk:
\"\"\"
{chunk}
\"\"\"

Metadata:
- Company: {company}
- Document type: {doc_type}
- Year: {year}

Return a JSON array of objects, each with "question" and "answer" keys:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""


def generate_qa_pairs(chunk: Dict, n_pairs: int = 2) -> List[Dict]:
    """
    Use GPT-4 to generate QA pairs from a single document chunk.
    Returns a list of dicts with question, ground_truth, ground_truth_contexts, metadata.
    """
    metadata = chunk.get("metadata", {})
    company = metadata.get("company", "Unknown")
    doc_type = metadata.get("document_type", "Unknown")
    year = metadata.get("year", "Unknown")

    prompt = QA_GENERATION_PROMPT.format(
        n=n_pairs,
        chunk=chunk["document"],
        company=company,
        doc_type=doc_type,
        year=year,
    )

    try:
        response = client.chat.completions.create(
            model=QA_GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown fences if present (```json ... ```)
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        parsed = json.loads(content)

        # Robustly extract the QA pairs list from whatever structure GPT returns
        pairs = None
        if isinstance(parsed, list):
            pairs = parsed
        elif isinstance(parsed, dict):
            # Case 1: single QA pair returned as a flat dict
            if "question" in parsed and "answer" in parsed:
                pairs = [parsed]
            else:
                # Case 2: look for the first value that is a list of dicts
                for v in parsed.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        pairs = v
                        break
        if not pairs:
            print(f"  [WARN] Unexpected JSON structure: {content[:200]}")
            return []

        qa_items = []
        for pair in pairs[:n_pairs]:
            if not isinstance(pair, dict) or "question" not in pair or "answer" not in pair:
                continue
            qa_items.append({
                "question": pair["question"],
                "ground_truth": pair["answer"],
                "ground_truth_contexts": [chunk["document"]],
                "metadata": {
                    "company": company,
                    "document_type": doc_type,
                    "year": year,
                },
            })
        return qa_items

    except Exception as e:
        print(f"  [WARN] QA generation failed for chunk: {e}")
        return []


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def generate_eval_dataset() -> str:
    """
    Generate a full evaluation dataset by sampling chunks across all reports
    (company × year) and creating QA pairs with GPT-4.

    Returns the path to the saved JSON file.
    """
    print("=" * 60)
    print("  Evaluation Dataset Generator")
    print("=" * 60)
    print(f"  Target: {QA_PAIRS_PER_REPORT} QA pairs per report")
    print(f"  Reports: {len(KNOWN_REPORTS)} ({', '.join(f'{c} {y}' for c, y in KNOWN_REPORTS)})")
    print(f"  Expected total: {QA_PAIRS_PER_REPORT * len(KNOWN_REPORTS)} QA pairs")

    # Initialize retriever once
    print("\n[1/3] Initializing retriever...")
    retriever = Retriever()

    # Sample chunks per report and generate QA pairs
    print(f"[2/3] Sampling {CHUNKS_TO_SAMPLE_PER_REPORT} chunks per report...")
    all_qa_pairs = []

    for report_idx, (company, year) in enumerate(KNOWN_REPORTS, 1):
        print(f"\n  --- {company} {year} ({report_idx}/{len(KNOWN_REPORTS)}) ---")
        chunks = sample_chunks_for_report(retriever, company, year, CHUNKS_TO_SAMPLE_PER_REPORT)
        print(f"  Sampled {len(chunks)} unique chunks")

        if not chunks:
            print(f"  [WARN] No chunks found for {company} {year}, skipping.")
            continue

        # Calculate how many QA pairs per chunk to reach target
        pairs_per_chunk = max(1, QA_PAIRS_PER_REPORT // max(len(chunks), 1))
        generated = 0

        for i, chunk in enumerate(chunks):
            if generated >= QA_PAIRS_PER_REPORT:
                break
            remaining = QA_PAIRS_PER_REPORT - generated
            n = min(pairs_per_chunk, remaining)
            qa_pairs = generate_qa_pairs(chunk, n_pairs=n)
            all_qa_pairs.extend(qa_pairs)
            generated += len(qa_pairs)
            print(f"  Chunk {i+1}/{len(chunks)}: generated {len(qa_pairs)} QA pairs")

        print(f"  Total for {company} {year}: {generated} QA pairs")

    # Save dataset
    print(f"\n[3/3] Saving dataset ({len(all_qa_pairs)} total QA pairs)...")
    os.makedirs(EVAL_DATA_DIR, exist_ok=True)
    output_path = os.path.join(EVAL_DATA_DIR, EVAL_DATASET_FILE)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")
    print("=" * 60)
    return output_path


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    generate_eval_dataset()
