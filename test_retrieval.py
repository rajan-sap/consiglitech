"""
Test script to evaluate embedding effectiveness and retrieval quality.

Usage:
    python test_retrieval.py "your query here"
    python test_retrieval.py  # Uses default test queries
"""

import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.constants import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH


def get_vector_store():
    """Load the existing vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name="documents",
    )


def l2_to_cosine_similarity(l2_distance: float) -> float:
    """Convert L2 distance to cosine similarity for normalized vectors."""
    # For normalized vectors: L2² = 2 * (1 - cosine_similarity)
    # Therefore: cosine_similarity = 1 - (L2² / 2)
    return 1 - (l2_distance ** 2) / 2


def test_retrieval(query: str, k: int = 5) -> None:
    """
    Test retrieval for a given query.
    
    Args:
        query: The search query
        k: Number of results to retrieve
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(query, k=k)
    
    if not results:
        print("No results found.")
        return
    
    print(f"\nTop {len(results)} Results:\n")
    
    for i, (doc, l2_distance) in enumerate(results, 1):
        similarity = l2_to_cosine_similarity(l2_distance)
        print(f"[{i}] Similarity: {similarity:.4f} (L2: {l2_distance:.4f})")
        print(f"    Page: {doc.metadata.get('page_number', 'N/A')}")
        print(f"    Type: {doc.metadata.get('document_type', 'N/A')}")
        print(f"    File: {doc.metadata.get('file_name', 'N/A')}")
        
        # Show preview of content
        content = doc.page_content.strip()
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"    Content:\n    {preview}\n")


def run_test_suite() -> None:
    """Run a suite of test queries to evaluate retrieval."""
    test_queries = [
        "What were the key factors that contributed to the opposition party's narrow victory in the recent elections?",
        "What potential effects might this interest rate hike have on consumer spending and borrowing?",
        "What is the purpose of Daniel Hayes' $5 million donation to the global wildlife conservation fund?",
        "How do you think the unexpected outcome of this election will impact the state's politics and policy agenda for the next term?",
        "How does this bill aim to bridge the educational gap between affluent and disadvantaged communities?",
    ]
    
    print("\n" + "="*60)
    print("RETRIEVAL TEST SUITE")
    print("="*60)
    
    for query in test_queries:
        test_retrieval(query, k=3)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom query from command line
        query = " ".join(sys.argv[1:])
        test_retrieval(query, k=5)
    else:
        # Run test suite
        run_test_suite()
