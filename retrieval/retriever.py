
"""
Hybrid Retriever: Metadata + Vector Embeddings
"""

from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingestion.constants import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH


class HybridRetriever:
    METADATA_KEYS = ["file_name", "document_type", "company", "year", "page_number"]

    # Helper functions are now imported from utils.py for modularity

    def __init__(self):
        # Initialize the embedding model (used to convert text to vectors)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # Initialize the Chroma vector store for document retrieval
        # Note: The embedding_function is used by Chroma to vectorize queries and documents internally
        self.vector_store = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name="documents",
        )

    def search(
        self,
        query: str,
        k: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
        auto_extract_metadata: bool = True,
    ) -> List[Dict]:
        """
        Hybrid search: extract metadata from query if not provided, then filter and vector search.

        Args:
            query: The search query (text)
            k: Number of results to return
            metadata: Dict of metadata values to filter on
            auto_extract_metadata: If True, extract metadata from query if not provided

        Returns:
            List of dicts with document, score, and metadata
        """
        # If metadata is not provided, optionally extract it from the query
        from .utils import extract_metadata_from_query
        if metadata is None and auto_extract_metadata:
            metadata = extract_metadata_from_query(query)
        if metadata is None:
            raise ValueError("metadata filter is required and cannot be None")
        # Use helper from utils.py to build the metadata filter
        from .utils import build_chroma_metadata_filter, METADATA_KEYS
        chroma_filter = build_chroma_metadata_filter(metadata, METADATA_KEYS)
        # Perform hybrid search: Chroma will internally vectorize the query using self.embeddings
        # and filter results using the provided metadata
        results = self.vector_store.similarity_search_with_score(
            query, k=k*3, filter=chroma_filter
        )

        def l2_to_cosine(l2):
            # Convert L2 distance to cosine similarity for easier interpretation
            return 1 - (l2 ** 2) / 2

        # Build the result list with document content, metadata, and similarity scores
        scored = [
            {
                "document": doc.page_content,
                "metadata": doc.metadata,
                "cosine_similarity": l2_to_cosine(l2),
                "l2_distance": l2,
            }
            for doc, l2 in results
            if doc is not None
        ]
        # Sort results by cosine similarity (descending)
        scored.sort(key=lambda x: -x["cosine_similarity"])
        return scored[:k]

if __name__ == "__main__":
    retriever = HybridRetriever()
    # Example: metadata auto-extracted from query
    user_query = "How much revenue did Tesla generate in 2023?"
    # Extract metadata for debug
    metadata = HybridRetriever.extract_metadata_from_query(user_query)
    print(f"Metadata filter used: {metadata}")
    results = retriever.search(user_query, k=3)
    print(f"Number of results found: {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Cosine: {r['cosine_similarity']:.4f}")
        print(f"Metadata: {r['metadata']}")
        print(f"Content: {r['document'][:300]}...")