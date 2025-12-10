
"""
Hybrid Retriever: Metadata + Vector Embeddings
"""

from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingestion.constants import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH
from retrieval.utils import extract_metadata_from_query


class HybridRetriever:
    METADATA_KEYS = ["file_name", "document_type", "company", "year", "page_number"]

    # Helper functions are now imported from utils.py for modularity

    def __init__(self):
        # Initialize the embedding model (used to convert text to vectors)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},)
        # Initialize the Chroma vector store for document retrieval
        # Note: The embedding_function is used by Chroma to vectorize queries and documents internally
        self.vector_store = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name="documents",
        )
    


    def _format_results(self, results: list) -> list:
        """Format raw results (doc, score) as dicts."""
        similarity = lambda l2: 1 - (l2 ** 2) / 2  # L2 to cosine similarity
        return [
            {
                "document": doc.page_content,
                "metadata": doc.metadata,
                "cosine_similarity": similarity(score),
            }
            for doc, score in results
        ]


    def search(
        self,
        query: str,
        k: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
        auto_extract_metadata: bool = True,
    ) -> list:
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
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return self._format_results(results)
    

 
        