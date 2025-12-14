
"""
Hybrid Retriever: Metadata + Vector Embeddings
"""
import os
import re
from openai import OpenAI
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingestion.constants import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH
from retrieval.utils import extract_metadata_from_query
import chromadb

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment.")
client = OpenAI(api_key=api_key)

# Step 1: Implementation of ChromaDB connection
def connect_chromadb(path="./chroma_db"):
    client = chromadb.PersistentClient(path=path)
    collections = client.list_collections()
    return client, collections


class Retriever:
    # Helper functions are now imported from utils.py for modularity

    def __init__(self, filtered_ids: Optional[List[str]] = None, query: Optional[str] = None):
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
        self.filtered_ids = filtered_ids
    


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


    def search(self, query, k=5, metadata_filter=None):
        """
        Perform top-k vector search on documents matching metadata_filter (if provided), else on all documents.

        Args:
            query: The search query (text)
            k: Number of top results to return

        Returns:
            List of dicts with document, score, and metadata
        """
        chroma_filter = None
        if metadata_filter:
            # Remove None values from filter
            clean_filter = {k: v for k, v in metadata_filter.items() if v is not None}
            if len(clean_filter) == 0:
                chroma_filter = None
            elif len(clean_filter) == 1:
                chroma_filter = clean_filter
            else:
                chroma_filter = {"$and": [{k: v} for k, v in clean_filter.items()]}
        results = self.vector_store.similarity_search_with_score(query, k=k, filter=chroma_filter)
        return self._format_results(results)


# Step 2: Implementation of query decompostion
def decompose_query(query, model="gpt-4-1106-preview"):
    """
    Decompose a query into single-shot factual questions using LLM. Returns a list of queries (strings).
    """
    try:
        system_prompt = (
            "You are a helpful assistant that decomposes complex queries into atomic factual questions. "
            "Return only a list of decomposed queries, one per line. Do not include any metadata or explanations."
        )
        # Few-shot example: all content is plain string
        example_user = "Provide a summary of revenue figures for Tesla, BMW, and Ford over the past three years."
        example_assistant = (
            "What was Tesla's revenue for the year 2020?\n"
            "What was Tesla's revenue for the year 2021?\n"
            "What was Tesla's revenue for the year 2022?\n"
            "What was BMW's revenue for the year 2020?\n"
            "What was BMW's revenue for the year 2021?\n"
            "What was BMW's revenue for the year 2022?\n"
            "What was Ford's revenue for the year 2020?\n"
            "What was Ford's revenue for the year 2021?\n"
            "What was Ford's revenue for the year 2022?"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example_user},
            {"role": "assistant", "content": example_assistant},
            {"role": "user", "content": query},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
            queries = [line.strip() for line in content.splitlines() if line.strip()]
            queries = [q for q in queries if not q.lower().startswith("error communicating with openai")]
            return queries
        else:
            return []
    except Exception as e:
        return [f"Error communicating with OpenAI: {e}"]


# Step 3: Implementaition of metadata extraction from query
def extract_metadata_from_query(single_shot_query):
    """
    Extract only company, document_type, and year for metadata filtering.
    Handles both News Article and Annual Report cases.
    """
    # Normalize possessives: "Ford's" -> "Ford"
    query_clean = re.sub(r"'s\\b", "", single_shot_query)

    # Extract year (4 consecutive digits)
    year_match = re.search(r"(20\d{2})", query_clean)
    year = year_match.group(1) if year_match else None

    # Extract company from a known list
    companies = ["BMW", "Tesla", "Ford"]
    company = next((c for c in companies if re.search(rf'\b{re.escape(c)}\b', query_clean, re.IGNORECASE)), None)

    # Extract document type (case-insensitive, allow both 'News Article' and 'Annual Report')
    doc_types = ["annual report", "news article"]
    document_type = next((d.title() for d in doc_types if d in query_clean.lower()), None)

    return {
        "company": company,
        "document_type": document_type,
        "year": year
    }
   

# Step 4: Retrieve aggregated context based on decomposed queries
def retrieve_aggregated_context(query, retriever):
    aggregated_context = ""
    decomposed_queries = decompose_query(query)
    for single_query in decomposed_queries:
        metadata_for_query = extract_metadata_from_query(single_query)
        results = retriever.search(single_query, k=3, metadata_filter=metadata_for_query)
        for res in results:
            # import pdb; pdb.set_trace()
            aggregated_context += f"Document: {res['document']}\nMetadata: {res['metadata']}\n\n"
    return aggregated_context