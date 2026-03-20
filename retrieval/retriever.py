"""
Hybrid Retriever: Metadata + Vector Embeddings
"""
import os
import re
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingestion.constants import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH
from retrieval.utils import extract_metadata_from_query
from llm_config import llm_client as client, LLM_MODEL
import chromadb

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
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Initialize the Chroma vector store for document retrieval
        # Note: The embedding_function is used by Chroma to vectorize queries and documents internally
        self.vector_store = None
        self.is_available = False
        
        try:
            # Check if the vector store directory exists and has content
            import os
            chroma_path = VECTOR_DB_PATH
            db_file = os.path.join(chroma_path, "chroma.sqlite3")
            
            if os.path.exists(chroma_path) and os.path.exists(db_file):
                # Try to load existing vector store
                self.vector_store = Chroma(
                    persist_directory=VECTOR_DB_PATH,
                    embedding_function=self.embeddings,
                    collection_name="documents",
                )
                # Verify the collection has documents
                try:
                    count = self.vector_store._collection.count()
                    self.is_available = count > 0
                except Exception:
                    self.is_available = False
            else:
                self.is_available = False
        except Exception as e:
            # If ChromaDB fails to initialize, mark as unavailable
            print(f"Warning: ChromaDB not available: {e}")
            self.vector_store = None
            self.is_available = False
        
        self.filtered_ids = filtered_ids
    


    def _format_results(self, results: list) -> list:
        """Format raw results (doc, score) as dicts."""
        similarity = lambda l2: 1 - (l2 ** 2) / 4  # L2 to cosine similarity
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
        # Return empty results if vector store is not available
        if not self.is_available or self.vector_store is None:
            return []
        
        try:
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
        except Exception as e:
            # Return empty list on any error during search
            print(f"Search error: {e}")
            return []


# Step 2: Implementation of query decompostion
def decompose_query(query, model=LLM_MODEL):
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
    Default to News Article if no company or annual report is specified.
    """
    # Normalize possessives: "Ford's" -> "Ford"
    query_clean = re.sub(r"'s\b", "", single_shot_query)

    # Extract year (4 consecutive digits)
    year_match = re.search(r"(20\d{2})", query_clean)
    year = year_match.group(1) if year_match else None

    # Extract company from a known list
    companies = ["BMW", "Tesla", "Ford"]
    company = next((c for c in companies if re.search(rf'\b{re.escape(c)}\b', query_clean, re.IGNORECASE)), None)

    # Extract document type (case-insensitive, allow both 'News Article' and 'Annual Report')
    doc_types = ["annual report", "news article"]
    document_type = next((d.title() for d in doc_types if d in query_clean.lower()), None)
    
    # Check for financial keywords that indicate annual report
    financial_keywords = ["revenue", "profit", "income", "financial", "earnings", "balance sheet", "cash flow", "fiscal", "quarterly", "annual"]
    has_financial_keyword = any(kw in query_clean.lower() for kw in financial_keywords)
    
    # If no company found AND no annual report specified, default to News Article
    if company is None and document_type != "Annual Report":
        document_type = "News Article"
    # If company is found with financial keyword, prefer Annual Report
    elif company is not None and has_financial_keyword:
        document_type = "Annual Report"

    return {
        "company": company,
        "document_type": document_type,
        "year": year
    }
   

# Step 4: Retrieve aggregated context based on decomposed queries
def retrieve_aggregated_context(query, retriever, document_filter=None):
    aggregated_context = ""
    decomposed_queries = decompose_query(query)
    for single_query in decomposed_queries:
        metadata_for_query = extract_metadata_from_query(single_query)
        # Combine automatic extraction with manual document_filter
        if document_filter:
            if metadata_for_query:
                metadata_for_query = {**metadata_for_query, **document_filter}
            else:
                metadata_for_query = document_filter
        results = retriever.search(single_query, k=3, metadata_filter=metadata_for_query)
        for res in results:
            # import pdb; pdb.set_trace()
            aggregated_context += f"Document: {res['document']}\nMetadata: {res['metadata']}\n\n"
    return aggregated_context
