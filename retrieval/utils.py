import re

def extract_metadata_from_query(query: str) -> dict:
    """
    Extracts metadata keys from the query string using simple rules.
    For production, replace with robust NER or ML-based extraction.
    """
    metadata = {}

    # Company extraction
    for company in ["Tesla", "BMW", "Ford"]:
        if company.lower() in query.lower():
            metadata["company"] = company

    # Year extraction
    year_match = re.search(r"(20\d{2})", query)
    if year_match:
        metadata["year"] = int(year_match.group(1))

    # Document type extraction
    if "annual report" in query.lower():
        metadata["document_type"] = "Annual Report"
    elif "news" in query.lower():
        metadata["document_type"] = "News Article"

    return metadata
"""
Utility functions for retrieval logic.
"""

from typing import Dict, Any, List

METADATA_KEYS = ["file_name", "document_type", "company", "year", "page_number"]

def build_chroma_metadata_filter(metadata: Dict[str, Any], keys: List[str] = METADATA_KEYS) -> Dict:
    """
    Build a Chroma-compatible metadata filter dict from metadata.
    Returns a filter in the format expected by Chroma ($and operator).
    """
    filter_items = [
        {k: v} for k, v in metadata.items() if k in keys and v is not None
    ]
    if not filter_items:
        raise ValueError(f"At least one valid metadata key ({keys}) with a non-None value is required.")
    return {"$and": filter_items} if len(filter_items) > 1 else filter_items[0]
