
import re

def extract_metadata_from_query(query):
    """
    Extract only company, document_type, and year for metadata filtering.
    Handles both News Article and Annual Report cases.
    """
    # Normalize possessives: "Ford's" -> "Ford"
    query_clean = re.sub(r"'s\\b", "", query)

    # Extract year (4 consecutive digits)
    year_match = re.search(r"(20\\d{2})", query)
    year = year_match.group(1) if year_match else None

    # Extract company from a known list
    companies = ["BMW", "Tesla", "Ford"]
    company = next((c for c in companies if re.search(rf'\\b{re.escape(c)}\\b', query_clean, re.IGNORECASE)), None)

    # Extract document type (case-insensitive, allow both 'News Article' and 'Annual Report')
    doc_types = ["annual report", "news article"]
    document_type = next((d.title() for d in doc_types if d in query_clean.lower()), None)

    return {
        "company": company,
        "document_type": document_type,
        "year": year
    }