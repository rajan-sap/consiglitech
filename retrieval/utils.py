import os
import re

from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, List


load_dotenv()


import re

def extract_metadata_from_query(query):
    """
    Extract only company, document_type, and year for metadata filtering.
    Handles both News Article and Annual Report cases.
    """
    # Normalize possessives: "Ford's" -> "Ford"
    query_clean = re.sub(r"'s\\b", "", query)

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



api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment.")
client = OpenAI(api_key=api_key)


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