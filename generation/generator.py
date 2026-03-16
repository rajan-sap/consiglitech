
import re
from retrieval.retriever import retrieve_aggregated_context
from retrieval.retriever import Retriever
from llm_config import llm_client as client, LLM_MODEL, truncate_context


retiever = Retriever()


# ─── Query Intent Classification (LLM-based) ────────────────────────────────

CLASSIFY_SYSTEM_PROMPT = (
    "You are an intent classifier for a document-intelligence assistant called DocIntel. "
    "DocIntel has access to annual reports from BMW, Ford, and Tesla.\n\n"
    "Classify the user's query into exactly ONE of these categories:\n"
    "  DOCUMENT — the query asks about company data, financials, reports, risks, strategy, "
    "or anything that would require looking up information from the annual reports.\n"
    "  GENERAL  — the query is a greeting, small talk, a question about the assistant itself, "
    "general knowledge, or anything that does NOT need document retrieval.\n\n"
    "Reply with a SINGLE word: DOCUMENT or GENERAL. Nothing else."
)


def is_general_query(query: str) -> bool:
    """Use the LLM to classify whether a query needs document retrieval.
    Returns True for general/meta queries, False for document queries."""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        label = response.choices[0].message.content.strip().upper()
        return "GENERAL" in label
    except Exception:
        # On any failure, fall back to document route (safer default)
        return False


# ─── System Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_RAG = (
    "You are DocIntel, a document-intelligence assistant. "
    "Answer the user's question using ONLY the retrieved document context below. "
    "If the context does not contain enough information to answer, say so clearly — "
    "do NOT fabricate facts. Cite the company name, year, or document type when relevant."
)

SYSTEM_PROMPT_GENERAL = (
    "You are DocIntel, a helpful document-intelligence assistant built on a RAG "
    "(Retrieval-Augmented Generation) pipeline. You analyse annual reports from "
    "BMW, Ford and Tesla. When the user asks a general or meta question (e.g. about "
    "yourself, your capabilities, or general knowledge), answer directly and concisely "
    "from your own knowledge. Do NOT reference any documents or reports unless asked."
)


# ─── Answer Generation ───────────────────────────────────────────────────────

# Step 4: Implementation of answer generation
def generate_answer(query, return_details=False, document_filter=None):
    """
    Generate an answer using the RAG pipeline (retrieve + generate).
    Automatically classifies the query: general/meta questions are answered
    directly without document retrieval; document-related queries go through
    the full RAG pipeline.

    Args:
        query: The user's question.
        return_details: If True, return a dict with answer, context, and raw response.
                        If False (default), return just the answer string.
        document_filter: Optional metadata filter to restrict search to specific documents.

    Returns:
        str (default) or dict with 'answer', 'context', 'response' keys.
    """

    # ── Route 1: General / meta questions — skip retrieval ──────────────
    if is_general_query(query):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_GENERAL},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()

        if return_details:
            return {
                "answer": answer,
                "context": "",
                "response": response,
            }
        return answer

    # ── Route 2: Document questions — full RAG pipeline ─────────────────
    aggregated_context = retrieve_aggregated_context(query, retiever, document_filter)
    aggregated_context = truncate_context(aggregated_context)

    prompt = (
        f"Answer the following question using the retrieved context.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved Context:\n{aggregated_context}"
    )
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    answer = response.choices[0].message.content.strip()

    if return_details:
        return {
            "answer": answer,
            "context": aggregated_context,
            "response": response,
        }
    return answer

