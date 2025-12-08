"""
RAG Chatbot - Query your documents using natural language.
"""

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.constants import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH


# =============================================================================
# CACHED RESOURCES (loaded once, kept in memory)
# =============================================================================


@st.cache_resource
def get_vector_store():
    """Load embedding model and vector store (cached - loads once)."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name="documents",
    )
    return vector_store, embeddings


def l2_to_cosine_similarity(l2_distance: float) -> float:
    """Convert L2 distance to cosine similarity for normalized vectors."""
    return 1 - (l2_distance ** 2) / 2


def get_rag_answer(query: str, k: int = 3) -> str:
    """
    Retrieve relevant chunks and format answer with sources.
    """
    vector_store, _ = get_vector_store()
    
    results = vector_store.similarity_search_with_score(query, k=k)
    
    if not results:
        return "No relevant information found in the documents."
    
    # Build response with retrieved context
    response_parts = []
    sources = []
    
    for i, (doc, l2_distance) in enumerate(results, 1):
        similarity = l2_to_cosine_similarity(l2_distance)
        content = doc.page_content.strip()
        page = doc.metadata.get("page_number", "N/A")
        file_name = doc.metadata.get("file_name", "Unknown")
        # Exclude text chunk from metadata display and format in specified sequence
        keys = ["file_name", "document_type", "company", "year", "page_number"]
        metadata_seq = {k: doc.metadata.get(k, None) for k in keys}
        # Only include if similarity is reasonable
        if similarity > 0.5:
            formatted_meta = "\n".join([f"{k}: {metadata_seq[k]}" for k in keys])
            response_parts.append(
                f"**[{i}]** {content[:500]}{'...' if len(content) > 500 else ''}"
                f"\n\n**Metadata:**\n{formatted_meta}"
            )
            sources.append(f"Page {page} from {file_name} (similarity: {similarity:.2f})")
    
    if not response_parts:
        return "No sufficiently relevant information found."
    
    answer = "\n\n".join(response_parts)
    source_info = "\n".join([f"- {s}" for s in sources])
    
    return f"{answer}\n\n---\n**Sources:**\n{source_info}"


# =============================================================================
# UI
# =============================================================================

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Document QA")
st.markdown("Ask questions about your ingested documents.")
st.session_state.chat_history = []

# Display chat history
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# Chat input
if user_query := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    # Get and display response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer = get_rag_answer(user_query)
        st.markdown(answer)
    
    # Save to history
    st.session_state.chat_history.append((user_query, answer))