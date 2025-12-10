import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingestion.constants import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH

# =====================
# Responsive & Modern UI
# =====================


st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide",
)



# =====================
# CACHED RESOURCES
# =====================
@st.cache_resource
def get_vector_store():
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
    return 1 - (l2_distance ** 2) / 2

def get_rag_answer(query: str, k: int = 3) -> str:
    vector_store, _ = get_vector_store()
    results = vector_store.similarity_search_with_score(query, k=k)
    if not results:
        return "<span style='color:#b91c1c;'>No relevant information found in the documents.</span>"
    response_parts = []
    sources = []
    for i, (doc, l2_distance) in enumerate(results, 1):
        similarity = l2_to_cosine_similarity(l2_distance)
        content = doc.page_content.strip()
        page = doc.metadata.get("page_number", "N/A")
        file_name = doc.metadata.get("file_name", "Unknown")
        keys = ["file_name", "document_type", "company", "year", "page_number"]
        metadata_seq = {k: doc.metadata.get(k, None) for k in keys}
        if similarity > 0.5:
            formatted_meta = "<ul style='margin:0;padding-left:1.2em;'>" + "".join([f"<li style='color:#e5e7eb;background:#23272f;padding:2px 8px;border-radius:6px;margin-bottom:2px;'><b>{k}</b>: {metadata_seq[k]}</li>" for k in keys]) + "</ul>"
            response_parts.append(
                f"<div style='background:#23272f;border-radius:10px;padding:1rem 1.5rem;margin-bottom:1.2rem;box-shadow:0 1px 6px rgba(0,0,0,0.10);color:#e5e7eb;'>"
                f"<span style='font-weight:600;color:#60a5fa;'>[{i}]</span> <span style='color:#e5e7eb;'>{content[:500]}{'...' if len(content) > 500 else ''}</span>"
                f"<div style='margin-top:0.5rem;font-size:0.95em;color:#a1a1aa;'><b>Metadata:</b> {formatted_meta}</div>"
                f"</div>"
            )
            sources.append(f"<span style='color:#60a5fa;'>Page {page} from {file_name} (similarity: {similarity:.2f})</span>")
    if not response_parts:
        return "<span style='color:#b91c1c;'>No sufficiently relevant information found.</span>"
    answer = "".join(response_parts)
    source_info = "<ul style='margin:0;padding-left:1.2em;'>" + "".join([f"<li style='color:#60a5fa;background:#23272f;padding:2px 8px;border-radius:6px;margin-bottom:2px;'>{s}</li>" for s in sources]) + "</ul>"
    return f"{answer}<hr style='margin:2rem 0 1rem 0;border:none;border-top:1px solid #333;' /><div style='font-size:1.05em;color:#e5e7eb;'><b>Sources:</b> {source_info}</div>"


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

# Place chat input inside the card
st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
user_query = st.chat_input("Ask a question about your documents...")
if user_query:
    with st.spinner("Searching documents..."):
        answer = get_rag_answer(user_query)
    st.markdown(
        f"<div class='signup-card' style='margin-top:2vh;'>"
        f"<div style='width:100%;text-align:left;'><b>You asked:</b><br><span style='color:#222;font-size:1.1em;'>{user_query}</span></div>"
        f"<div style='margin-top:1.2em;width:100%;'>{answer}</div>"
        f"</div>",
        unsafe_allow_html=True
    )