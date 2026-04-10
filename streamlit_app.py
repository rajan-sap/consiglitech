import streamlit as st
import os
import re
from generation.generator import generate_answer, generate_answer_for_uploads
from ingestion.upload_processor import process_uploaded_file
from retrieval.session_retriever import SessionRetriever
from llm_config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocIntel — Document Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# API KEY VALIDATION
# ─────────────────────────────────────────────
if not LLM_API_KEY:
    st.warning(
        "⚠️ **LLM API key not found.** The app cannot generate answers.\n\n"
        "If you're running locally, create `.streamlit/secrets.toml` with:\n"
        "```\nGEMINI_API_KEY = \"your_key_here\"\n```\n\n"
        "On Streamlit Cloud, go to **Settings → Secrets** and add the key there.",
        icon="🔑",
    )

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; font-size: 1.08rem; }

/* ── Outer page background (visible outside the app) ── */
html, body { background: #090b0e !important; }

/* ── Main area ── */
.stApp { background: #0f1117; color: #c9cdd4; }
.stApp > header { background: transparent !important; }

/* ── Constrain entire app to 70% on desktop ── */
@media (min-width: 1024px) {
    .stApp {
        max-width: 70vw !important;
        margin: 0 auto !important;
        border-left: 1px solid #1e222b;
        border-right: 1px solid #1e222b;
    }
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161a22;
    border-right: 1px solid #1e222b;
    height: 100vh !important;
    min-height: 100vh !important;
    top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    display: flex; flex-direction: column;
    height: 100vh; min-height: 100vh;
    padding-top: 1rem; padding-bottom: 1rem;
    overflow-y: auto; gap: 0.3rem;
}
section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar { width: 4px; }
section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.08); border-radius: 2px;
}
section[data-testid="stSidebar"] * { color: #8b919a !important; font-size: 0.88rem; }
section[data-testid="stSidebar"] { min-width: 270px !important; max-width: 300px !important; }
section[data-testid="stSidebar"] hr { border-color: #1e222b; margin: 0.5rem 0; }

/* ── Chat bubbles ── */
.chat-row { display: flex; margin: 0.6rem 0; align-items: flex-start; gap: 0.6rem; }
.chat-row.user { flex-direction: row-reverse; }
.chat-avatar {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 0.8rem; font-weight: 600;
}
.chat-avatar.user-av { background: #2a2d35; color: #8b919a; }
.chat-avatar.bot-av  { background: #1c2028; color: #6b7280; }
.chat-bubble {
    max-width: 70%; padding: 0.85rem 1.1rem; border-radius: 0.7rem;
    line-height: 1.6; font-size: 0.95rem;
}
.chat-bubble.user-bubble {
    background: #1a2332; color: #bcc5d0;
    border: 1px solid #232d3b;
}
.chat-bubble.bot-bubble {
    background: #181c24; color: #a0a7b0;
    border: 1px solid #1e222b;
}

/* ── Input area ── */
.stChatInput > div { background: #161a22 !important; }
.stChatInput textarea {
    background: #161a22 !important; color: #c9cdd4 !important;
    border-color: #1e222b !important;
}
.stChatInput textarea::placeholder { opacity: 0.35 !important; }

/* ── Stat cards ── */
.stat-card {
    background: #161a22; border: 1px solid #1e222b;
    border-radius: 0.4rem; padding: 0.4rem 0.5rem; text-align: center;
}
.stat-card .stat-value { font-size: 1.3rem; font-weight: 600; color: #c9cdd4 !important; }
.stat-card .stat-label {
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: #555b65 !important;
}

/* ── Welcome card ── */
.welcome-card {
    background: #161a22; border: 1px solid #1e222b;
    border-radius: 0.75rem; padding: 2.5rem 2rem; text-align: center;
    margin: 2rem auto; max-width: 600px;
}
.welcome-card h2 {
    font-weight: 600; margin-bottom: 0.4rem; font-size: 1.5rem;
    color: #d8dce2 !important; letter-spacing: -0.02em;
}
.welcome-card p { color: #6b7280 !important; font-size: 0.95rem; line-height: 1.6; }

/* ── Example questions ── */
.example-label { text-align: center; color: #555b65; font-size: 0.85rem; margin-bottom: 0.5rem; }
div[data-testid="stHorizontalBlock"] button {
    background: #161a22 !important; color: #8b919a !important;
    border: 1px solid #1e222b !important;
    border-radius: 0.5rem !important; font-size: 0.88rem !important;
    padding: 0.5rem 0.8rem !important; transition: border-color 0.15s !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
    border-color: #333a45 !important; color: #c9cdd4 !important;
}

/* ── Sidebar buttons ── */
section[data-testid="stSidebar"] button {
    background: #1a1e26 !important; border: 1px solid #1e222b !important;
    color: #6b7280 !important; border-radius: 0.4rem !important;
}
section[data-testid="stSidebar"] button:hover {
    border-color: #333a45 !important; color: #a0a7b0 !important;
}

/* ── Upload area ── */
.upload-card {
    background: #161a22; border: 1px dashed #2a2f3a;
    border-radius: 0.75rem; padding: 2rem; text-align: center;
    margin: 1.5rem auto; max-width: 600px;
}
.upload-card h3 { font-weight: 600; color: #d8dce2 !important; margin-bottom: 0.3rem; }
.upload-card p { color: #555b65 !important; font-size: 0.9rem; }

/* ── File chip ── */
.file-chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: #1a1e26; border: 1px solid #252a34;
    border-radius: 0.35rem; padding: 0.25rem 0.6rem; margin: 0.15rem;
    font-size: 0.8rem; color: #8b919a !important;
}

/* ── Sidebar section titles ── */
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5 {
    font-size: 0.78rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
    color: #555b65 !important;
    margin-top: 0.5rem !important; margin-bottom: 0.3rem !important;
}

/* ── Spinner ── */
.stSpinner > div { color: #555b65 !important; }

/* ── Hide chrome ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "upload_messages" not in st.session_state:
    st.session_state.upload_messages = []
if "upload_pending_query" not in st.session_state:
    st.session_state.upload_pending_query = None
if "user_retriever" not in st.session_state:
    st.session_state.user_retriever = None
if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = {}  # {filename: chunk_count}
import time

MAX_FREE_QUERIES = 10
LIMIT_WINDOW_SECONDS = 3600  # 1 hour

if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "limit_hit_time" not in st.session_state:
    st.session_state.limit_hit_time = None

# Reset counter if 1 hour has passed since hitting the limit
if (
    st.session_state.limit_hit_time is not None
    and time.time() - st.session_state.limit_hit_time >= LIMIT_WINDOW_SECONDS
):
    st.session_state.query_count = 0
    st.session_state.limit_hit_time = None

# ─────────────────────────────────────────────
# HELPER: count ingested docs
# ─────────────────────────────────────────────
def count_data_files():
    DEFAULT_PER_COMPANY = {"BMW": 3, "Ford": 3, "Tesla": 2}
    DEFAULT_NEWS = 1
    DEFAULT_TOTAL = sum(DEFAULT_PER_COMPANY.values()) + DEFAULT_NEWS

    if not os.path.isdir("./data"):
        return DEFAULT_TOTAL, DEFAULT_PER_COMPANY, DEFAULT_NEWS

    companies = ["BMW", "Ford", "Tesla"]
    total = 0
    per_company = {}
    for c in companies:
        folder = os.path.join("./data", c)
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            per_company[c] = len(files)
            total += len(files)
        else:
            per_company[c] = 0
    news_folder = os.path.join("./data", "News")
    news_count = 0
    if os.path.isdir(news_folder):
        news_count = len([f for f in os.listdir(news_folder) if os.path.isfile(os.path.join(news_folder, f))])
        total += news_count
    root_files = [f for f in os.listdir("./data") if os.path.isfile(os.path.join("./data", f))]
    total += len(root_files)
    return total, per_company, news_count

total_docs, docs_per_company, news_docs = count_data_files()

def count_pages():
    """Count unique pages in the ChromaDB collection."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        col = client.get_collection("documents")
        results = col.get(include=["metadatas"])
        pages = set()
        for m in results["metadatas"]:
            pages.add((m.get("file_name", ""), m.get("page_number", "")))
        return len(pages)
    except Exception:
        return 2000  # fallback estimate

total_pages = count_pages()

def count_chunks():
    """Count total chunks in the ChromaDB collection."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        col = client.get_collection("documents")
        return col.count()
    except Exception:
        return 3001

total_chunks = count_chunks()

# ─────────────────────────────────────────────
# HELPER: extract metadata from filename
# ─────────────────────────────────────────────
def extract_metadata_from_filename(filename):
    name = filename.replace('.pdf', '')
    company = name.split('_')[0] if '_' in name else None
    year_match = re.search(r'(20\d{2})', name)
    year = year_match.group(1) if year_match else None
    return {"company": company, "year": year, "document_type": "Annual Report"}

# ─────────────────────────────────────────────
# HELPER: render chat messages
# ─────────────────────────────────────────────
def render_chat(messages):
    for msg in messages:
        if msg["role"] == "user":
            avatar_class, bubble_class, row_class, avatar_text = "user-av", "user-bubble", "user", "You"
        else:
            avatar_class, bubble_class, row_class, avatar_text = "bot-av", "bot-bubble", "", "DI"
        st.markdown(f"""
        <div class="chat-row {row_class}">
            <div class="chat-avatar {avatar_class}">{avatar_text}</div>
            <div class="chat-bubble {bubble_class}">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.3rem 0 0.1rem 0;">
        <div style="font-size:1.05rem; font-weight:700; letter-spacing:-0.02em; color:#c9cdd4 !important;">
            DocIntel
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Default Knowledge Base stats ──
    st.markdown("##### Knowledge Base")
    row1 = st.columns(2)
    with row1[0]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_docs}</div>
            <div class="stat-label">Documents</div>
        </div>""", unsafe_allow_html=True)
    with row1[1]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(docs_per_company)}</div>
            <div class="stat-label">Companies</div>
        </div>""", unsafe_allow_html=True)
    row2 = st.columns(2)
    with row2[0]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_pages}</div>
            <div class="stat-label">Pages</div>
        </div>""", unsafe_allow_html=True)
    with row2[1]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_chunks}</div>
            <div class="stat-label">Chunks</div>
        </div>""", unsafe_allow_html=True)

    for company, count in docs_per_company.items():
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:0.3rem 0.5rem; margin:0.1rem 0; border-radius:0.35rem;
                    background:#1a1e26; font-size:0.84rem;">
            <span>{company}</span>
            <span style="color:#6b7280 !important; font-size:0.8rem; font-weight:500;">{count}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:0.3rem 0.5rem; margin:0.1rem 0; border-radius:0.35rem;
                    background:#1a1e26; font-size:0.84rem;">
            <span>News & Ads</span>
            <span style="color:#6b7280 !important; font-size:0.8rem; font-weight:500;">{news_docs}</span>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Sample Document ──
    st.markdown("##### Sample Document")
    tesla_folder = "./data/Tesla"
    if os.path.isdir(tesla_folder):
        tesla_pdfs = sorted([f for f in os.listdir(tesla_folder) if f.endswith('.pdf')])
        if tesla_pdfs:
            pdf_file = tesla_pdfs[0]
            pdf_path = os.path.join(tesla_folder, pdf_file)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label=pdf_file,
                    data=f,
                    file_name=pdf_file,
                    mime="application/pdf",
                    use_container_width=True,
                )
            st.markdown(
                '<div style="font-size:0.75rem; color:#555b65 !important; margin-top:0.2rem;">'
                'Download to see the complexity of source documents.</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="font-size:0.78rem; color:#555b65 !important;">'
            'Tesla_Annual_Report_2022.pdf<br>'
            '<span style="font-size:0.72rem;">Sample not available on cloud. Clone the repo to access.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── User Uploads stats ──
    if st.session_state.processed_uploads:
        st.markdown("##### Your Uploads")
        upload_count = len(st.session_state.processed_uploads)
        chunk_count = sum(st.session_state.processed_uploads.values())
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{upload_count}</div>
                <div class="stat-label">Files</div>
            </div>""", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{chunk_count}</div>
                <div class="stat-label">Chunks</div>
            </div>""", unsafe_allow_html=True)

        for fname in st.session_state.processed_uploads:
            st.markdown(f'<div class="file-chip">{fname}</div>', unsafe_allow_html=True)

        st.divider()

    # ── Tech Stack ──
    st.markdown("##### Stack")
    if LLM_BASE_URL and "groq" in LLM_BASE_URL:
        _llm_provider = "Groq"
    elif LLM_BASE_URL and "localhost" in LLM_BASE_URL:
        _llm_provider = "LM Studio"
    elif LLM_BASE_URL and "google" in LLM_BASE_URL:
        _llm_provider = "Google"
    else:
        _llm_provider = "OpenAI"
    _llm_display = f"{_llm_provider} / {LLM_MODEL}"

    st.markdown(f"""
    <div style="font-size:0.8rem; color:#555b65 !important; line-height:1.55;">
        Embeddings: BGE-base-en v1.5<br>
        LLM: {_llm_display}<br>
        Vector DB: ChromaDB<br>
        Retrieval: Hybrid
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Actions ──
    st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.upload_messages = []
        st.rerun()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown(
    '<p style="font-size:1.15rem; color:#555b65; margin:0 0 0.5rem 0;">'
    'Advanced RAG system to talk with your text and tabular data</p>',
    unsafe_allow_html=True,
)

tab_kb, tab_upload = st.tabs(["Knowledge Base", "Your Documents"])

# ═════════════════════════════════════════════
# TAB 1: Knowledge Base (existing functionality)
# ═════════════════════════════════════════════
with tab_kb:
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <h2>DocIntel</h2>
            <p>Ask questions about BMW, Ford, and Tesla annual reports.<br>
            Answers are grounded in real documents using hybrid retrieval.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("<p class='example-label'>Ask questions like these</p>", unsafe_allow_html=True)

        example_questions = [
            "What was Tesla's revenue in 2022 and 2023?",
            "Which Tesla models were in development phase in 2022?",
            "Compare BMW and Ford's net income over the past 3 years",
        ]
        cols = st.columns(3)
        for i, q in enumerate(example_questions):
            with cols[i % 3]:
                if st.button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.session_state.pending_query = q
                    st.rerun()
    else:
        render_chat(st.session_state.messages)

    if prompt := st.chat_input("Ask about BMW, Ford, or Tesla...", key="kb_chat"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_query = prompt
        st.rerun()

    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        if st.session_state.query_count >= MAX_FREE_QUERIES:
            answer = "You've reached the free chat limit (10 queries per hour). This is a beta version with limited resources. Please try again later."
        else:
            try:
                with st.spinner("Searching documents..."):
                    answer = generate_answer(query)
                st.session_state.query_count += 1
                if st.session_state.query_count >= MAX_FREE_QUERIES:
                    st.session_state.limit_hit_time = time.time()
            except Exception as e:
                error_type = type(e).__name__
                answer = f"⚠️ **Error generating answer:** {error_type} — {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# ═════════════════════════════════════════════
# TAB 2: Your Documents (upload + chat)
# ═════════════════════════════════════════════
with tab_upload:
    # ── Upload area ──
    if not st.session_state.processed_uploads:
        st.markdown("""
        <div class="upload-card">
            <h3>Upload your documents</h3>
            <p>Upload PDF or DOCX files and chat with them.<br>
            Documents are processed in your session and not stored permanently.</p>
        </div>
        """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        # Find files not yet processed
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_uploads]

        if new_files:
            if st.button(f"Process {len(new_files)} new file(s)", use_container_width=True):
                # Initialize retriever if needed
                if st.session_state.user_retriever is None:
                    st.session_state.user_retriever = SessionRetriever()

                progress = st.progress(0, text="Processing...")
                for idx, file in enumerate(new_files):
                    progress.progress(
                        (idx) / len(new_files),
                        text=f"Processing {file.name}...",
                    )
                    try:
                        chunks = process_uploaded_file(file.name, file.getvalue())
                        if chunks:
                            st.session_state.user_retriever.add_documents(chunks, file.name)
                            st.session_state.processed_uploads[file.name] = len(chunks)
                        else:
                            st.warning(f"No text extracted from **{file.name}** — it may be a scanned/image PDF.")
                    except Exception as e:
                        st.error(f"Failed to process **{file.name}**: {e}")

                progress.progress(1.0, text="Done!")
                st.rerun()

    # ── Show processed files ──
    if st.session_state.processed_uploads:
        st.markdown("---")
        cols = st.columns([3, 1])
        with cols[0]:
            file_chips = " ".join(
                f'<span class="file-chip">{name} ({chunks} chunks)</span>'
                for name, chunks in st.session_state.processed_uploads.items()
            )
            st.markdown(file_chips, unsafe_allow_html=True)
        with cols[1]:
            if st.button("Clear uploads", use_container_width=True):
                st.session_state.user_retriever = None
                st.session_state.processed_uploads = {}
                st.session_state.upload_messages = []
                st.rerun()

        st.markdown("")

        # ── Chat with uploaded documents ──
        if st.session_state.upload_messages:
            render_chat(st.session_state.upload_messages)

        if upload_prompt := st.chat_input("Ask about your uploaded documents...", key="upload_chat"):
            st.session_state.upload_messages.append({"role": "user", "content": upload_prompt})
            st.session_state.upload_pending_query = upload_prompt
            st.rerun()

        if st.session_state.upload_pending_query:
            query = st.session_state.upload_pending_query
            st.session_state.upload_pending_query = None
            if st.session_state.query_count >= MAX_FREE_QUERIES:
                answer = "You've reached the free chat limit (10 queries). This is a beta version with limited resources. Thank you for trying DocIntel!"
            else:
                try:
                    with st.spinner("Searching your documents..."):
                        answer = generate_answer_for_uploads(query, st.session_state.user_retriever)
                    st.session_state.query_count += 1
                except Exception as e:
                    error_type = type(e).__name__
                    answer = f"⚠️ **Error generating answer:** {error_type} — {e}"
            st.session_state.upload_messages.append({"role": "assistant", "content": answer})
            st.rerun()
    elif not uploaded_files:
        st.markdown(
            "<p style='text-align:center; opacity:0.4; margin-top:1rem;'>"
            "Upload documents above to get started.</p>",
            unsafe_allow_html=True,
        )
