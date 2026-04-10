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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; font-size: 1.05rem; }

/* ── Main area: warm dark canvas ── */
.stApp {
    background: #121518;
    color: #d4d4d8;
}
.stApp > header { background: transparent !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #191d23;
    border-right: 1px solid rgba(255,255,255,0.06);
    height: 100vh !important;
    min-height: 100vh !important;
    top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    display: flex; flex-direction: column; height: 100vh;
    min-height: 100vh;
    padding-top: 1rem; padding-bottom: 1rem;
    justify-content: flex-start;
    overflow-y: auto;
    overflow-x: hidden;
    gap: 0.35rem;
}
section[data-testid="stSidebar"] > div:first-child > div {
    flex-shrink: 0;
}

/* Sidebar scrollbar styling */
section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
    width: 5px;
}
section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track {
    background: transparent;
}
section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.12); border-radius: 3px;
}
section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb:hover {
    background: rgba(255,255,255,0.22);
}
section[data-testid="stSidebar"] * {
    color: #b8bcc4 !important;
    font-size: 0.92rem;
}
section[data-testid="stSidebar"] { min-width: 280px !important; max-width: 310px !important; }
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.06);
    margin: 0.4rem 0;
}

/* ── Chat bubbles ── */
.chat-row { display: flex; margin: 0.75rem 0; align-items: flex-start; gap: 0.7rem; }
.chat-row.user { flex-direction: row-reverse; }

.chat-avatar {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    background: transparent;
}
.chat-avatar.user-av { font-size: 2rem; color: #93c5fd; }
.chat-avatar.bot-av  { font-size: 2.5rem; color: #34d399; }

.chat-bubble {
    max-width: 72%; padding: 1rem 1.25rem; border-radius: 0.85rem;
    line-height: 1.65; font-size: 1.02rem;
}
.chat-bubble.user-bubble {
    background: #1e3a5f; color: #e0ecff; border-bottom-right-radius: 0.2rem;
    border: 1px solid rgba(37,99,235,0.2);
}
.chat-bubble.bot-bubble {
    background: #1e2329; color: #c8cdd5; border-bottom-left-radius: 0.2rem;
    border: 1px solid rgba(255,255,255,0.06);
}

/* ── Input area ── */
.stChatInput {
    border-top: 1px solid rgba(255,255,255,0.06);
}
.stChatInput > div { background: #191d23 !important; }
.stChatInput textarea {
    background: #191d23 !important; color: #d4d4d8 !important;
    border-color: rgba(255,255,255,0.08) !important;
}
.stChatInput textarea::placeholder {
    font-size: 0.88rem !important;
    opacity: 0.4 !important;
    font-style: italic !important;
}

/* ── Stat cards ── */
.stat-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 0.5rem; padding: 0.35rem 0.5rem; margin-bottom: 0.2rem;
    text-align: center;
}
.stat-card .stat-value { font-size: 1.5rem; font-weight: 700; color: #60a5fa !important; }
.stat-card .stat-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.55; }

/* ── Welcome card ── */
.welcome-card {
    background: #191d23;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 1rem; padding: 2.8rem 2rem; text-align: center;
    color: #d4d4d8; margin: 2.5rem auto; max-width: 640px;
}
.welcome-card h2 { font-weight: 700; margin-bottom: 0.5rem; font-size: 1.75rem; color: #f0f0f2; }
.welcome-card p  { opacity: 0.65; font-size: 1.05rem; margin: 0; line-height: 1.6; }

/* ── Example questions ── */
.example-label { text-align:center; color:#71757e; font-size:0.95rem; margin-bottom:0.5rem; }

/* Style Streamlit buttons in example area */
div[data-testid="stHorizontalBlock"] button {
    background: #191d23 !important; color: #a0a4ac !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 0.6rem !important; font-size: 0.95rem !important;
    padding: 0.55rem 0.9rem !important; transition: all 0.2s !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
    background: #1e2329 !important; color: #e0e0e4 !important;
    border-color: rgba(96,165,250,0.35) !important;
}

/* ── Pipeline badge ── */
.pipeline-step {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.04); border-radius: 0.5rem;
    padding: 0.45rem 0.75rem; margin: 0.25rem; font-size: 0.9rem;
    border: 1px solid rgba(255,255,255,0.04);
}
.pipeline-step .step-icon { font-size: 1.15rem; }

/* ── Scrollable chat container ── */
.chat-container {
    max-height: 62vh; overflow-y: auto; padding: 0.5rem 0.5rem 1rem 0.5rem;
    scroll-behavior: smooth;
}

/* ── Spinner text ── */
.stSpinner > div { color: #8b8f97 !important; }

/* ── Clear button ── */
section[data-testid="stSidebar"] button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #a0a4ac !important; border-radius: 0.5rem !important;
}
section[data-testid="stSidebar"] button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #e0e0e4 !important;
}

/* ── Upload area styling ── */
.upload-card {
    background: #191d23;
    border: 2px dashed rgba(96,165,250,0.25);
    border-radius: 1rem; padding: 2rem; text-align: center;
    color: #d4d4d8; margin: 1.5rem auto; max-width: 640px;
}
.upload-card h3 { font-weight: 600; margin-bottom: 0.4rem; color: #f0f0f2; }
.upload-card p  { opacity: 0.55; font-size: 0.95rem; }

/* ── File chip ── */
.file-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(96,165,250,0.08); border: 1px solid rgba(96,165,250,0.18);
    border-radius: 0.5rem; padding: 0.3rem 0.7rem; margin: 0.2rem;
    font-size: 0.85rem; color: #93c5fd !important;
}

/* ── Sidebar section titles ── */
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.3rem !important;
}

/* ── Hide default Streamlit elements ── */
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
            avatar_class, bubble_class, row_class, avatar_text = "user-av", "user-bubble", "user", "👤"
        else:
            avatar_class, bubble_class, row_class, avatar_text = "bot-av", "bot-bubble", "", "🤖"
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
    <div style="text-align:center; padding: 0 0 0.2rem 0;">
        <div style="font-size:1.8rem;">📑</div>
        <div style="font-size:1.15rem; font-weight:700; letter-spacing:-0.02em; margin-top:0.1rem; color:#e8e8ec !important;">
            DocIntel
        </div>
        <div style="font-size:0.75rem; opacity:0.45; text-transform:uppercase; letter-spacing:0.1em;">
            Document Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Default Knowledge Base stats ──
    st.markdown("##### 📊 Knowledge Base")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_docs}</div>
            <div class="stat-label">Documents</div>
        </div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(docs_per_company)}</div>
            <div class="stat-label">Companies</div>
        </div>""", unsafe_allow_html=True)

    for company, count in docs_per_company.items():
        icon = {"BMW": "🚗", "Ford": "🚙", "Tesla": "⚡"}.get(company, "📁")
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:0.25rem 0.5rem; margin:0.1rem 0; border-radius:0.4rem;
                    background:rgba(255,255,255,0.04); font-size:0.88rem;">
            <span style="font-size:0.9rem;">{icon} {company}</span>
            <span style="background:rgba(96,165,250,0.12); padding:0.1rem 0.45rem;
                         border-radius:1rem; font-size:0.82rem; font-weight:600; color:#60a5fa !important;">
                {count}
            </span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:0.25rem 0.5rem; margin:0.1rem 0; border-radius:0.4rem;
                    background:rgba(255,255,255,0.04); font-size:0.88rem;">
            <span style="font-size:0.9rem;">📰 News & Ads</span>
            <span style="background:rgba(96,165,250,0.12); padding:0.1rem 0.45rem;
                         border-radius:1rem; font-size:0.82rem; font-weight:600; color:#60a5fa !important;">
                {news_docs}
            </span>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── User Uploads stats ──
    if st.session_state.processed_uploads:
        st.markdown("##### 📤 Your Uploads")
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
            st.markdown(f'<div class="file-chip">📄 {fname}</div>', unsafe_allow_html=True)

        st.divider()

    # ── Tech Stack ──
    st.markdown("##### ⚙️ Tech Stack")
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
    <div style="font-size:0.85rem; opacity:0.65; line-height:1.45;">
        <b>Embeddings:</b> BGE-base-en v1.5<br>
        <b>LLM:</b> {_llm_display}<br>
        <b>Vector DB:</b> ChromaDB<br>
        <b>Retrieval:</b> Hybrid (metadata + vector)
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Actions ──
    st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.upload_messages = []
        st.rerun()


# ─────────────────────────────────────────────
# MAIN CONTENT — TABS
# ─────────────────────────────────────────────
tab_kb, tab_upload = st.tabs(["📚 Knowledge Base", "📤 Your Documents"])

# ═════════════════════════════════════════════
# TAB 1: Knowledge Base (existing functionality)
# ═════════════════════════════════════════════
with tab_kb:
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <div style="font-size:2.5rem; margin-bottom:0.6rem;">📑</div>
            <h2>DocIntel</h2>
            <p>Ask questions about BMW, Ford, and Tesla — annual reports and financials.<br>
            Answers grounded in real documents using hybrid retrieval.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("<p class='example-label'>Try one of these questions</p>", unsafe_allow_html=True)

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
        try:
            with st.spinner("🔍 Searching documents and generating answer..."):
                answer = generate_answer(query)
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
            <div style="font-size:2.5rem; margin-bottom:0.4rem;">📤</div>
            <h3>Upload Your Documents</h3>
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
            if st.button(f"⚡ Process {len(new_files)} new file(s)", use_container_width=True):
                # Initialize retriever if needed
                if st.session_state.user_retriever is None:
                    st.session_state.user_retriever = SessionRetriever()

                progress = st.progress(0, text="Processing documents...")
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
                f'<span class="file-chip">📄 {name} ({chunks} chunks)</span>'
                for name, chunks in st.session_state.processed_uploads.items()
            )
            st.markdown(file_chips, unsafe_allow_html=True)
        with cols[1]:
            if st.button("🗑️ Clear uploads", use_container_width=True):
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
            try:
                with st.spinner("🔍 Searching your documents..."):
                    answer = generate_answer_for_uploads(query, st.session_state.user_retriever)
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
