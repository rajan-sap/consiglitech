import streamlit as st
import os
import glob
from generation.generator import generate_answer
from llm_config import LLM_MODEL, LLM_BASE_URL

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
    padding-top: 1.5rem; padding-bottom: 1.5rem;
    justify-content: space-evenly;
    overflow-y: auto;
    overflow-x: hidden;
    gap: 0.8rem;
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
    font-size: 1.02rem;
}
section[data-testid="stSidebar"] { min-width: 330px !important; }
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.06);
    margin: 0.8rem 0;
}

/* ── Chat bubbles ── */
.chat-row { display: flex; margin: 0.75rem 0; align-items: flex-start; gap: 0.7rem; }
.chat-row.user { flex-direction: row-reverse; }

.chat-avatar {
    width: 38px; height: 38px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; flex-shrink: 0; font-weight: 600;
    letter-spacing: 0.02em;
}
.chat-avatar.user-av { background: #2563eb; color: #dbeafe; }
.chat-avatar.bot-av  { background: #1a7f64; color: #d1fae5; }

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

/* ── Stat cards ── */
.stat-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 0.6rem; padding: 0.5rem 0.75rem; margin-bottom: 0.3rem;
    text-align: center;
}
.stat-card .stat-value { font-size: 1.8rem; font-weight: 700; color: #60a5fa !important; }
.stat-card .stat-label { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.55; }

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

# ─────────────────────────────────────────────
# HELPER: count ingested docs
# ─────────────────────────────────────────────
def count_data_files():
    """Return basic stats about the data directory."""
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
    # Count News and Advertisement folder
    news_folder = os.path.join("./data", "News and Advertisement")
    news_count = 0
    if os.path.isdir(news_folder):
        news_count = len([f for f in os.listdir(news_folder) if os.path.isfile(os.path.join(news_folder, f))])
        total += news_count
    # count root-level data files too
    root_files = [f for f in os.listdir("./data") if os.path.isfile(os.path.join("./data", f))]
    total += len(root_files)
    return total, per_company, news_count

total_docs, docs_per_company, news_docs = count_data_files()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo / branding
    st.markdown("""
    <div style="text-align:center; padding: 0rem 0 0.4rem 0;">
        <div style="font-size:2.2rem;">📑</div>
        <div style="font-size:1.35rem; font-weight:700; letter-spacing:-0.02em; margin-top:0.15rem; color:#e8e8ec !important;">
            DocIntel
        </div>
        <div style="font-size:0.82rem; opacity:0.45; text-transform:uppercase; letter-spacing:0.1em;">
            Document Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Knowledge base stats ──
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

    # Per-company breakdown + News
    for company, count in docs_per_company.items():
        icon = {"BMW": "🚗", "Ford": "🚙", "Tesla": "⚡"}.get(company, "📁")
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:0.35rem 0.6rem; margin:0.15rem 0; border-radius:0.45rem;
                    background:rgba(255,255,255,0.04); font-size:1rem;">
            <span style="font-size:1.02rem;">{icon} {company}</span>
            <span style="background:rgba(96,165,250,0.12); padding:0.15rem 0.55rem;
                         border-radius:1rem; font-size:0.92rem; font-weight:600; color:#60a5fa !important;">
                {count}
            </span>
        </div>""", unsafe_allow_html=True)

    # News and Advertisements row
    st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:0.35rem 0.6rem; margin:0.15rem 0; border-radius:0.45rem;
                    background:rgba(255,255,255,0.04); font-size:1rem;">
            <span style="font-size:1.02rem;">📰 News & Ads</span>
            <span style="background:rgba(96,165,250,0.12); padding:0.15rem 0.55rem;
                         border-radius:1rem; font-size:0.92rem; font-weight:600; color:#60a5fa !important;">
                {news_docs}
            </span>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── RAG Pipeline info ──
    st.markdown("##### ⚙️ Tech Stack")
    # Derive a friendly display name for the active LLM provider + model
    if LLM_BASE_URL and "groq" in LLM_BASE_URL:
        _llm_provider = "Groq"
    elif LLM_BASE_URL and "localhost" in LLM_BASE_URL:
        _llm_provider = "LM Studio"
    else:
        _llm_provider = "OpenAI"
    _llm_display = f"{_llm_provider} / {LLM_MODEL}"

    st.markdown(f"""
    <div style="font-size:0.95rem; opacity:0.65; line-height:1.55;">
        <b>Embeddings:</b> BGE-base-en v1.5<br>
        <b>LLM:</b> {_llm_display}<br>
        <b>Vector DB:</b> ChromaDB<br>
        <b>Retrieval:</b> Hybrid (metadata + vector)
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── RAG Evals ──
    with st.expander("📈 RAG Evals", expanded=False):
        st.markdown("""
        <div style="font-size:0.9rem; line-height:1.65; opacity:0.75;">
            <b>Retrieval Metrics</b>
            <ul style="margin:0.3rem 0 0.6rem 1.2rem; padding:0;">
                <li><b>Context Precision</b> — Are the retrieved chunks relevant to the query?</li>
                <li><b>Context Recall</b> — Are all necessary chunks retrieved?</li>
                <li><b>MRR</b> — How high does the correct document rank?</li>
                <li><b>Hit Rate @k</b> — Is the answer in the top-k results?</li>
            </ul>
            <b>Generation Metrics</b>
            <ul style="margin:0.3rem 0 0.6rem 1.2rem; padding:0;">
                <li><b>Faithfulness</b> — Is the answer grounded in retrieved context?</li>
                <li><b>Answer Relevancy</b> — Does the answer address the question?</li>
                <li><b>Hallucination Rate</b> — % of claims not supported by context</li>
            </ul>
            <b>End-to-End</b>
            <ul style="margin:0.3rem 0 0.2rem 1.2rem; padding:0;">
                <li><b>Correctness</b> — Does the answer match the ground truth?</li>
                <li><b>Latency (P50/P95)</b> — Response time percentiles</li>
                <li><b>Token Cost</b> — Tokens consumed per query</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Actions ──
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
# Welcome screen when no messages yet
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div style="font-size:2.5rem; margin-bottom:0.6rem;">📑</div>
        <h2>DocIntel</h2>
        <p>Ask questions about BMW, Ford, and Tesla — annual reports, financials, and news.<br>
        Answers grounded in real documents using hybrid retrieval.<br>
        <span style="opacity:0.5; font-size:0.9rem;">Real-time news coming soon.</span></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Example questions
    st.markdown("<p class='example-label'>Try one of these questions</p>", unsafe_allow_html=True)

    example_questions = [
        "What was Tesla's revenue in 2022?",
        "Compare BMW and Ford's net income over the past 3 years",
        "Summarize recent news about Tesla",
        "What are BMW's key financial highlights for 2023?",
        "How did Ford perform in the EV market?",
        "What is Tesla's operating margin trend?",
    ]

    # Render example questions as clickable buttons
    cols = st.columns(3)
    for i, q in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.session_state.pending_query = q
                st.rerun()

else:
    # ── Chat history ──
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            avatar_class = "user-av"
            bubble_class = "user-bubble"
            row_class = "user"
            avatar_text = "You"
        else:
            avatar_class = "bot-av"
            bubble_class = "bot-bubble"
            row_class = ""
            avatar_text = "AI"

        st.markdown(f"""
        <div class="chat-row {row_class}">
            <div class="chat-avatar {avatar_class}">{avatar_text[0]}</div>
            <div class="chat-bubble {bubble_class}">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHAT INPUT (always visible at bottom)
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about BMW, Ford, or Tesla..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_query = prompt
    st.rerun()

# ─────────────────────────────────────────────
# GENERATE ANSWER FOR PENDING QUERY
# ─────────────────────────────────────────────
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

    with st.spinner("🔍 Searching documents and generating answer..."):
        answer = generate_answer(query)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()