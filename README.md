# DocIntel

A RAG-based document intelligence app that lets you chat with company annual reports. Upload your own documents or query the built-in knowledge base covering BMW, Ford, and Tesla financials.

**Live demo:** [consiglitech.streamlit.app](https://consiglitech.streamlit.app)

---

## What it does

- Parses PDFs and DOCX files, splits them into chunks, and embeds them with BGE-base-en-v1.5
- Stores embeddings in ChromaDB with metadata (company, year, document type, page number)
- Decomposes complex queries into atomic sub-questions using an LLM
- Retrieves relevant chunks using hybrid search (metadata filtering + vector similarity)
- Generates grounded answers with source attribution — refuses to fabricate facts
- Classifies user intent to route between document queries and general conversation

## Two modes

| | Knowledge Base | Your Documents |
|---|---|---|
| **Data** | 9 pre-loaded annual reports (BMW, Ford, Tesla) | User-uploaded PDF/DOCX |
| **Storage** | Persistent ChromaDB (committed to repo) | In-memory, session-scoped |
| **Retrieval** | Hybrid: metadata filter + vector search | Pure vector similarity |
| **Parser** | LlamaParse (high quality) | pdfplumber (no API key needed) |

---

## Tech stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash (OpenAI-compatible endpoint) |
| Embeddings | BAAI/bge-base-en-v1.5 (768 dims) |
| Vector DB | ChromaDB |
| Orchestration | LangChain |
| PDF parsing | LlamaParse (ingestion), pdfplumber (uploads) |
| Deployment | Streamlit Community Cloud |

## Architecture

```
User Query
    |
    v
Intent Classification (LLM)
    |
    +--> GENERAL --> Direct LLM answer (no retrieval)
    |
    +--> DOCUMENT --> Query Decomposition (LLM)
                          |
                          v
                     Metadata Extraction (regex)
                          |
                          v
                     Hybrid Search (ChromaDB)
                          |
                          v
                     Context Aggregation
                          |
                          v
                     Grounded Answer (LLM)
```

---

## Project structure

```
├── streamlit_app.py              # App entry point (Streamlit UI)
├── llm_config.py                 # LLM provider configuration
│
├── ingestion/                    # Document processing pipeline
│   ├── constants.py              # Paths, models, chunk sizes
│   ├── ingest.py                 # Ingestion orchestrator
│   ├── utils.py                  # File loaders + chunk creators
│   └── upload_processor.py       # In-memory parser for user uploads
│
├── retrieval/                    # Search and query processing
│   ├── retriever.py              # Retriever class, query decomposition
│   ├── session_retriever.py      # In-memory retriever for uploads
│   └── utils.py                  # Metadata extraction utilities
│
├── generation/                   # Answer generation
│   └── generator.py              # Intent classification + RAG generation
│
├── evaluation/                   # RAG quality metrics
│   ├── config.py                 # Eval settings
│   ├── metrics.py                # MRR, Hit Rate, latency, token cost
│   ├── dataset_generator.py      # Synthetic QA pair generation
│   └── evaluate.py               # Evaluation orchestrator
│
├── tests/                        # 95 tests (unit + integration)
│   ├── unit/                     # Fast, no API key needed
│   └── integration/              # Requires LLM API + ChromaDB
│
├── docs/                         # Documentation
│   ├── PIPELINE_DOCS.md          # Full pipeline technical reference
│   └── TESTING.md                # Test guide with explanations
│
├── data/                         # Source PDFs (local only)
├── chroma_db/                    # Pre-built vector store
├── requirements.txt
├── pytest.ini
├── Dockerfile
└── docker-compose.yml
```

---

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/rajan-sap/consiglitech.git
cd consiglitech

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 2. Set up API keys

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

### 3. Run the app

```bash
streamlit run streamlit_app.py
```

Open [localhost:8501](http://localhost:8501).

### 4. Run tests

```bash
pytest tests/unit -v       # 86 unit tests, no API key needed
pytest tests/ -v -m unit   # same thing
```

---

## Configuration

All LLM settings are in `llm_config.py`. Switch providers by changing the base URL and model:

| Provider | Base URL | Model |
|---|---|---|
| Gemini (default) | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| OpenAI | `None` (default) | `gpt-4-1106-preview` |
| LM Studio (local) | `http://localhost:1234/v1` | `local-model` |

Chunking and embedding settings are in `ingestion/constants.py`.

---

## Rate limiting

The live demo is capped at 10 queries per hour per session to manage API costs. This resets automatically.

---

## License

MIT
