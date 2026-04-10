# DocIntel — Pipeline Documentation

Technical reference for the DocIntel RAG pipeline. Covers the full data flow from document ingestion to answer generation, for both the pre-loaded knowledge base and user-uploaded documents.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Ingestion Pipeline (Pre-loaded KB)](#ingestion-pipeline-pre-loaded-kb)
4. [Flow 1: User Chatting with Knowledge Base](#flow-1-user-chatting-with-knowledge-base)
5. [Flow 2: User Chatting with Own Documents](#flow-2-user-chatting-with-own-documents)
6. [Component Details](#component-details)
7. [Configuration Reference](#configuration-reference)
8. [Deployment (Streamlit Cloud)](#deployment-streamlit-cloud)

---

## Architecture Overview

DocIntel has two independent RAG paths that share the same LLM and embedding model:

| | Knowledge Base (Tab 1) | Your Documents (Tab 2) |
|---|---|---|
| **Data source** | Pre-loaded PDFs (BMW, Ford, Tesla) | User-uploaded PDF/DOCX |
| **Vector store** | Persistent ChromaDB (`./chroma_db`) | In-memory ChromaDB (session-scoped) |
| **Parser** | LlamaParse (cloud API) | pdfplumber / python-docx (local) |
| **Collection** | `"documents"` | `"user_documents"` |
| **Retrieval** | Hybrid (metadata filter + vector) | Pure vector similarity |
| **Query processing** | Decomposition + metadata extraction | Direct search (no decomposition) |
| **Persistence** | Survives reboot (committed to git) | Lost when session ends |

---

## Project Structure

```
├── streamlit_app.py              # Main UI — two tabs (KB + Upload)
├── llm_config.py                 # LLM provider config (Gemini 2.5 Flash)
├── main.py                       # CLI entry point for ingestion
│
├── ingestion/                    # Document processing
│   ├── constants.py              # Paths, embedding model, chunk sizes
│   ├── ingest.py                 # Ingestion orchestrator (pre-loaded KB)
│   ├── utils.py                  # Loaders (LlamaParse, DOCX, TXT) + chunk creators
│   └── upload_processor.py       # User upload parser (pdfplumber, python-docx)
│
├── retrieval/                    # Query processing & vector search
│   ├── retriever.py              # Retriever class, query decomposition, metadata extraction
│   ├── session_retriever.py      # In-memory retriever for user uploads
│   └── utils.py                  # Metadata extraction utilities
│
├── generation/                   # LLM answer generation
│   └── generator.py              # Intent classifier + two generation paths
│
├── data/                         # Source PDFs (not on Streamlit Cloud)
│   ├── BMW/                      # 3 annual reports (2021-2023)
│   ├── Ford/                     # 3 annual reports (2021-2023)
│   ├── Tesla/                    # 2 annual reports (2022-2023)
│   └── News/                     # 1 news article
│
├── chroma_db/                    # Persistent vector store (committed to git)
│   ├── chroma.sqlite3            # SQLite database with embeddings
│   └── .processed_files.json     # Tracks which files have been ingested
│
├── requirements.txt              # Python dependencies
└── runtime.txt                   # Python version pin for Streamlit Cloud
```

---

## Ingestion Pipeline (Pre-loaded KB)

This runs locally via `python main.py` before deployment. It processes PDFs from `./data` into ChromaDB.

```mermaid
flowchart TD
    A[python main.py] --> B[create_vector_store]
    B --> C{ChromaDB exists?}

    C -->|No| D[collect_file_paths ./data]
    D --> E[load_all_documents]
    E --> F[For each file: process_file]
    F --> G[load_pdf via LlamaParse]
    F --> H[load_docx via python-docx]
    G --> I[Determine doc type from folder]
    H --> I
    I -->|Company folder| J[chunk with ANNUAL_REPORT_SPLITTER<br/>1000 chars, 200 overlap]
    I -->|Other| K[chunk with NEWS_ARTICLE_SPLITTER<br/>500 chars, 100 overlap]
    J --> L[Attach metadata:<br/>file_name, company, year,<br/>document_type, page_number]
    K --> M[Attach metadata:<br/>file_name, document_type,<br/>page_number]
    L --> N[Batch embed with BGE-base-en-v1.5<br/>768 dims, batches of 100]
    M --> N
    N --> O[Store in ChromaDB<br/>collection: documents]
    O --> P[Save .processed_files.json]

    C -->|Yes| Q[load_processed_files]
    Q --> R[collect_file_paths ./data]
    R --> S[get_new_files diff]
    S -->|New files found| F
    S -->|No new files| T[Return existing store]

    style A fill:#1e3a5f,color:#e0ecff
    style O fill:#1e3a5f,color:#e0ecff
    style T fill:#1e3a5f,color:#e0ecff
```

**Key functions:**

| Function | File | Purpose |
|---|---|---|
| `create_vector_store()` | `ingestion/ingest.py:216` | Main entry point — creates or updates ChromaDB |
| `process_file(path)` | `ingestion/ingest.py:151` | Load, detect type, chunk, attach metadata |
| `load_pdf(path)` | `ingestion/utils.py:28` | Parse PDF via LlamaParse (markdown output) |
| `chunk_documents(docs, config, creator)` | `ingestion/ingest.py:102` | Split text + apply metadata |
| `load_processed_files()` | `ingestion/ingest.py:53` | Read tracking JSON for deduplication |

**Metadata schema (Annual Report):**
```json
{
  "file_name": "Tesla_Annual_Report_2023.pdf",
  "document_type": "Annual Report",
  "company": "Tesla",
  "year": "2023",
  "page_number": 42
}
```

**Metadata schema (News Article):**
```json
{
  "file_name": "news.pdf",
  "document_type": "News Article",
  "page_number": 3
}
```

---

## Flow 1: User Chatting with Knowledge Base

This is the default tab. Queries go through intent classification, query decomposition, metadata-filtered retrieval, and grounded answer generation.

```mermaid
flowchart TD
    A[User enters query in Tab 1] --> B[generate_answer query]
    B --> C{is_general_query?<br/>LLM classifies GENERAL vs DOCUMENT}

    C -->|GENERAL| D[LLM generates direct answer<br/>No retrieval needed<br/>temperature: 0.7]
    D --> E[Display answer]

    C -->|DOCUMENT| F[_get_retriever<br/>Lazy-load Retriever + ChromaDB]
    F --> G{retriever.is_available?}

    G -->|No| H[Show: Knowledge base not available]

    G -->|Yes| I[retrieve_aggregated_context]
    I --> J[decompose_query<br/>LLM breaks into atomic sub-queries]

    J --> K["Example:<br/>'Tesla revenue 2022 and 2023'<br/>→ 'Tesla revenue 2022?'<br/>→ 'Tesla revenue 2023?'"]

    K --> L[For each sub-query]
    L --> M[extract_metadata_from_query<br/>Regex: company, year, doc_type]
    M --> N["Build ChromaDB filter<br/>{$and: [{company: Tesla}, {year: 2022}]}"]
    N --> O[retriever.search<br/>k=3, similarity_search_with_score]
    O --> P[Convert L2 → cosine similarity<br/>cosine = 1 - L2^2/4]

    P --> Q[Aggregate all results into context string]
    Q --> R[truncate_context<br/>max 30,000 chars]
    R --> S["Build prompt:<br/>Question: {query}<br/>Context: {aggregated_context}"]
    S --> T[LLM generates grounded answer<br/>SYSTEM_PROMPT_RAG<br/>temperature: 0.1]
    T --> E

    style A fill:#1e3a5f,color:#e0ecff
    style E fill:#1e3a5f,color:#e0ecff
    style D fill:#2d4a2d,color:#c8e6c8
    style T fill:#2d4a2d,color:#c8e6c8
```

**Detailed step-by-step:**

1. **Intent Classification** (`generator.py:30`) — LLM receives the query with a system prompt that asks for a single word: `DOCUMENT` or `GENERAL`. Temperature 0.2 for deterministic output. On failure, defaults to DOCUMENT (safer).

2. **General Route** — If GENERAL, the LLM answers directly without any retrieval. Temperature 0.7 for natural conversation.

3. **Document Route** — If DOCUMENT:
   - **Query Decomposition** (`retriever.py:24`) — LLM breaks complex queries into atomic sub-questions using few-shot prompting. Example: "Compare BMW and Ford revenue over 3 years" becomes 6 separate queries.
   - **Metadata Extraction** (`retriever.py:66`) — Regex-based extraction of company names (BMW/Ford/Tesla), years (20XX pattern), and document type. Financial keywords trigger Annual Report preference.
   - **Hybrid Search** (`retriever.py:184`) — For each sub-query, ChromaDB filters by metadata first, then ranks by vector similarity (BGE embeddings). Returns top-3 per sub-query.
   - **Context Aggregation** (`retriever.py:235`) — All results merged into a single context string with document content and metadata.
   - **Answer Generation** (`generator.py:160`) — LLM receives the context with `SYSTEM_PROMPT_RAG` instructing it to answer ONLY from the provided context. Temperature 0.1 for factual precision.

---

## Flow 2: User Chatting with Own Documents

This is the "Your Documents" tab. Users upload files, which are parsed and embedded in a session-scoped in-memory vector store.

```mermaid
flowchart TD
    A[User uploads PDF/DOCX in Tab 2] --> B[Click 'Process Documents']

    B --> C[Initialize SessionRetriever<br/>In-memory chromadb.Client]

    C --> D[For each uploaded file]
    D --> E[process_uploaded_file]
    E --> F{File type?}

    F -->|PDF| G[parse_uploaded_pdf<br/>pdfplumber extracts text per page]
    F -->|DOCX| H[parse_uploaded_docx<br/>python-docx extracts paragraphs]

    G --> I[chunk_uploaded_documents<br/>1000 chars, 200 overlap]
    H --> I
    I --> J[Attach metadata:<br/>file_name, page_number,<br/>document_type: User Upload]

    J --> K[session_retriever.add_documents<br/>Embed with BGE + store in memory]
    K --> L[Show progress + chunk count]
    L --> M[Files ready for chat]

    M --> N[User enters query in Tab 2]
    N --> O[generate_answer_for_uploads]
    O --> P[session_retriever.search<br/>k=5, pure vector similarity<br/>No metadata filtering]
    P --> Q[Build context from top-5 results]
    Q --> R[truncate_context<br/>max 30,000 chars]
    R --> S["Build prompt:<br/>Question: {query}<br/>Context: {context}"]
    S --> T[LLM generates answer<br/>SYSTEM_PROMPT_UPLOAD_RAG<br/>temperature: 0.1]
    T --> U[Display answer]

    style A fill:#1e3a5f,color:#e0ecff
    style N fill:#1e3a5f,color:#e0ecff
    style U fill:#1e3a5f,color:#e0ecff
    style T fill:#2d4a2d,color:#c8e6c8
```

**Key differences from the KB flow:**

| Aspect | Knowledge Base (Tab 1) | User Uploads (Tab 2) |
|---|---|---|
| Intent classification | Yes (GENERAL vs DOCUMENT) | No (always searches docs) |
| Query decomposition | Yes (complex → atomic) | No (direct search) |
| Metadata filtering | Yes (company, year, type) | No (pure vector similarity) |
| Top-k per query | 3 per sub-query | 5 total |
| Parser | LlamaParse (cloud, high quality) | pdfplumber (local, good enough) |
| Storage | Persistent ChromaDB on disk | In-memory, session-scoped |
| Embeddings | BGE-base-en-v1.5 (shared) | BGE-base-en-v1.5 (shared) |

**Session lifecycle:**
- `SessionRetriever` is created when the user first clicks "Process Documents"
- Stored in `st.session_state.user_retriever`
- Survives Streamlit reruns within the same browser session
- Destroyed when the tab is closed or the user clicks "Clear uploads"
- Uses `@st.cache_resource` to share the BGE embedding model with the KB retriever (avoids loading 109 MB twice)

---

## Component Details

### LLM Configuration (`llm_config.py`)

```
Provider:     Google (Gemini)
Model:        gemini-2.5-flash
Endpoint:     https://generativelanguage.googleapis.com/v1beta/openai/
API Key:      GEMINI_API_KEY (from Streamlit secrets)
Client:       OpenAI SDK (compatible endpoint)
Timeout:      60 seconds
Context max:  30,000 characters
```

The app uses the OpenAI Python SDK pointed at Google's OpenAI-compatible endpoint. Switching providers requires changing only `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` in `llm_config.py`.

### Embedding Model

```
Model:        BAAI/bge-base-en-v1.5
Dimensions:   768
Max tokens:   512
Size:         109 MB
Device:       CPU
Normalization: Enabled
```

### ChromaDB

```
Persistent store:   ./chroma_db/chroma.sqlite3
Collection name:    "documents" (KB), "user_documents" (uploads)
Distance metric:    L2 (converted to cosine: 1 - L2^2/4)
```

### System Prompts

| Prompt | Used for | Key instruction |
|---|---|---|
| `CLASSIFY_SYSTEM_PROMPT` | Intent classification | "Reply with DOCUMENT or GENERAL" |
| `SYSTEM_PROMPT_RAG` | KB answers | "Answer using ONLY the retrieved context. Do NOT fabricate facts." |
| `SYSTEM_PROMPT_UPLOAD_RAG` | Upload answers | "Answer using ONLY the retrieved context. Cite file name or page." |
| `SYSTEM_PROMPT_GENERAL` | General chat | Conversational assistant persona |

### LLM Temperature Settings

| Task | Temperature | Rationale |
|---|---|---|
| Intent classification | 0.2 | Near-deterministic single-word output |
| RAG answer generation | 0.1 | Factual, grounded in context |
| General conversation | 0.7 | Natural, creative responses |

---

## Configuration Reference

### `ingestion/constants.py`

| Constant | Value | Purpose |
|---|---|---|
| `DATA_PATH` | `./data` | Source document directory |
| `VECTOR_DB_PATH` | `./chroma_db` | ChromaDB persist directory |
| `PROCESSED_FILES_PATH` | `./chroma_db/.processed_files.json` | Ingestion tracking |
| `SUPPORTED_FILE_TYPES` | `.pdf, .docx, .txt` | Accepted file extensions |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |
| `BATCH_SIZE` | `100` | Embedding batch size |
| `COMPANY_FOLDERS` | `BMW, Ford, Tesla` | Known company directories |
| `ANNUAL_REPORT_SPLITTER` | 1000 chars, 200 overlap | Chunk config for reports |
| `NEWS_ARTICLE_SPLITTER` | 500 chars, 100 overlap | Chunk config for news |

### `requirements.txt` key dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `openai` | LLM client (OpenAI-compatible, used for Gemini) |
| `chromadb` | Vector database |
| `langchain-chroma` | LangChain ChromaDB integration |
| `langchain-huggingface` | HuggingFace embeddings wrapper |
| `sentence-transformers` | BGE embedding model runtime |
| `pdfplumber` | PDF parsing for user uploads |
| `python-docx` | DOCX parsing for user uploads |
| `llama-parse` | PDF parsing for pre-loaded KB (cloud API) |
| `protobuf>=3.19.0,<5.0.0` | Pinned to avoid opentelemetry crash |

---

## Deployment (Streamlit Cloud)

### Secrets required

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

### Settings

- **Python version**: 3.11 (set in Advanced Settings, NOT via runtime.txt)
- **Main file**: `streamlit_app.py`
- **Repository**: GitHub repo with `chroma_db/` committed

### How it works on Streamlit Cloud

1. Streamlit Cloud clones the repo (includes `chroma_db/` with pre-built embeddings)
2. Installs `requirements.txt` dependencies
3. Runs `streamlit_app.py`
4. The `Retriever` class loads ChromaDB from `./chroma_db` on first query (lazy init)
5. User uploads create in-memory ChromaDB collections (ephemeral, no disk writes needed)
6. On reboot/redeploy, the repo is re-cloned — pre-loaded KB is intact, user uploads are gone

### Adding new documents to the KB

1. Place PDFs in `./data/<Company>/` locally
2. Run `python main.py` to process and embed
3. Commit updated `chroma_db/` and `data/` to git
4. Push to GitHub — Streamlit Cloud redeploys automatically
