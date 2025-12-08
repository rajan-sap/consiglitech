# RAG PDF Question Answering Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for question answering over PDF documents. Built with LangChain, ChromaDB, and Streamlit.

## Features

- **PDF Ingestion**: Parse PDFs using LlamaParse with markdown extraction
- **Smart Chunking**: Different chunking strategies for annual reports vs news articles
- **Vector Search**: ChromaDB with HuggingFace BGE embeddings (768 dimensions)
- **Chat Interface**: Streamlit-based RAG chatbot

## Project Structure

```
consiglitech/
├── ingestion/          # Document loading, chunking, embedding
│   ├── constants.py    # Configuration (chunk sizes, paths, etc.)
│   ├── ingest.py       # Main ingestion pipeline
│   └── utils.py        # PDF/DOCX loaders, chunk creators
├── retrieval/          # Document retrieval logic
├── generation/         # Answer generation (LLM integration)
├── evaluation/         # Evaluation scripts
├── data/               # PDF documents (not tracked in git)
│   ├── BMW/            # Company annual reports
│   ├── Tesla/
│   └── news.pdf
├── chroma_db/          # Vector database (auto-generated)
├── main.py             # Entry point for ingestion
├── streamlit_app.py    # RAG chatbot UI
└── requirements.txt    # Python dependencies
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/rajan-sap/consiglitech.git
cd consiglitech
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy the template
cp .env.example .env

# Edit .env with your API keys
# Required: LLAMA_CLOUD_API_KEY (get from https://cloud.llamaindex.ai/)
```

### 4. Add Your Documents

Place PDF files in the `data/` folder:
- Company reports go in subfolders: `data/BMW/`, `data/Tesla/`, etc.
- News articles go directly in `data/`

### 5. Run Ingestion

```bash
python main.py
```

This will:
- Parse all PDFs using LlamaParse
- Chunk documents based on type (annual report vs news)
- Embed chunks using BGE embeddings
- Store in ChromaDB

### 6. Start the App

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Docker Setup

### Using Docker Compose (Recommended)

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 2. Add your PDFs to data/ folder

# 3. Run ingestion (first time only, or when adding new documents)
docker-compose run --rm ingestion

# 4. Start the app
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the app
docker-compose down
```

### Using Docker Directly

```bash
# Build the image
docker build -t consiglitech .

# Run the app
docker run -p 8501:8501 \
  -v ./data:/app/data \
  -v ./chroma_db:/app/chroma_db \
  --env-file .env \
  consiglitech

# Or run ingestion only
docker run -v ./data:/app/data -v ./chroma_db:/app/chroma_db --env-file .env consiglitech python main.py
```

## Configuration

Key settings in `ingestion/constants.py`:

| Setting | Value | Description |
|---------|-------|-------------|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |
| `ANNUAL_REPORT_SPLITTER.chunk_size` | 1000 | Chunk size for reports |
| `NEWS_ARTICLE_SPLITTER.chunk_size` | 500 | Chunk size for news |
| `BATCH_SIZE` | 100 | Embedding batch size |

## Utilities

```bash
# Inspect the vector database
python -m inspect_db

# Test retrieval quality
python -m test_retrieval "your query here"
```

## Requirements

- Python 3.10+
- LlamaParse API key (for PDF parsing)
- ~2GB disk space for embeddings model

## License

MIT
