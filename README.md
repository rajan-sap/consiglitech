# DocIntel — Document Intelligence App

A RAG-based document query tool that lets you chat with company annual reports or 1000 of pages. Drop in PDFs from BMW, Ford, Tesla (or others), and ask questions — the app retrieves the right chunks and generates grounded answers using an LLM.

Built with Streamlit, ChromaDB, and Groq (llama-3.3-70b).

**Live demo:** [consiglitech.streamlit.app](https://consiglitech.streamlit.app) *(Streamlit Community Cloud)*

---

## What it does

- Parses PDFs (annual reports, news articles) and splits them into chunks
- Embeds chunks with HuggingFace BGE and stores them in ChromaDB
- Retrieves relevant context using hybrid search (metadata filters + vector similarity)
- Classifies user queries with the LLM — general/meta questions get answered directly, document questions go through the full RAG pipeline
- Shows a dark-theme chat UI with a sidebar that breaks down the knowledge base

## Project layout

```
├── streamlit_app.py        # Main UI
├── llm_config.py           # LLM provider config (Groq / LM Studio / OpenAI)
├── main.py                 # Ingestion entry point
│
├── ingestion/              # PDF loading, chunking, embedding
│   ├── constants.py        # Chunk sizes, model name, paths
│   ├── ingest.py           # Ingestion pipeline
│   └── utils.py            # Loaders + chunk helpers
│
├── retrieval/              # Hybrid retrieval logic
│   ├── retriever.py
│   └── utils.py
│
├── generation/             # LLM query classification + answer generation
│   ├── generator.py        # Intent classifier + two-route generation
│   └── prompts.py
│
├── evaluation/             # Synthetic QA eval pipeline
│   ├── config.py           # Known reports, QA pairs per report
│   ├── dataset_generator.py
│   ├── evaluate.py
│   ├── metrics.py          # MRR, Hit Rate, latency, token cost
│   └── eval_data/          # Generated datasets + results
│
├── data/                   # PDFs — not tracked in git
│   ├── BMW/                # 3 annual reports (2021–2023)
│   ├── Ford/               # 3 annual reports (2021–2023)
│   ├── Tesla/              # 2 annual reports (2022–2023)
│   └── News and Advertisement/
│
├── chroma_db/              # Vector DB (auto-generated after ingestion)
├── .streamlit/secrets.toml # Local API keys — gitignored
├── ISSUES_AND_FIXES.md     # Dev log of every bug + resolution
├── requirements.txt        # 17 direct deps, loose pins
├── Dockerfile
└── docker-compose.yml
```

## Getting started

### 1. Clone and set up

```bash
git clone https://github.com/rajan-sap/consiglitech.git
cd consiglitech

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Add your API keys

Create `.streamlit/secrets.toml` (this file is gitignored):

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

You'll also need a `LLAMA_CLOUD_API_KEY` in your `.env` if you want to re-run ingestion with LlamaParse. Get one at [cloud.llamaindex.ai](https://cloud.llamaindex.ai/).

### 3. Add PDFs and run ingestion

Put your PDF files under `data/` — company reports in subfolders (`data/BMW/`, `data/Ford/`, etc.), news articles in `data/News and Advertisement/`.

```bash
python main.py
```

This parses every PDF, chunks them (1000 chars for reports, 500 for news), embeds with BGE-base-en-v1.5, and stores everything in `chroma_db/`.

### 4. Launch the app

```bash
streamlit run streamlit_app.py
```

Then open [localhost:8501](http://localhost:8501).

## Docker (alternative)

```bash
# Set up .env with your keys first, then:
docker-compose run --rm ingestion   # one-time ingest
docker-compose up -d                # start the app
docker-compose logs -f              # tail logs
```

Or build and run manually:

```bash
docker build -t consiglitech .
docker run -p 8501:8501 -v ./data:/app/data -v ./chroma_db:/app/chroma_db --env-file .env consiglitech
```

## Key config

All LLM settings live in `llm_config.py` — switch providers by commenting/uncommenting a block:

| Setting | Where | Default |
|---------|-------|---------|
| LLM provider | `llm_config.py` | Groq (`llama-3.3-70b-versatile`) |
| Embedding model | `ingestion/constants.py` | `BAAI/bge-base-en-v1.5` (768-dim) |
| Report chunk size | `ingestion/constants.py` | 1000 chars |
| News chunk size | `ingestion/constants.py` | 500 chars |
| Context limit | `llm_config.py` | 30,000 chars |
| Client timeout | `llm_config.py` | 60s |

## Evaluation

The eval pipeline generates synthetic QA pairs from the actual reports and scores retrieval + generation quality. See `evaluation/` for details or run:

```bash
python -m evaluation.evaluate
```

Results land in `evaluation/eval_data/eval_results.json`.

## Useful commands

```bash
python -m inspect_db        # peek inside the vector DB
```

## Known issues

See [ISSUES_AND_FIXES.md](ISSUES_AND_FIXES.md) for a running log of every bug encountered during development and how each was resolved (10 issues documented so far).

## Requirements

- Python 3.10+
- A Groq API key (free tier works fine) — or swap in OpenAI / LM Studio in `llm_config.py`
- ~2 GB disk for the embedding model on first run

## License

MIT
