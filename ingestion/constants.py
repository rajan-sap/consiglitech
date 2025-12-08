"""
Ingestion Pipeline Configuration.

All settings for document processing, embedding, and chunking.
"""

# =============================================================================
# PATHS
# =============================================================================

DATA_PATH = "./data"
VECTOR_DB_PATH = "./chroma_db"
PROCESSED_FILES_PATH = "./chroma_db/.processed_files.json"
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt"]

# =============================================================================
# EMBEDDING
# =============================================================================

# EMBEDDING_MODEL_NAME = "text-embedding-3-small"   # 512, 1536 or 3072 dims embeddings

# Architecture: BGE model,  max seq length:512 tokens, embedding dimension:768, parameters: 109 MB
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"   

# Architecture: Gemma 3 model, max seq length:2048 tokens, embedding dimension: 768, parameters: 308M
# Almost 1 million downloads last month in November 2025, multilingual
# EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"

BATCH_SIZE = 100

# =============================================================================
# DOCUMENT TYPES
# =============================================================================

COMPANY_FOLDERS = ["BMW", "Ford", "Tesla"]
DOC_TYPE_ANNUAL_REPORT = "Annual Report"
DOC_TYPE_NEWS_ARTICLE = "News Article"

# =============================================================================
# CHUNKING
# =============================================================================

# Annual reports: larger chunks for financial context
ANNUAL_REPORT_SPLITTER = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separators": ["\n\n", "\n", ".\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
}

# News: smaller chunks for precise retrieval
NEWS_ARTICLE_SPLITTER = {
    "chunk_size": 500,
    "chunk_overlap": 100,
    "separators": ["\n\n", "\n", ".\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
}
