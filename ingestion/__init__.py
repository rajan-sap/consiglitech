"""
Ingestion Package.

Main entry point: create_vector_store()
"""

from .constants import (
    BATCH_SIZE,
    COMPANY_FOLDERS,
    DATA_PATH,
    DOC_TYPE_ANNUAL_REPORT,
    DOC_TYPE_NEWS_ARTICLE,
    EMBEDDING_MODEL_NAME,
    FINANCIAL_REPORT_SPLITTER,
    NEWS_ARTICLE_SPLITTER,
    PROCESSED_FILES_PATH,
    SUPPORTED_FILE_TYPES,
    VECTOR_DB_PATH,
)
from .ingest import (
    chunk_documents,
    create_vector_store,
    get_new_files,
    load_all_documents,
    load_processed_files,
    process_file,
    save_processed_files,
)
from .utils import (
    create_news_chunk,
    create_report_chunk,
    extract_year_from_filename,
    get_company_from_path,
    load_docx,
    load_document,
    load_pdf,
    load_txt,
)

__all__ = [
    # Main entry
    "create_vector_store",
    # Processing
    "load_all_documents",
    "process_file",
    "chunk_documents",
    # Loaders
    "load_document",
    "load_pdf",
    "load_docx",
    "load_txt",
    # Chunk creators
    "create_report_chunk",
    "create_news_chunk",
    # Helpers
    "extract_year_from_filename",
    "get_company_from_path",
    # Constants
    "DATA_PATH",
    "VECTOR_DB_PATH",
    "SUPPORTED_FILE_TYPES",
    "BATCH_SIZE",
    "EMBEDDING_MODEL_NAME",
    "COMPANY_FOLDERS",
    "DOC_TYPE_ANNUAL_REPORT",
    "DOC_TYPE_NEWS_ARTICLE",
    "FINANCIAL_REPORT_SPLITTER",
    "NEWS_ARTICLE_SPLITTER",
]
