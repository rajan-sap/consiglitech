"""
Ingestion Pipeline for RAG Document Processing.

Workflow:
1. Scan data directory for documents
2. Check for new files not yet in vector store
3. Load and extract text from new files
4. Chunk text with appropriate splitter
5. Apply metadata based on document type
6. Embed and store in vector database
"""

import json
import os
from typing import Callable, Dict, List, Set

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .constants import (
    BATCH_SIZE,
    COMPANY_FOLDERS,
    DATA_PATH,
    EMBEDDING_MODEL_NAME,
    ANNUAL_REPORT_SPLITTER,
    NEWS_ARTICLE_SPLITTER,
    PROCESSED_FILES_PATH,
    SUPPORTED_FILE_TYPES,
    VECTOR_DB_PATH,
)
from .utils import (
    create_news_chunk,
    create_report_chunk,
    extract_year_from_filename,
    get_company_from_path,
    load_document,
)

load_dotenv()


# =============================================================================
# FILE TRACKING
# =============================================================================


def load_processed_files() -> Set[str]:
    """Load set of already processed file paths."""
    if not os.path.exists(PROCESSED_FILES_PATH):
        return set()
    
    with open(PROCESSED_FILES_PATH, "r") as f:
        return set(json.load(f))


def save_processed_files(files: Set[str]) -> None:
    """Save set of processed file paths."""
    os.makedirs(os.path.dirname(PROCESSED_FILES_PATH), exist_ok=True)
    with open(PROCESSED_FILES_PATH, "w") as f:
        json.dump(list(files), f, indent=2)


def get_new_files(all_files: List[str], processed: Set[str]) -> List[str]:
    """Return files that haven't been processed yet."""
    # Normalize paths for comparison
    processed_normalized = {os.path.normpath(p) for p in processed}
    return [f for f in all_files if os.path.normpath(f) not in processed_normalized]


# =============================================================================
# TEXT SPLITTING
# =============================================================================


def get_text_splitter(config: Dict) -> RecursiveCharacterTextSplitter:
    """Create a text splitter from configuration."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=config["separators"],
    )


# =============================================================================
# CHUNKING
# =============================================================================


def chunk_documents(
    documents: List[Document],
    splitter_config: Dict,
    chunk_creator: Callable,
    **metadata,
) -> List[Document]:
    """
    Split documents into chunks and apply metadata.
    
    Args:
        documents: Loaded document objects
        splitter_config: Chunking configuration dict
        chunk_creator: Function to create chunk with metadata
        **metadata: Additional metadata for each chunk
    
    Returns:
        List of Document chunks with metadata
    """
    splitter = get_text_splitter(splitter_config)
    chunks = []

    for doc in documents:
        texts = splitter.split_text(doc.page_content)
        page = doc.metadata.get("page_number", 1)

        for text in texts:
            chunk = chunk_creator(text=text, page_number=page, **metadata)
            chunks.append(chunk)

    return chunks


# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================


def collect_file_paths(data_path: str) -> List[str]:
    """Recursively collect all supported file paths."""
    paths = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in SUPPORTED_FILE_TYPES:
                paths.append(os.path.join(root, file))
    return paths


def process_file(file_path: str) -> List[Document]:
    """
    Process a single file into chunks with appropriate metadata.
    
    Determines document type from folder structure:
    - Company folders (BMW, Ford, Tesla) → Annual Report
    - Other files → News Article
    """
    file_name = os.path.basename(file_path)
    documents = load_document(file_path)

    if not documents:
        return []

    # Check if file is in a company folder
    company = get_company_from_path(file_path, COMPANY_FOLDERS)

    if company:
        # Annual Report
        year = extract_year_from_filename(file_name)
        return chunk_documents(
            documents=documents,
            splitter_config=ANNUAL_REPORT_SPLITTER,
            chunk_creator=create_report_chunk,
            file_name=file_name,
            company=company,
            year=year,
        )
    else:
        # News Article
        return chunk_documents(
            documents=documents,
            splitter_config=NEWS_ARTICLE_SPLITTER,
            chunk_creator=create_news_chunk,
            file_name=file_name,
        )


def load_all_documents(data_path: str = DATA_PATH) -> List[Document]:
    """
    Load and process all documents from data directory.
    
    Returns list of all chunks with metadata.
    """
    file_paths = collect_file_paths(data_path)
    print(f"Found {len(file_paths)} documents to process.")

    all_chunks = []

    for path in tqdm(file_paths, desc="Processing"):
        try:
            chunks = process_file(path)
            all_chunks.extend(chunks)
            tqdm.write(f"✓ {os.path.basename(path)}: {len(chunks)} chunks")
        except Exception as e:
            tqdm.write(f"✗ {os.path.basename(path)}: {e}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


# =============================================================================
# VECTOR STORE
# =============================================================================


def create_vector_store() -> Chroma:
    """
    Create or load the vector store, processing any new documents.
    
    - If store doesn't exist: processes all documents
    - If store exists: checks for new files and only processes those
    """
    print("Initializing vector store...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Get all current files in data directory
    all_files = collect_file_paths(DATA_PATH)
    
    # Load tracking of already processed files
    processed_files = load_processed_files()

    # Check if vector store exists
    if os.path.exists(VECTOR_DB_PATH):
        print("✓ Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings,
            collection_name="documents",
        )
        
        # Find new files
        new_files = get_new_files(all_files, processed_files)
        
        if not new_files:
            print("✓ No new documents to process.")
            return vector_store
        
        print(f"\n📄 Found {len(new_files)} new document(s) to process...")
        
        # Process only new files
        new_chunks = []
        for path in tqdm(new_files, desc="Processing new files"):
            try:
                chunks = process_file(path)
                new_chunks.extend(chunks)
                processed_files.add(os.path.normpath(path)) # Mark as processed
                tqdm.write(f"✓ {os.path.basename(path)}: {len(chunks)} chunks") # Log success
            except Exception as e:
                tqdm.write(f"✗ {os.path.basename(path)}: {e}")
        
        if new_chunks:
            print(f"\n⏳ Embedding {len(new_chunks)} new chunks...")
            for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Embedding"):
                batch = new_chunks[i : i + BATCH_SIZE]
                vector_store.add_documents(batch)
            
            # Save updated tracking
            save_processed_files(processed_files)
            print(f"✓ Added {len(new_chunks)} chunks to vector store.")
        
        return vector_store

    # Create new store - process all documents
    print("Creating new vector store...")
    chunks = load_all_documents()

    if not chunks:
        raise ValueError("No documents found. Check your data directory.")

    print(f"\n⏳ Embedding {len(chunks)} chunks...")

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
        collection_name="documents",
    )

    # Batch insert
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
        batch = chunks[i : i + BATCH_SIZE]
        vector_store.add_documents(batch)

    # Track all files as processed
    for path in all_files:
        processed_files.add(os.path.normpath(path))
    save_processed_files(processed_files)

    print(f"\n✓ Ingestion complete! Saved to {VECTOR_DB_PATH}")
    return vector_store
