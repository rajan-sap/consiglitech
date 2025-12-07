from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from ingestion.ingest import create_vector_store


if __name__ == "__main__":
    print("Starting RAG System Orchestration...")
    
    # 1. Ensure Vector Store is ready
    vector_store = create_vector_store()
    
    print("Vector Store is initialized. Ready for Retrieval and Generation.")

    
    # 2. Proceed to initialize the Retrieval/Generation pipeline...
    # ...