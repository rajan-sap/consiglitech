import os
from generation.generation import generate_answer
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from ingestion.ingest import create_vector_store

print(f"Starting RAG System Orchestration...")
print(f"Is there any new document to add to the vector store? (y/n): ", end="")

user_input = input().strip().lower()    
if user_input == "y":
    print("Processing new documents and updating vector store...")
    create_vector_store()
    print("Vector store updated with new documents.")
else:
    print("Skipping document ingestion.")

print("Initializing Vector Store for Retrieval/Generation...")
print("✓ Loading or creating vector store...")
print("✓ Embedding model is set up.")

if __name__ == "__main__":
    
    while True:
        user_input = input("Enter your query about documents: ")
        answer = generate_answer(user_input)
        print("\nGenerated Answer:\n")
        print(answer)
        again = input("\nWould you like to query again? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("Goodbye!")
            break