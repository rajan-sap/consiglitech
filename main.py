import os
from generation.generator import generate_answer
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from ingestion.ingest import create_vector_store



def main():
    print("Is there any new document to add to the vector store? (y/n): ", end="")
    # Update vector store if new documents are to be added
    user_input = input().strip().lower()
    if user_input == 'y':
        create_vector_store()
    
    else:
        print("Notthing to update.")     
        
    # Interactive query loop
    while True:
        print("\nEnter your query (or type 'exit' to quit): ", end="")
        query = input().strip() 
        if query.lower() == 'exit':
            print("Exiting the RAG system. Goodbye!")
            break
        elif query:
            answer = generate_answer(query)
            print(f"\nAnswer:\n{answer}\n") 

if __name__ == "__main__":
    main()
    
