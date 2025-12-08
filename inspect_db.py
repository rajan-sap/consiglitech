"""
Script to inspect ChromaDB vector store.
"""

import chromadb
import numpy as np


def inspect_chroma_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get the main collection
    collections = client.list_collections()
    print(f"Collections: {[c.name for c in collections]}\n")
    
    for col in collections:
        print(f"{'='*50}")
        print(f"Collection: {col.name}")
        print(f"Total documents: {col.count()}")
        print(f"{'='*50}")
        
        if col.count() == 0:
            continue
        
        # Get sample with all data
        sample = col.get(
            limit=5,
            include=["documents", "metadatas", "embeddings"]
        )
        
        # Show embedding info
        if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:
            emb = np.array(sample["embeddings"][0])
            print(f"\nEmbedding dimensions: {len(emb)}")
            print(f"Embedding norm: {np.linalg.norm(emb):.4f}")
        
        # Helper to format metadata in consistent order
        def format_metadata(meta):
            key_order = ["file_name", "document_type", "company", "year", "page_number"]
            lines = []
            for k in key_order:
                if k in meta:
                    lines.append(f"      {k}: {meta[k]}")
            return "\n".join(lines)
        
        # Show 5 random chunks with metadata
        import random
        total = col.count()
        if total > 0:
            # Get all IDs by fetching all documents
            all_docs = col.get(
                limit=total,
                include=["documents"]
            )
            all_ids = all_docs["ids"]
            sample_size = min(5, len(all_ids))
            sampled_ids = random.sample(all_ids, sample_size)
            print(f"\n--- 5 Random Chunks ---")
            random_sample = col.get(
                ids=sampled_ids,
                include=["documents", "metadatas"]
            )
            for i in range(len(random_sample["documents"])):
                print(f"\n[Random {i+1}] Metadata:")
                print(format_metadata(random_sample['metadatas'][i]))
                text = random_sample["documents"][i]
                print(f"    Text ({len(text)} chars):")
                print(f"    {text}")


if __name__ == "__main__":
    inspect_chroma_db()
