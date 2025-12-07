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
        
        # Show sample documents with metadata
        print(f"\n--- First 3 Chunks ---")
        first_sample = col.get(
            limit=3,
            include=["documents", "metadatas"]
        )
        for i in range(len(first_sample["documents"])):
            print(f"\n[{i+1}] Metadata:")
            print(format_metadata(first_sample['metadatas'][i]))
            text = first_sample["documents"][i]
            print(f"    Text ({len(text)} chars):")
            print(f"    {text}")
        
        # Get last 3 chunks
        print(f"\n--- Last 3 Chunks ---")
        all_ids = col.get(include=[])["ids"]
        last_ids = all_ids[-3:]
        last_sample = col.get(
            ids=last_ids,
            include=["documents", "metadatas"]
        )
        for i in range(len(last_sample["documents"])):
            idx = col.count() - 3 + i + 1
            print(f"\n[{idx}] Metadata:")
            print(format_metadata(last_sample['metadatas'][i]))
            text = last_sample["documents"][i]
            print(f"    Text ({len(text)} chars):")
            print(f"    {text}")


if __name__ == "__main__":
    inspect_chroma_db()
