# generation/generation.py
import os
from openai import OpenAI
from retrieval.utils import decompose_query
from retrieval.utils import extract_metadata_from_query
from retrieval.retriever import HybridRetriever

def generate_answer(question: str, k: int = 5) -> str:
    """
    Full RAG generation pipeline:
    1. Decompose the question into sub-questions.
    2. Retrieve top-k relevant chunks for each sub-question.
    3. Implement metadata based filtering post-retrieval.
    4. Aggregate context.
    5. Generate final answer using LLM.
    """
    # 1. Decompose the question
    sub_questions = decompose_query(question)
    retriever = HybridRetriever()
    relevant_chunks = []

    # 2. Retrieve top-k chunks for each sub-question

    for sub_q in sub_questions:
        results = retriever.search(sub_q, k=k)

        # Post-retrieval metadata filtering of query
        metadata = extract_metadata_from_query(sub_q)

        filter_dict = {k: v for k, v in metadata.items() if v is not None}

        if filter_dict:
            # Require both 'year' and 'company' to match
            required_keys = ["year", "company"]
            filtered_results = [
                r for r in results
                if all(r["metadata"].get(k) == filter_dict.get(k) for k in required_keys if k in filter_dict)
            ]
            if filtered_results:
                best_chunk = max(filtered_results, key=lambda x: x["cosine_similarity"])
                relevant_chunks.append(best_chunk["document"])
                continue
        # Fallback: use best chunk from all results
        import pdb; pdb.set_trace()
        if results:
            best_chunk = max(results, key=lambda x: x["cosine_similarity"])
            relevant_chunks.append(best_chunk["document"])

    # 3. Aggregate context
    aggregated_context = "\n\n".join(relevant_chunks)

    # 4. Generate answer using LLM

    prompt = (
        f"Given the following context chunks, answer the original question as accurately as possible.\n"
        f"Original Question: {question}\n"
        f"Context Chunks:\n{aggregated_context}"
    )

    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions using provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    answer = response.choices[0].message.content.strip()
    return answer