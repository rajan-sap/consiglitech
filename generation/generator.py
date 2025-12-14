
import os
from openai import OpenAI
from dotenv import load_dotenv
from retrieval.retriever import retrieve_aggregated_context
from retrieval.retriever import Retriever


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment.")
client = OpenAI(api_key=api_key)


retiever = Retriever()

# Step 4: Implementation of answer generation
def generate_answer(query) -> str:
    aggregated_context = retrieve_aggregated_context(query, retiever)
    # Metadata filter is applied during retrieval

    prompt = (
        f"Given the following data as a ground of truth, answer the original question as accurately as possible.\n"
        f"Original Question: {query}\n"
        f"Information:\n{aggregated_context}"
    )
    
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions using provided question and context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    answer = response.choices[0].message.content.strip()
    return answer  

