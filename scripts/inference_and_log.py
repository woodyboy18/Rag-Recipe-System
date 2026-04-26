import faiss
import pickle
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================
FAISS_INDEX_PATH = "data/faiss.index"
DOCSTORE_PATH = "data/documents.pkl"
OUTPUT_LOG_PATH = "evaluation/logs/llama3_outputs.jsonl"

TOP_K = 5
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"

# ================= LOAD MODELS =================
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ================= LOAD INDEX & DOCS =================
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading documents...")
with open(DOCSTORE_PATH, "rb") as f:
    documents = pickle.load(f)

# ================= OLLAMA CALL =================
def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]

# ================= RETRIEVE =================
def retrieve(query, top_k=TOP_K):
    q_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(q_emb), top_k)
    return [documents[i] for i in indices[0]]

# ================= MAIN LOOP =================
def run_evaluation(queries):
    with open(OUTPUT_LOG_PATH, "w", encoding="utf-8") as f:
        for q in queries:
            retrieved_docs = retrieve(q)

            context = "\n\n".join(
                doc.page_content for doc in retrieved_docs
            )

            prompt = f"""
You are a cooking assistant.

Use ONLY the information below to answer.

{context}

Question:
{q}
"""

            generated = call_ollama(prompt)

            record = {
                "query": q,
                "reference": retrieved_docs[0].page_content,
                "generated": generated
            }

            f.write(json.dumps(record) + "\n")

            print(f"✔ Logged output for query: {q}")

# ================= RUN =================
if __name__ == "__main__":
    sample_queries = [
        "chicken biryani recipe",
        "paneer curry",
        "lemon dessert",
        "vegetarian soup",
        "tofu recipes"
    ]

    run_evaluation(sample_queries)
    print("✅ Logging completed for BLEU evaluation")
