import json
import os
import faiss
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================
FAISS_INDEX_PATH = "data/recipes_faiss.index"
METADATA_PATH = "data/recipes_metadata.pkl"

TOP_K = 20
OLLAMA_MODEL = "phi"
OLLAMA_URL = "http://localhost:11434/api/generate"

# BLEU logging
LOG_DIR = "evaluation/logs"
LOG_PATH = os.path.join(LOG_DIR, "phi_outputs.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

# ================= LOAD MODELS =================
print("Loading embedding model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading recipe metadata...")
df = pd.read_pickle(METADATA_PATH)

# Ensure dataframe index aligns with FAISS IDs
df = df.reset_index(drop=True)

print("✅ RAG system ready\n")

# ================= OLLAMA CALL =================
def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        return f"[OLLAMA HTTP ERROR {response.status_code}] {response.text}"

    data = response.json()

    # Safe extraction
    if "response" in data and data["response"].strip():
        return data["response"].strip()

    if "error" in data:
        return f"[OLLAMA ERROR] {data['error']}"

    return "[ERROR] Empty or unexpected Ollama response]"


# ================= RETRIEVE =================
def retrieve(query, top_k=TOP_K):
    q_emb = embed_model.encode([query])
    _, indices = index.search(np.array(q_emb), top_k)

    retrieved_docs = []
    for idx in indices[0]:
        row = df.iloc[int(idx)]
        retrieved_docs.append(
            f"Title: {row['title']}\n{row['document']}"
        )

    return retrieved_docs

# ================= INTERACTIVE LOOP =================
def interactive_chat():
    print("🔹 Interactive RAG with LLaMA‑3")
    print("🔹 Type 'exit' to quit\n")

    while True:
        query = input("Enter your Query: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting RAG chat")
            break

        retrieved_docs = retrieve(query)

        context = "\n\n".join(
            f"- {doc}" for doc in retrieved_docs
        )

        prompt = f"""
You are a helpful cooking assistant.

Your task:
- Answer ONLY using the recipe information provided below.
- Focus ONLY on recipes that clearly match the user query.
- Ignore unrelated or irrelevant recipes.
- Do NOT invent ingredients or steps.

If information is missing, say:
"That detail is not mentioned in the provided recipes."

Recipes:
{context}

User request:
{query}

Respond in clear, simple, human-readable language.
Use bullet points only if the user explicitly asks for them.
"""

        answer = call_ollama(prompt)

        # ===== BLEU LOGGING =====
        log_entry = {
            "query": query,
            "reference": context,
            "generated": answer
        }

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        print("\n🤖 Answer:\n")
        print(answer)
        print("\n" + "-" * 70 + "\n")

# ================= RUN =================
if __name__ == "__main__":
    interactive_chat()
