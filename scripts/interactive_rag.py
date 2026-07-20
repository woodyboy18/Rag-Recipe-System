import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import json
import re
import faiss
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from memory.retrieval_cache import save_retrieval_cache

from query.query_rewritter import rewrite_query

from memory.conversation_memory import save_turn
from memory.memory_retriever import retrieve_memory
from constraints.constraint_extractor import extract_constraints

from constraints.constraint_extractor import extract_constraints
from substitution.ingredient_filter import filter_recipes
# ================= CONFIG =================
FAISS_INDEX_PATH = "data/recipes_faiss.index"
METADATA_PATH = "data/recipes_metadata.pkl"
TOP_K = 20 #tokens
OLLAMA_MODEL = "qwen:0.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"

LOG_DIR = "evaluation/logs"
LOG_PATH = os.path.join(LOG_DIR, "qwen_outputs.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

# ================= CLEAN =================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ================= LOAD =================
print("Loading embedding model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata...")
df = pd.read_pickle(METADATA_PATH).reset_index(drop=True)

from rank_bm25 import BM25Okapi

print("Building BM25 index...")

tokenized_corpus = [clean_text(doc).split() for doc in df["document"]]
bm25 = BM25Okapi(tokenized_corpus)

print("RAG Ready\n")

# ================= OLLAMA =================
def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 200
        }
    }

    r = requests.post(OLLAMA_URL, json=payload)
    if r.status_code != 200:
        return ""

    return r.json().get("response", "").strip()

# ================= RETRIEVE =================
def retrieve(query: str):

    # ===== FAISS =====
    emb = embed_model.encode([query])
    faiss_scores, faiss_indices = index.search(np.array(emb), TOP_K)

    # ===== BM25 =====
    tokenized_query = clean_text(query).split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_idx = int(np.argmax(bm25_scores))

    # ===== COMBINE =====
    final_indices = set(faiss_indices[0])
    final_indices.add(bm25_idx)

    docs = []
    indices = []

    for idx in final_indices:
        if int(idx) < len(df):
            docs.append(df.iloc[int(idx)]["document"])
            indices.append(idx)

    return docs, [indices]
# ================= REFERENCE =================
def extract_ingredients(indices):
    ingredients = []

    for idx in indices[0]:
        if int(idx) >= len(df):
            continue

        row = df.iloc[int(idx)]

        if "ingredients" not in row:
            continue

        ing = row["ingredients"]

        if isinstance(ing, str):
            parts = [p.split("(")[0].strip() for p in ing.split("|")]
            for p in parts:
                p = clean_text(p)
                if p and p not in ingredients:
                    ingredients.append(p)

    return " ".join(ingredients)

# ================= CHAT =================
def interactive_chat():

    print("🔹 Interactive RAG")
    print("🔹 type 'exit' to quit\n")

    while True:

        query = input("Enter your Query: ").strip()

        conversation_history = retrieve_memory(query)

        constraints = extract_constraints(query)

        print("\n========== MEMORY ==========")
        print(conversation_history)
        print("============================\n")

        print("\n========== CONSTRAINTS ==========")
        print(constraints)
        print("=================================\n")

        if query.lower() in ["exit","quit"]:
            print("Exiting...")
            break
        

        rewritten_query = rewrite_query(
            query,
            conversation_history,
            constraints
        )

        print("\nRewritten Query:", rewritten_query)

        docs, indices = retrieve(rewritten_query)
        
        save_retrieval_cache(rewritten_query, docs)

        print("\nRetrieved Recipes:")

        for i, doc in enumerate(docs):
            print(f"\nRecipe {i+1}")
            print(doc[:250])

# Apply ingredient filtering
        docs = filter_recipes(
            docs,
            constraints["ingredients_to_avoid"]
        )

# If every recipe is removed
        if len(docs) == 0:
            print("\nNo recipe found after applying ingredient constraints.\n")
            continue

# remove duplicate context chunks
        context = "\n".join(list(dict.fromkeys(docs)))



        prompt = f"""
You are a Memory-Augmented Conversational Culinary Assistant.

Your task is to answer the user's query using ONLY the recipes provided in the CONTEXT.

Previous Conversation:
{conversation_history}

Current User Constraints:
{constraints}

Recipe Context:
{context}

Current User Query:
{query}

Instructions:

1. Generate ONLY One Recipe
2. Always use ONLY the Recipe Context.
3. Do NOT invent ingredients or cooking steps.
4. Do not generate multiple recipes with same title.
5. If the current query modifies a previous request (e.g., "I don't have eggs", "without yogurt", "make it vegetarian"), use only the filtered Recipe Context provided.
6. Respect all extracted constraints such as meal type, diet, cooking time and ingredient restrictions.
7. If no suitable recipe exists in the Recipe Context, reply exactly:
Recipe not found in database.

Return ONLY one complete recipe in the following format.

Title: <recipe title>

Ingredients:
- item
- item

Instructions:
1. 
2. 
3. 

Answer:
"""

        answer = call_ollama(prompt)

# ================= SAVE MEMORY =================
        save_turn(query, answer, constraints)

        # ===== BLEU OPTIMIZED LOGGING =====
        reference_text = extract_ingredients(indices)

        ref_words = set(clean_text(reference_text).split())
        gen_words = clean_text(answer).split()

        filtered = [w for w in gen_words if w in ref_words]
        generated_filtered = " ".join(filtered)

        log_entry = {
            "query": query,
            "reference": clean_text(reference_text),
            "generated": generated_filtered
        }

        with open(LOG_PATH,"a",encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-"*60 + "\n")

# ================= RUN =================
if __name__ == "__main__":
    interactive_chat()
