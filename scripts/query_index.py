import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load FAISS index and metadata
index = faiss.read_index("data/recipes_faiss.index")
df = pd.read_pickle("data/recipes_metadata.pkl")

# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Query function
def search_recipes(query, top_k=5):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return df.iloc[indices[0]]

# 4. Test query
query = "ingredients for chicken biryani"
results = search_recipes(query)

print("\nTop matching recipes:\n")
for i, row in results.iterrows():
    print("Recipe:", row['title'])
    print("Ingredients:", row['ingredients'][:300])
    print("-" * 60)
