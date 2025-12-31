import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. Load cleaned dataset
df = pd.read_csv("data/recipes_35k_clean.csv")
documents = df["document"].tolist()

print(f"Loaded {len(documents)} documents")

# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Generate embeddings
embeddings = model.encode(
    documents,
    batch_size=64,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

# 4. Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index built. Total vectors:", index.ntotal)

# 5. Save index and metadata
faiss.write_index(index, "data/recipes_faiss.index")
df.to_pickle("data/recipes_metadata.pkl")

print("Index and metadata saved successfully.")
