import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= CONFIG =================
CSV_PATH = "data/recipes.csv"

FAISS_INDEX_PATH = "data/recipes_faiss.index"
METADATA_PATH = "data/recipes_metadata.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 512

# ================= LOAD DATA =================
print("Loading recipes CSV...")
df = pd.read_csv(CSV_PATH)

# Safety: keep only required columns
required_cols = ["Name", "Description", "RecipeInstructions"]
df = df[required_cols].fillna("")

# Build single unstructured document per recipe
print("Building recipe documents...")
df["document"] = (
    "Recipe Name: " + df["Name"] + "\n\n"
    "Description: " + df["Description"] + "\n\n"
    "Instructions: " + df["RecipeInstructions"]
)

# Rename for consistency
df = df.rename(columns={"Name": "title"})

# Reset index so FAISS IDs == DataFrame row IDs
df = df.reset_index(drop=True)

print(f"Total recipes: {len(df)}")

# ================= LOAD EMBEDDING MODEL =================
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# ================= CREATE FAISS INDEX =================
embedding_dim = embed_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)

print("Creating embeddings and building FAISS index...")

for start in tqdm(range(0, len(df), BATCH_SIZE)):
    end = start + BATCH_SIZE
    texts = df["document"].iloc[start:end].tolist()

    embeddings = embed_model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    index.add(embeddings)

# ================= SAVE OUTPUTS =================
print("Saving FAISS index...")
faiss.write_index(index, FAISS_INDEX_PATH)

print("Saving metadata...")
df[["title", "document"]].to_pickle(METADATA_PATH)

print("\n✅ Indexing completed successfully!")
print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
print(f"Metadata saved to: {METADATA_PATH}")
