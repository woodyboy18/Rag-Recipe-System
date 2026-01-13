import pandas as pd
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# ================= CONFIG =================
CSV_PATH = "data/recipes.csv"
FAISS_INDEX_PATH = "data/faiss.index"
DOCSTORE_PATH = "data/documents.pkl"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 1000   # safe for 500k rows

# ================= EMBEDDING MODEL =================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ================= TEXT UTILS =================
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x)

def build_unstructured_text(row):
    """
    Combine multiple columns into ONE unstructured recipe text
    """
    text = f"""
    Recipe Name: {clean_text(row['Name'])}

    Description:
    {clean_text(row['Description'])}

    Ingredients:
    {clean_text(row['RecipeIngredientParts'])}

    Instructions:
    {clean_text(row['RecipeInstructions'])}
    """
    return text.strip()

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ================= BUILD INDEX =================
def build_index():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    required_cols = [
        "Name",
        "Description",
        "RecipeIngredientParts",
        "RecipeInstructions"
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    print(f"Total recipes: {len(df)}")

    documents = []
    all_embeddings = []

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i:i + BATCH_SIZE]

        batch_texts = []
        batch_docs = []

        for idx, row in batch.iterrows():
            raw_text = build_unstructured_text(row)
            chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

            for chunk in chunks:
                batch_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"recipe_id": int(row["ecipeId"])}
                    )
                )
                batch_texts.append(chunk)

        # Embed batch
        embeddings = model.encode(
            batch_texts,
            show_progress_bar=False
        )

        documents.extend(batch_docs)
        all_embeddings.append(embeddings)

    # Stack all embeddings
    embeddings = np.vstack(all_embeddings)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save documents (REFERENCE TEXT FOR BLEU)
    with open(DOCSTORE_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("✅ FAISS index built successfully")
    print(f"Total chunks indexed: {len(documents)}")

# ================= MAIN =================
if __name__ == "__main__":
    build_index()
