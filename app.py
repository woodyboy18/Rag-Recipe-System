import streamlit as st
from transformers import pipeline
from scripts.rag_chain import retrieve_recipes

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Recipe Assistant",
    layout="centered"
)

st.title("Recipe Assistant")
st.write(
    "Local recipe search using FAISS + Google FLAN‑T5 "
    "(Semantic + Optional Strict Keyword Matching)"
)

# ---------------- Load FLAN (cached) ----------------
@st.cache_resource
def load_flan():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

flan = load_flan()  # (loaded for future use, not generating yet)

# ---------------- User Controls ----------------
query = st.text_input("Enter your query:")

strict_mode = st.checkbox(
    "Strict keyword match (only show recipes containing the exact word)"
)

top_k = st.slider(
    "Number of recipes to show",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

# ---------------- Search ----------------
if st.button("Search Recipe"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving recipes..."):
            docs = retrieve_recipes(
                query,
                k=top_k,
                strict=strict_mode
            )

        st.subheader("📋 Top Matching Recipes")

        if not docs:
            st.info("No matching recipes found.")
        else:
            for i, doc in enumerate(docs, 1):
                st.markdown(
                    f"### {i}. {doc.metadata.get('title', 'Unknown Recipe')}"
                )
                st.write(doc.page_content)
                st.markdown("---")


