import streamlit as st
from transformers import pipeline
from scripts.rag_chain import retrieve_recipes

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="🍽️ Recipe Recommendation Assistant",
    layout="centered"
)

st.title("🍽️ Recipe Recommendation Assistant (Google FLAN)")
st.write(
    "Local recipe search using FAISS + Google FLAN-T5"
)

# ---------------- Load FLAN (cached) ----------------
@st.cache_resource
def load_flan():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

flan = load_flan()

# ---------------- User Input ----------------
query = st.text_input(
    "Enter your query (e.g., 'ingredients for chicken biryani'):"
)

# ---------------- Search ----------------
if st.button("Search Recipe"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving recipes..."):
            docs = retrieve_recipes(query)

        st.subheader("📋 Top Matching Recipes")

        if not docs:
            st.info("No matching recipes found.")
        else:
            for i, doc in enumerate(docs, 1):
                st.markdown(f"### {i}. {doc.metadata.get('title','Unknown Recipe')}")
                st.write(doc.page_content)
                st.markdown("---")
