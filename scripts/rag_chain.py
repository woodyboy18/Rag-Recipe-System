import pandas as pd
import faiss

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.fake import FakeListLLM

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import TOP_K


# ================== EMBEDDINGS ==================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ================== LOAD METADATA ==================
df = pd.read_pickle("data/recipes_metadata.pkl")

documents = [
    Document(
        page_content=row["document"],
        metadata={"title": row["title"]}
    )
    for _, row in df.iterrows()
]


# ================== LOAD FAISS INDEX ==================
faiss_index = faiss.read_index("data/recipes_faiss.index")

# ✅ THIS IS THE CRITICAL FIX
docstore = InMemoryDocstore(
    {str(i): documents[i] for i in range(len(documents))}
)

vectorstore = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id={i: str(i) for i in range(len(documents))}
)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ================== PROMPT ==================
prompt = ChatPromptTemplate.from_template("""
You are a cooking assistant.

Use ONLY the information from the recipes below.
If the user asks for vegetarian food, ignore any recipe
containing meat, chicken, fish, or eggs.

Recipes:
{context}

User Question:
{question}

Task:
Give a clean, accurate list of required ingredients.
""")


# ================== TEMP LLM ==================
llm = FakeListLLM(
    responses=["[RAG OK] Retrieval done"]
)


# ================== LCEL RAG CHAIN ==================
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# ================== RUN ==================
if __name__ == "__main__":
    query = input("Enter your query: ")
    result = rag_chain.invoke(query)

    print("\nANSWER:\n")
    print(result)
