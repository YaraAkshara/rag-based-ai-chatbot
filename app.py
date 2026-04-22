import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# ===== PAGE =====
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG-Based AI Chatbot")
st.write("Upload a PDF and ask questions!")

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # ===== LOAD PDF =====
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # ===== SPLIT TEXT =====
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # ===== EMBEDDINGS =====
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ===== VECTOR STORE =====
    vectorstore = FAISS.from_documents(docs, embeddings)

    st.success("✅ PDF processed successfully!")

    # ===== LOAD QA MODEL (IMPORTANT CHANGE) =====
    qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    tokenizer="distilbert-base-cased-distilled-squad"
    )
    # ===== USER INPUT =====
    query = st.text_input("💬 Ask a question:")

    if query:

        # Retrieve context
        results = vectorstore.similarity_search(query, k=3)
        context = " ".join([doc.page_content for doc in results])

        # ===== GET ANSWER =====
        result = qa_pipeline(
            question=query,
            context=context
        )

        answer = result["answer"]

        # ===== DISPLAY =====
        st.subheader("🤖 Answer")
        st.write(answer)
    
