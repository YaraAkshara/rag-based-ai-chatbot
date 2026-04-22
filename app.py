import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# ===== PAGE SETTINGS =====
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG-Based AI Chatbot")
st.write("Upload a PDF and ask questions!")

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

if uploaded_file is not None:

    # Save file temporarily
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

    # ===== LOAD MODEL =====
    generator = pipeline(
        "text-generation",
        model="gpt2"
    )

    # ===== USER INPUT =====
    query = st.text_input("💬 Ask a question:")

    if query:

        # Retrieve relevant chunks
        results = vectorstore.similarity_search(query, k=3)
        context = " ".join([doc.page_content for doc in results])

        # ===== BETTER PROMPT =====
        prompt = f"""
You are an intelligent assistant.

Answer the question clearly and briefly in 2-3 sentences.

Context:
{context}

Question: {query}

Answer:
"""

        # ===== GENERATE ANSWER =====
        output = generator(
            prompt,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.3
        )[0]["generated_text"]

        # ===== CLEAN OUTPUT =====
        if "Answer:" in output:
            output = output.split("Answer:")[-1].strip()

        # ===== DISPLAY =====
        st.subheader("🤖 Answer")
        st.write(output)
