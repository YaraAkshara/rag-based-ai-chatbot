import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# ===== UI =====
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG-based AI Chatbot")
st.write("Upload a PDF and ask questions!")

# ===== Upload PDF =====
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # ===== Load PDF =====
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # ===== Split Text =====
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # ===== Embeddings =====
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ===== Vector Store =====
    vectorstore = FAISS.from_documents(docs, embeddings)

    st.success("✅ PDF processed successfully!")

    # ===== Load Model =====
    generator = pipeline("text-generation", model="distilgpt2")

    # ===== User Input =====
    query = st.text_input("Ask a question:")

    if query:
        # Retrieve relevant docs
        results = vectorstore.similarity_search(query, k=3)
        context = " ".join([doc.page_content for doc in results])

        # Prompt
        prompt = f"""
        Answer the question based on context.

        Context:
        {context}

        Question: {query}
        Answer:
        """

        # Generate answer
        output = generator(prompt, max_new_tokens=100)[0]["generated_text"]

        st.subheader("🤖 Answer")
        st.write(output)
