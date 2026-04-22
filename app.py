import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# UI
st.title("🤖 RAG-Based AI Chatbot")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success("✅ PDF processed successfully!")

    query = st.text_input("Ask a question")

    if query:
        results = vectorstore.similarity_search(query, k=3)

        context = " ".join([doc.page_content for doc in results])

        # SIMPLE ANSWER (NO HF PIPELINE → NO ERRORS)
        answer = context[:500]

        st.subheader("Answer")
        st.success(answer)
