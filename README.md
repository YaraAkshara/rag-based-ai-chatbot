# 🤖 RAG-Based AI Chatbot

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that answers questions based on a PDF document.

It uses **LangChain, FAISS, and HuggingFace Transformers** to retrieve relevant information and generate accurate answers.

## 📌 Features

* 📄 Loads and reads PDF documents
* ✂️ Splits text into chunks
* 🔎 Converts text into embeddings
* 📚 Stores embeddings using FAISS (vector database)
* 🤖 Answers user queries using LLM
* 💬 Interactive question-answer system
  
## 🛠️ Tech Stack

* Python
* LangChain
* FAISS
* HuggingFace Transformers
* PyPDF

## ⚙️ How it Works

1. Load PDF file
2. Split text into smaller chunks
3. Convert chunks into embeddings
4. Store embeddings in FAISS
5. Retrieve relevant chunks for a query
6. Generate answer using language model

## ▶️ How to Run

1. Open the notebook in Google Colab
2. Install dependencies:

```bash
pip install langchain langchain-community faiss-cpu pypdf sentence-transformers transformers
```

3. Upload your PDF file
4. Run all cells
5. Ask questions in the input prompt

## 💡 Example

**Input:**
`What is DBMS?`

**Output:**
`Database Management System`

## 📁 Project Structure

```
rag-based-ai-chatbot/
│
├── rag_chatbot.ipynb
├── README.md
```

## 🚀 Future Improvements

* Add web interface (Streamlit)
* Support multiple PDFs
* Improve answer accuracy
* Deploy as web app

## 👩‍💻 Author

Yara Akshara
