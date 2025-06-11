import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from transformers import pipeline

# UI Setup
st.set_page_config(page_title="Qdrant PDF Chatbot", layout="wide")
st.title("📄 Chat with your PDF (Qdrant Cloud + HF)")

# Qdrant Cloud Config
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.e3jaGBypE5dqsvrdU8LREwePXV0HcKRHACd2BV55764"
QDRANT_URL = "https://0a3a6878-eae4-4415-b1da-8c39a6cc5817.us-east4-0.gcp.cloud.qdrant.io"
COLLECTION_NAME = "pdf_chunks"

# Setup Embedding Model & Client
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Upload and Process PDF
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyMuPDFLoader("uploaded.pdf")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Build or connect vectorstore
    vectorstore = Qdrant.from_documents(
        docs,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )

    # QA Model
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # Query
    user_input = st.text_input("Ask a question:")
    if user_input:
        matched_docs = vectorstore.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in matched_docs])
        answer = qa_pipeline(question=user_input, context=context)
        st.write("💡", answer["answer"])

    st.info("Ask: 'What are ingestion models?', 'How to set up IAM Role Anywhere?', etc.")
else:
    st.warning("Upload a PDF to get started.")
