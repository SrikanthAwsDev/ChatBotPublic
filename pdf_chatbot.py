import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import torch

st.set_page_config(page_title="PDF Chatbot (HF)", layout="wide")
st.title("📄🤗 Chat with your PDF using Hugging Face")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_file is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Step 2: Load & Split
    loader = PyMuPDFLoader("uploaded.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Step 3: Embed with HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Qdrant.from_documents(docs, embedding=embeddings, url="http://localhost:6333", collection_name="my_docs")

    # Step 4: Hugging Face QA Pipeline
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if torch.cuda.is_available() else -1)

    # Step 5: User Query Input
    user_input = st.text_input("Ask something about the Ingestion Models 👇")
    if user_input:
        # Step 6: Retrieve Relevant Context
        docs_and_scores = vectorstore.similarity_search_with_score(user_input, k=3)
        context = "\n".join([doc.page_content for doc, score in docs_and_scores])

        # Step 7: Ask the QA model
        with st.spinner("🤔 Thinking..."):
            result = qa_model(question=user_input, context=context)
            st.write("💡", result["answer"])

    st.info("Example questions: \n- What is IAM Role Anywhere?\n- What details needed for SFTP?")
else:
    st.warning("Please upload a PDF file to begin.")
