import streamlit as st
import requests

st.set_page_config(page_title="GDP Handbook Chatbot", layout="wide")
st.title("📄 Welcome to GDP Handbook Chatbot. Ask away!!")

user_input = st.text_input("Ask a question:")
if user_input:
    response = requests.post("http://localhost:8000/rag-answer", json={"question": user_input})
    if response.status_code == 200:
        st.write("💡", response.json()["answer"])
    else:
        st.error("Failed to get response from RAG API.")

st.info("Ask: 'What are ingestion models?', 'How to set up IAM Role Anywhere?', etc.")
