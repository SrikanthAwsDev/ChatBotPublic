from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from transformers import pipeline
import uvicorn

app = FastAPI()

# Qdrant Config
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.X-iuZOhPdH6BoPpwhKJlqjxYaNIzrPkRyhRFyA20BtM"
QDRANT_URL = "https://b2e451a6-90ae-4db2-a715-4894c90663c0.us-east4-0.gcp.cloud.qdrant.io"
COLLECTION_NAME = "streamlitCluster"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vectorstore = Qdrant(
    client=qdrant_client,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME
)

@app.post("/rag-answer")
async def rag_answer(request: Request):
    data = await request.json()
    question = data.get("question")
    matched_docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in matched_docs])
    answer = qa_pipeline(question=question, context=context)
    return JSONResponse(content={"answer": answer["answer"]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
