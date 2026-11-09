from fastapi import FastAPI
from pydantic import BaseModel
from rag.rag import retrived_documents, generate_answer

app = FastAPI(
    title="Employee RAG Assistant",
    description="Retrieves relevant policy info from a vector DB and passes it to an LLM to answer.",
)


class QueryInput(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Welcome to Employee policy RAG system"}


@app.post("/response")
def response(body: QueryInput):
    context = retrived_documents(body.query)
    answer = generate_answer(body.query, context)
    return answer