from fastapi import FastAPI, Query
from pydantic import BaseModel
from app.rag import retrieve_chunks, generate_answer

app = FastAPI()

class QueryInput(BaseModel):
    query: str
    role: str = "user"

@app.post("/query")
def query_bot(input: QueryInput):
    retrieved = retrieve_chunks(input.query)

    if input.role.lower() == "manager":
        retrieved = [chunk for chunk in retrieved if "manager" in chunk[0].lower()]

    response = generate_answer(input.query, retrieved)
    sources = [chunk[1]['source'] for chunk in retrieved]
    return {"response": response, "sources": list(set(sources))}
