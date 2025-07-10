import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CHROMA_PATH = "vectorstore/chroma"

def retrieve_chunks(query, k=3):
    client = chromadb.Client(Settings(
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))
    collection = client.get_or_create_collection(name="rag")

    query_embedding = EMBED_MODEL.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    chunks = list(zip(results['documents'][0], results['metadatas'][0]))
    return chunks

def generate_answer(query, retrieved_chunks):
    context = "\n".join([f"[{i+1}] {chunk[0]}" for i, chunk in enumerate(retrieved_chunks)])
    prompt = f"""
Answer the question using the context below. Include source numbers in brackets.

Context:
{context}

Question: {query}
Answer:
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )

    return response['choices'][0]['message']['content']
