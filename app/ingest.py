import os
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

CHUNK_SIZE = 500
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
CHROMA_PATH = "vectorstore/chroma"

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i+size] for i in range(0, len(text), size)]

def ingest_documents(folder_path='data'):
    client = chromadb.Client(Settings(
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))
    collection = client.get_or_create_collection(name="rag")

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        ext = os.path.splitext(file)[1].lower()

        if ext == ".pdf":
            content = read_pdf(path)
        elif ext == ".docx":
            content = read_docx(path)
        elif ext == ".txt":
            content = read_txt(path)
        else:
            continue

        chunks = chunk_text(content)
        embeddings = EMBED_MODEL.encode(chunks).tolist()

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = f"{file}_{i}"
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": file}],
                ids=[doc_id]
            )
            
if __name__ == "__main__":
    ingest_documents()
