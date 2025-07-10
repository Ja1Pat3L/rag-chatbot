# RAG Chatbot System

## Features
- Document ingestion from TXT, PDF, DOCX
- Embedding generation using Sentence Transformers
- ChromaDB-based vector similarity search
- Query answering using OpenAI GPT-3.5
- FastAPI backend

## Getting Started

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Add documents to `data/` folder.

3. Run document ingestion:
```
python app/ingest.py
```

4. Start API server:
```
uvicorn app.api:app --reload
```

## Sample Queries
- "What benefits are offered to employees?"
- "Summarize the hiring process."

## Notes
- Ensure OpenAI API key is set in environment:
```
export OPENAI_API_KEY=your_key_here
```

## Structure
- `app/`: Core logic
- `data/`: Documents
- `vectorstore/`: Stored ChromaDB index

##  License
MIT
