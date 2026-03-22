# NeuralDocs-Endee

Lightweight Retrieval-Augmented Generation (RAG) system using Endee vector database and Groq LLM for semantic document search.

## Overview
Semantic document retrieval system using Endee vector database.

## Problem Statement
Traditional keyword-based search fails to capture semantic intent across large unstructured documents.

NeuralDocs-Endee implements embedding-based semantic retrieval using a vector database pipeline and Retrieval-Augmented Generation (RAG) to enable context-aware question answering over documents.

## Features
- Dense embedding-based semantic document search
- Containerized Endee vector database deployment
- Top-k similarity retrieval pipeline
- Groq LLM integration for RAG-based answering
- REST API ingestion and query interface
- Source citation tracking in generated responses

## Architecture
Document
  -> Chunking (fixed-size overlapping character segments: 550 chars, 120 overlap)
  -> Embedding generation (MiniLM 384-dim when available)
  -> Vector indexing (Endee)
  -> Top-k similarity retrieval
  -> Prompt augmentation
  -> Groq LLM answer generation

User Query
  -> Query Embedding
  -> Top-k Similarity Retrieval (Endee)
  -> Context Assembly
  -> Prompt Augmentation
  -> Groq LLM Response Generation

## Endee Usage
Endee is used as the vector storage and similarity search engine.

The backend uses Endee APIs for:
- index creation
- vector insertion
- top-k similarity search

## Embeddings
Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) when installed.

If `sentence-transformers` is unavailable, the app uses a lightweight local fallback embedding method so the system still runs.

## Why Endee?
Endee provides a lightweight containerized vector database with API-first indexing and retrieval support, making it suitable for portable semantic search systems and rapid experimentation with RAG pipelines.

## Retrieval Strategy
Top-k similarity search is performed over indexed embeddings stored in Endee (default space type: cosine). Retrieved chunks are injected into prompt context before LLM response generation.

## Setup
1. Clone your fork and open repository root.
2. Start Docker Desktop.
3. Run Endee container:

```powershell
docker rm -f endee-server
docker run -d --ulimit nofile=100000:100000 -p 18080:8080 -v "${PWD}/endee-data:/data" --name endee-server --restart unless-stopped endeeio/endee-server:latest
```

4. Create and activate Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r student-project/requirements.txt
```

5. Create local env file:

```powershell
Copy-Item student-project/.env.template student-project/.env
```

6. Edit `student-project/.env`:
- `ENDEE_BASE_URL=http://127.0.0.1:18080`
- `GROQ_API_KEY=<your_valid_key>`

## Run
From repository root:

```powershell
.\.venv\Scripts\python.exe student-project/run.py
```

Open:
`http://127.0.0.1:8000`

## Example Output
Ingest response:

```json
{
  "status": "indexed",
  "source": "smoke",
  "chunks_indexed": 1,
  "embedding_dimension": 384
}
```

Search response:

```json
{
  "query": "What is gradient descent?",
  "hits": [
    {
      "id": "smoke-chunk-1",
      "similarity": 0.1767,
      "text": "Gradient descent updates model parameters to minimize loss.",
      "metadata": {"source": "smoke", "chunk": 1}
    }
  ]
}
```

RAG response:

```json
{
  "query": "Explain gradient descent in simple words",
  "answer": "Gradient descent updates model parameters to minimize loss...",
  "citations": [
    {"id": "smoke-chunk-1"}
  ]
}
```

## Limitations
- Uses fixed-size chunking instead of semantic chunk boundaries
- No reranking stage implemented
- No hybrid keyword + vector retrieval pipeline

## Future Improvements
- Add cross-encoder reranking stage
- Support multi-document ingestion workflows
- Add hybrid BM25 + dense retrieval
- Implement streaming response support


