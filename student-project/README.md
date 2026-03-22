# Student Notes Semantic Search + RAG with Endee and Groq

This project implements a practical AI workflow for student notes using:

- Endee as the vector database
- FastAPI as the backend API
- Sentence Transformers for text embeddings
- Groq API for grounded answer generation
- Plain HTML/CSS/JS for the frontend (no Streamlit)

## Problem Statement

Students spend a lot of time searching long class notes. Keyword search fails when wording changes. This project solves that by indexing notes as embeddings in Endee, enabling semantic retrieval and retrieval-augmented generation (RAG).

## Key Features

- Text/file ingestion and chunking
- Embedding generation and vector insertion into Endee
- Semantic search endpoint
- RAG endpoint with citations
- Browser UI built only with HTML/CSS/JS

## System Design

1. Ingestion:
   - Notes are split into overlapping chunks
   - Each chunk is embedded with `all-MiniLM-L6-v2`
   - Vectors + chunk text metadata are inserted into Endee

2. Retrieval:
   - Query text is embedded
   - Top-k nearest chunks are fetched from Endee

3. Generation:
   - Retrieved chunks are passed as context to Groq LLM
   - Final answer is generated with source references

## How Endee Is Used

The app calls Endee HTTP APIs directly:

- `POST /api/v1/index/create` to create index
- `POST /api/v1/index/{index_name}/vector/insert` to insert chunks
- `POST /api/v1/index/{index_name}/search` to retrieve nearest chunks
- `GET /api/v1/index/list` and `GET /api/v1/health` for setup/health

Search responses are decoded from MessagePack returned by Endee.

## Project Structure

```text
student-project/
  app/
    main.py
    config.py
    models.py
    services/
      embedding_service.py
      endee_client.py
      rag_service.py
      text_utils.py
    static/
      index.html
  .env.example
  requirements.txt
  run.py
  README.md
```

## Setup and Execution

### 1. Start Endee server

From repository root:

```powershell
mkdir data
$env:NDD_DATA_DIR="./data"
./build/ndd
```

Default Endee URL is `http://127.0.0.1:8080`.

### 2. Create Python environment

```powershell
cd student-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional (better embedding quality, heavier install):

```powershell
pip install sentence-transformers==3.4.1
```

### 3. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set:

- `GROQ_API_KEY=...`
- optionally `ENDEE_AUTH_TOKEN` if Endee auth is enabled

Note: if `sentence-transformers` is not installed, the app automatically uses a lightweight local fallback embedding so the demo still runs.

### 4. Run FastAPI server

```powershell
python run.py
```

App URL: `http://127.0.0.1:8000`

## API Endpoints

- `GET /api/health`
- `POST /api/ingest/text`
- `POST /api/ingest/file` (txt file)
- `POST /api/search`
- `POST /api/rag`

## Example Requests

### Ingest text

```bash
curl -X POST http://127.0.0.1:8000/api/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"source_name":"ml_unit_1","content":"Gradient descent updates model weights..."}'
```

### Semantic search

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is gradient descent?","k":5}'
```

### RAG answer

```bash
curl -X POST http://127.0.0.1:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarize gradient descent in 3 points","k":5}'
```

## Evaluation Notes

- Core vector search is performed in Endee
- Retrieval output is shown in semantic search
- RAG answer cites retrieved source chunks
- The solution is practical, reproducible, and hosted inside the candidate's Endee fork

## Limitations and Future Work

- Current file ingestion is optimized for `.txt` (PDF parsing can be added)
- No automated relevance benchmark yet
- Can be extended with per-user notes, filters, and hybrid sparse+dense search
