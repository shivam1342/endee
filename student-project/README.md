# Student Notes Semantic Search + RAG with Endee and Groq

This project implements a practical AI workflow for student notes using:

- Endee as the vector database
- FastAPI as the backend API
- Sentence Transformers for text embeddings
- Groq API for grounded answer generation
- Plain HTML/CSS/JS for the frontend (no Streamlit)

## What It Does

- Ingest notes (text or `.txt` file)
- Convert chunks into embeddings and store them in Endee
- Run semantic search on stored notes
- Generate RAG answers with Groq using retrieved context

## Endee Usage

The backend talks to Endee HTTP APIs:

- `POST /api/v1/index/create`
- `POST /api/v1/index/{index_name}/vector/insert`
- `POST /api/v1/index/{index_name}/search`
- `GET /api/v1/health`

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
  .env.template
  requirements.txt
  run.py
  README.md
```

## Run Project

### 1. Start Endee server

From repository root:

```powershell
mkdir data
$env:NDD_DATA_DIR="./data"
./build/ndd
```

Default Endee URL is `http://127.0.0.1:8080`.

### 2. Create Python environment (repo root)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r student-project/requirements.txt
```

Optional (better embedding quality, heavier install):

```powershell
pip install sentence-transformers==3.4.1
```

### 3. Configure environment variables

```powershell
Copy-Item student-project/.env.template student-project/.env
```

Edit `student-project/.env` and set:

- `GROQ_API_KEY=...`
- optionally `ENDEE_AUTH_TOKEN` if Endee auth is enabled

Note: if `sentence-transformers` is not installed, the app automatically uses a lightweight local fallback embedding so the demo still runs.

### 4. Run FastAPI server

```powershell
.\.venv\Scripts\python.exe student-project/run.py
```

App URL: `http://127.0.0.1:8000`

## Quick Test

1. Open `http://127.0.0.1:8000`
2. In "Ingest Notes (Text)", paste notes and click "Index Notes"
3. In "Semantic Search", ask a question from those notes
4. In "RAG Answer", ask the same question and verify citations are shown

## API Test (PowerShell)

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/ingest/text" -ContentType "application/json" -Body '{"source_name":"demo","content":"Gradient descent updates weights iteratively."}'

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/search" -ContentType "application/json" -Body '{"query":"What is gradient descent?","k":3}'

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/rag" -ContentType "application/json" -Body '{"query":"Explain gradient descent in simple words","k":3}'
```
