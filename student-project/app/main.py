from pathlib import Path
import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models import IngestTextRequest, RAGRequest, RAGResponse, SearchRequest, SearchResponse
from app.services.embedding_service import EmbeddingService
from app.services.endee_client import EndeeClient
from app.services.rag_service import RAGService
from app.services.text_utils import chunk_text


app = FastAPI(title="Endee Student Notes RAG", version="1.0.0")

root_dir = Path(__file__).resolve().parent
static_dir = root_dir / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

embedding_service = EmbeddingService()
endee_client = EndeeClient()
rag_service = RAGService()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
async def health() -> dict:
    endee_health = await endee_client.health()
    return {
        "app": "ok",
        "endee": endee_health,
        "index_name": settings.endee_index_name,
        "embedding_model": settings.embedding_model,
        "groq_enabled": rag_service.enabled,
    }


@app.post("/api/ingest/text")
async def ingest_text(payload: IngestTextRequest) -> dict:
    text = payload.content.strip()
    if not text:
        raise HTTPException(status_code=400, detail="content cannot be empty")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="no chunks generated from content")

    embeddings = embedding_service.embed_texts(chunks)
    await endee_client.ensure_index(dim=embedding_service.dimension)

    vectors = []
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings), start=1):
        vector_id = f"{payload.source_name}-chunk-{idx}"
        vectors.append(
            {
                "id": vector_id,
                "meta": chunk,
                "vector": vector,
                "filter": json.dumps(
                    {
                        "source": payload.source_name,
                        "chunk": idx,
                    }
                ),
            }
        )

    await endee_client.insert_vectors(vectors)

    return {
        "status": "indexed",
        "source": payload.source_name,
        "chunks_indexed": len(vectors),
        "embedding_dimension": embedding_service.dimension,
    }


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")

    payload = IngestTextRequest(source_name=file.filename, content=text)
    return await ingest_text(payload)


@app.post("/api/search", response_model=SearchResponse)
async def semantic_search(payload: SearchRequest) -> SearchResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    k = payload.k or settings.endee_top_k
    query_vector = embedding_service.embed_query(query)
    hits = await endee_client.search(query_vector=query_vector, k=k)

    return SearchResponse(query=query, hits=hits)


@app.post("/api/rag", response_model=RAGResponse)
async def rag_answer(payload: RAGRequest) -> RAGResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    k = payload.k or settings.endee_top_k
    query_vector = embedding_service.embed_query(query)
    hits = await endee_client.search(query_vector=query_vector, k=k)

    answer = rag_service.answer(query=query, contexts=hits)

    return RAGResponse(query=query, answer=answer, citations=hits)
