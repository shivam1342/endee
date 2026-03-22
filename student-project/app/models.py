from typing import Any

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    source_name: str = Field(default="manual_input")
    content: str


class SearchRequest(BaseModel):
    query: str
    k: int | None = None


class SearchHit(BaseModel):
    id: str
    similarity: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class RAGRequest(BaseModel):
    query: str
    k: int | None = None


class RAGResponse(BaseModel):
    query: str
    answer: str
    citations: list[SearchHit]
