import json
from typing import Any

import httpx
import msgpack

from app.config import settings
from app.models import SearchHit


class EndeeClient:
    def __init__(self) -> None:
        self.base_url = settings.endee_base_url.rstrip("/")
        self.index_name = settings.endee_index_name
        self._headers = {"Content-Type": "application/json"}
        if settings.endee_auth_token:
            self._headers["Authorization"] = settings.endee_auth_token

    async def _get(self, path: str) -> httpx.Response:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers)
            response.raise_for_status()
            return response

    async def _post(self, path: str, payload: dict[str, Any] | list[dict[str, Any]]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}{path}",
                headers=self._headers,
                content=json.dumps(payload),
            )
            response.raise_for_status()
            return response

    async def health(self) -> dict[str, Any]:
        response = await self._get("/api/v1/health")
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return {"status": response.text}

    async def list_indexes(self) -> list[str]:
        response = await self._get("/api/v1/index/list")
        data: Any
        if response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
        else:
            data = response.text

        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict):
            for key in ("indexes", "index_list", "data", "result"):
                value = data.get(key)
                if isinstance(value, list):
                    names: list[str] = []
                    for item in value:
                        if isinstance(item, str):
                            names.append(item)
                        elif isinstance(item, dict):
                            name = item.get("name") or item.get("index_name")
                            if isinstance(name, str):
                                names.append(name)
                    if names:
                        return names
        return []

    async def ensure_index(self, dim: int) -> None:
        indexes = await self.list_indexes()
        index_exists = any(
            name == self.index_name or name.endswith(f"/{self.index_name}") for name in indexes
        )
        if index_exists:
            return

        payload = {
            "index_name": self.index_name,
            "dim": dim,
            "space_type": settings.endee_space_type,
            "M": 16,
            "ef_con": 200,
            "sparse_model": "None",
        }

        try:
            await self._post("/api/v1/index/create", payload)
        except httpx.HTTPStatusError as exc:
            # Endee returns 409 when index already exists.
            if exc.response.status_code == 409:
                return
            raise

    async def insert_vectors(self, vectors: list[dict[str, Any]]) -> None:
        if not vectors:
            return
        await self._post(f"/api/v1/index/{self.index_name}/vector/insert", vectors)

    @staticmethod
    def _decode_meta(meta: Any) -> str:
        if isinstance(meta, bytes):
            return meta.decode("utf-8", errors="ignore")
        if isinstance(meta, str):
            return meta
        if isinstance(meta, list):
            try:
                return bytes(meta).decode("utf-8", errors="ignore")
            except Exception:
                return ""
        return ""

    @staticmethod
    def _parse_filter(filter_value: Any) -> dict[str, Any]:
        if isinstance(filter_value, str) and filter_value.strip():
            try:
                parsed = json.loads(filter_value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {"raw_filter": filter_value}
        return {}

    async def search(self, query_vector: list[float], k: int) -> list[SearchHit]:
        payload = {"vector": query_vector, "k": k, "include_vectors": False}
        response = await self._post(f"/api/v1/index/{self.index_name}/search", payload)

        unpacked = msgpack.unpackb(response.content, raw=False, strict_map_key=False)

        hits: list[SearchHit] = []

        # Endee commonly returns a list-of-lists in MessagePack format:
        # [similarity, id, meta, filter, norm, vector]
        if isinstance(unpacked, list):
            for item in unpacked:
                if isinstance(item, dict):
                    hits.append(
                        SearchHit(
                            id=str(item.get("id", "")),
                            similarity=float(item.get("similarity", 0.0)),
                            text=self._decode_meta(item.get("meta", b"")),
                            metadata=self._parse_filter(item.get("filter", "")),
                        )
                    )
                elif isinstance(item, list) and len(item) >= 4:
                    hits.append(
                        SearchHit(
                            id=str(item[1]),
                            similarity=float(item[0]),
                            text=self._decode_meta(item[2]),
                            metadata=self._parse_filter(item[3]),
                        )
                    )
        elif isinstance(unpacked, dict):
            bucket = unpacked.get("results") or unpacked.get("dense")
            if isinstance(bucket, list):
                for rec in bucket:
                    if isinstance(rec, dict):
                        hits.append(
                            SearchHit(
                                id=str(rec.get("id", "")),
                                similarity=float(rec.get("similarity", 0.0)),
                                text=self._decode_meta(rec.get("meta", b"")),
                                metadata=self._parse_filter(rec.get("filter", "")),
                            )
                        )

        return hits
