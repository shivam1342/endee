from __future__ import annotations

import hashlib
import math

import numpy as np

from app.config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self._dim = 384
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embedding_model)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            # Fallback keeps the app runnable even when torch/sentence-transformers
            # are unavailable on the machine.
            self._model = None

    @property
    def dimension(self) -> int:
        return self._dim

    def _fallback_embed(self, text: str) -> list[float]:
        tokens = text.lower().split()
        vector = np.zeros(self._dim, dtype=np.float32)

        for token in tokens:
            h = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self._dim
            sign = -1.0 if int(h[8:10], 16) % 2 else 1.0
            vector[idx] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return [vec.tolist() for vec in vectors]
        return [self._fallback_embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        if self._model is not None:
            vector = self._model.encode([query], normalize_embeddings=True)[0]
            return vector.tolist()
        return self._fallback_embed(query)
