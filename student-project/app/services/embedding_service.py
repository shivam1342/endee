from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self._model = SentenceTransformer(settings.embedding_model)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vec.tolist() for vec in vectors]

    def embed_query(self, query: str) -> list[float]:
        vector = self._model.encode([query], normalize_embeddings=True)[0]
        return vector.tolist()
