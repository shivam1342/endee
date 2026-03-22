from groq import Groq
from groq import APIError, AuthenticationError

from app.config import settings
from app.models import SearchHit


class RAGService:
    def __init__(self) -> None:
        self._enabled = bool(settings.groq_api_key)
        self._client = Groq(api_key=settings.groq_api_key) if self._enabled else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def answer(self, query: str, contexts: list[SearchHit]) -> str:
        if not self._enabled or self._client is None:
            return (
                "Groq is not configured. Set GROQ_API_KEY in your .env file and retry."
            )

        context_block = "\n\n".join(
            [
                f"[Source: {hit.id}]\n{hit.text}"
                for hit in contexts
                if hit.text.strip()
            ]
        )

        prompt = (
            "You are a concise study assistant. Answer strictly from the provided context. "
            "If context is insufficient, say so clearly. Include source ids in the answer.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context_block}"
        )

        try:
            completion = self._client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You provide grounded answers using only retrieved context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=700,
            )
        except AuthenticationError:
            return (
                "Groq authentication failed (invalid API key). "
                "Update GROQ_API_KEY in student-project/.env and restart backend."
            )
        except APIError as exc:
            return f"Groq API error: {exc}"
        except Exception as exc:
            return f"RAG temporarily unavailable: {exc}"

        return completion.choices[0].message.content or "No response generated."
