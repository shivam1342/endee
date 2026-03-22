from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8000"))

    endee_base_url: str = os.getenv("ENDEE_BASE_URL", "http://127.0.0.1:8080")
    endee_auth_token: str = os.getenv("ENDEE_AUTH_TOKEN", "")
    endee_index_name: str = os.getenv("ENDEE_INDEX_NAME", "student_notes")
    endee_space_type: str = os.getenv("ENDEE_SPACE_TYPE", "cosine")
    endee_top_k: int = int(os.getenv("ENDEE_TOP_K", "5"))

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


settings = Settings()
