from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG Chatbot API"
    VERSION: str = "1.0.0"

    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "info"

    # Security
    API_KEY: str = os.getenv("API_KEY", "default-api-key")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "4f9b8c2e1a7d5f3b8e0c9d6a1f2e7b4c3d8a9e0f1b2c3d4e5f6a7b8c9d0e1f2")  # In production, use a strong secret key
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./rag_chatbot.db")

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION_NAME: str = "documentation_chunks"

    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "AIzaSyAvSq9AVdYQzx0uBeVy39f1WXDlgLZZdBU")
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]  # In production, specify exact origins

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # Embedding settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    CONTEXT_WINDOW: int = 3

    # Timeout settings
    API_TIMEOUT: int = 30
    DATABASE_TIMEOUT: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()