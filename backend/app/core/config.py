from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG Chatbot API"
    VERSION: str = "1.0.0"

    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "info"

    # Security
    API_KEY: str
    SECRET_KEY: str = "your-secret-key-here"  # In production, use a strong secret key
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Database
    DATABASE_URL: str

    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "documentation_chunks"

    # Gemini
    GEMINI_API_KEY: str
    GEMINI_MODEL_NAME: str = "gemini-pro"

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