from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class RAGQueryRequest(BaseModel):
    query: str
    selected_text: str
    context_window: int = 3
    session_id: Optional[str] = None
    source_url: Optional[str] = None


class RAGSource(BaseModel):
    url: str
    title: str
    content_snippet: str
    similarity_score: float


class RAGQueryResponse(BaseModel):
    response: str
    sources: List[RAGSource]
    conversation_id: Optional[str] = None
    tokens_used: Dict[str, int]
    response_time_ms: int


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    source_url: Optional[str] = None


class RAGSearchResult(BaseModel):
    id: str
    content: str
    source_url: str
    page_title: str
    similarity_score: float


class RAGSearchResponse(BaseModel):
    results: List[RAGSearchResult]


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    dependencies: Dict[str, bool]