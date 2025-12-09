from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    content: str
    source_url: str
    page_title: str
    section: Optional[str] = None
    chunk_order: int
    metadata: Optional[Dict[str, Any]] = None


class ProcessDocumentsRequest(BaseModel):
    documents: List[DocumentChunk]
    chunk_size: int = 512
    overlap: int = 50


class ProcessDocumentsResponse(BaseModel):
    status: str
    processed_chunks: int
    collection_name: str
    processing_time_ms: int


class RefreshDocumentsRequest(BaseModel):
    urls: List[str]
    force_recreate: bool = False


class RefreshDocumentsResponse(BaseModel):
    status: str
    refreshed_chunks: int
    deleted_chunks: int


class DeleteDocumentsRequest(BaseModel):
    urls: List[str]


class DeleteDocumentsResponse(BaseModel):
    status: str
    deleted_chunks: int


class EmbeddingRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embeddings: List[float]
    model: str
    usage: Dict[str, int]