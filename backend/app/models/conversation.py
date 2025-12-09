from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime


class ConversationCreateRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    conversation_id: str
    session_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class MessageCreateRequest(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    tokens_used: Optional[int] = None
    source_chunks: Optional[List[str]] = None


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    timestamp: datetime
    tokens_used: Optional[int] = None
    source_chunks: Optional[List[str]] = None


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    session_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    messages: List[MessageResponse]
    metadata: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    message_id: str
    score: int  # 1-5
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str


class StatsResponse(BaseModel):
    total_conversations: int
    total_messages: int
    active_sessions: int
    avg_response_time: float
    queries_today: int