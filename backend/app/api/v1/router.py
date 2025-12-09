from fastapi import APIRouter
from app.api.v1 import embeddings, rag, conversations

api_router = APIRouter()

# Include all API routes
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["embeddings"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
api_router.include_router(conversations.router, prefix="/conversations", tags=["conversations"])