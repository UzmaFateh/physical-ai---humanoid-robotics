from fastapi import APIRouter, Depends, HTTPException, status
from app.models.rag import (
    RAGQueryRequest, RAGQueryResponse,
    RAGSearchRequest, RAGSearchResponse,
    HealthCheckResponse
)
from app.core.security import verify_api_key
from app.services.rag_service import RAGService
from app.services.conversation_service import ConversationService
from app.core.config import settings
import time
from typing import Dict, Any

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint
    """
    # In a real implementation, you would check actual dependencies
    dependencies = {
        "qdrant": True,
        "database": True,
        "gemini_api": True
    }

    return HealthCheckResponse(
        status="healthy",
        timestamp="2025-12-09T16:00:00Z",
        dependencies=dependencies
    )


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Main RAG query endpoint with selected text context
    """
    try:
        start_time = time.time()

        rag_service = RAGService()
        result = await rag_service.query(
            query=request.query,
            selected_text=request.selected_text,
            context_window=request.context_window,
            session_id=request.session_id,
            source_url=request.source_url
        )

        response_time_ms = int((time.time() - start_time) * 1000)

        # Add response time to the result
        result.response_time_ms = response_time_ms

        # If session_id was provided, save the conversation
        if request.session_id:
            conversation_service = ConversationService()
            await conversation_service.add_message(
                session_id=request.session_id,
                role="user",
                content=request.query
            )
            await conversation_service.add_message(
                session_id=request.session_id,
                role="assistant",
                content=result.response
            )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing RAG query: {str(e)}"
        )


@router.post("/search", response_model=RAGSearchResponse)
async def rag_search(
    request: RAGSearchRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Search for relevant chunks without generating response
    """
    try:
        rag_service = RAGService()
        results = await rag_service.search(
            query=request.query,
            top_k=request.top_k,
            source_url=request.source_url
        )
        return RAGSearchResponse(results=results)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing search: {str(e)}"
        )