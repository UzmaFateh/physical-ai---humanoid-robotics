from fastapi import APIRouter, Depends, HTTPException, status
from app.models.conversation import (
    ConversationCreateRequest, ConversationResponse,
    MessageCreateRequest, MessageResponse,
    ConversationDetailResponse, FeedbackRequest, FeedbackResponse,
    StatsResponse
)
from app.core.security import verify_api_key
from app.services.conversation_service import ConversationService

router = APIRouter()


@router.post("/", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationCreateRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Create new conversation session
    """
    try:
        conversation_service = ConversationService()
        result = await conversation_service.create_conversation(request.metadata)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating conversation: {str(e)}"
        )


@router.get("/{session_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Retrieve conversation history
    """
    try:
        conversation_service = ConversationService()
        result = await conversation_service.get_conversation(session_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving conversation: {str(e)}"
        )


@router.post("/{session_id}/messages", response_model=MessageResponse)
async def add_message(
    session_id: str,
    request: MessageCreateRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Add message to conversation
    """
    try:
        conversation_service = ConversationService()
        result = await conversation_service.add_message(
            session_id=session_id,
            role=request.role,
            content=request.content,
            tokens_used=request.tokens_used,
            source_chunks=request.source_chunks
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding message: {str(e)}"
        )


@router.post("/{session_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    session_id: str,
    request: FeedbackRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Submit feedback for a specific response
    """
    try:
        conversation_service = ConversationService()
        result = await conversation_service.submit_feedback(
            message_id=request.message_id,
            score=request.score,
            comment=request.comment
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting feedback: {str(e)}"
        )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    api_key: str = Depends(verify_api_key)
):
    """
    Get system statistics
    """
    try:
        conversation_service = ConversationService()
        result = await conversation_service.get_stats()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )