from fastapi import APIRouter, Depends, HTTPException, status
from app.models.embedding import (
    ProcessDocumentsRequest, ProcessDocumentsResponse,
    RefreshDocumentsRequest, RefreshDocumentsResponse,
    DeleteDocumentsRequest, DeleteDocumentsResponse
)
from app.core.security import verify_api_key
from app.services.embedding_service import EmbeddingService
from app.core.config import settings

router = APIRouter()


@router.post("/process", response_model=ProcessDocumentsResponse)
async def process_documents(
    request: ProcessDocumentsRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Process and store document chunks with embeddings
    """
    try:
        embedding_service = EmbeddingService()
        result = await embedding_service.process_documents(
            request.documents,
            chunk_size=request.chunk_size,
            overlap=request.overlap
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing documents: {str(e)}"
        )


@router.post("/refresh", response_model=RefreshDocumentsResponse)
async def refresh_documents(
    request: RefreshDocumentsRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Refresh embeddings for specific content or URLs
    """
    try:
        embedding_service = EmbeddingService()
        result = await embedding_service.refresh_documents(
            request.urls,
            force_recreate=request.force_recreate
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error refreshing documents: {str(e)}"
        )


@router.delete("/documents", response_model=DeleteDocumentsResponse)
async def delete_documents(
    request: DeleteDocumentsRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Remove document chunks from vector store
    """
    try:
        embedding_service = EmbeddingService()
        result = await embedding_service.delete_documents(request.urls)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting documents: {str(e)}"
        )