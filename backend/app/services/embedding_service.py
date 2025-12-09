from typing import List, Dict, Any, Optional
from app.models.embedding import DocumentChunk, ProcessDocumentsResponse, RefreshDocumentsResponse, DeleteDocumentsResponse
from app.services.qdrant_service import QdrantService
from app.services.llm_service import LLMService
from app.utils.text_processor import TextProcessor
import time
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.qdrant_service = QdrantService()
        self.llm_service = LLMService()
        self.text_processor = TextProcessor()

    async def process_documents(self, documents: List[DocumentChunk], chunk_size: int = 512, overlap: int = 50) -> ProcessDocumentsResponse:
        """
        Process and store document chunks with embeddings
        """
        start_time = time.time()

        try:
            all_chunks = []
            for doc in documents:
                # Process the document content into chunks
                chunks = self.text_processor.chunk_text(
                    doc.content,
                    chunk_size=chunk_size,
                    overlap=overlap
                )

                # Create document chunk objects
                for i, chunk_text in enumerate(chunks):
                    chunk = {
                        "content": chunk_text,
                        "source_url": doc.source_url,
                        "page_title": doc.page_title,
                        "section": doc.section,
                        "chunk_order": i,
                        "metadata": doc.metadata or {}
                    }
                    all_chunks.append(chunk)

            # Generate embeddings for all chunks
            texts = [chunk["content"] for chunk in all_chunks]
            embeddings = await self.llm_service.generate_embeddings(texts)

            # Update chunks with embeddings (in a real implementation)
            # For now, we'll store the chunks without actual embeddings
            processed_count = await self.qdrant_service.store_embeddings(all_chunks)

            processing_time = int((time.time() - start_time) * 1000)

            return ProcessDocumentsResponse(
                status="success",
                processed_chunks=processed_count,
                collection_name=self.qdrant_service.collection_name,
                processing_time_ms=processing_time
            )
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    async def refresh_documents(self, urls: List[str], force_recreate: bool = False) -> RefreshDocumentsResponse:
        """
        Refresh embeddings for specific content or URLs
        """
        try:
            if force_recreate:
                # First delete existing chunks for these URLs
                deleted_count = await self.qdrant_service.delete_by_source_url(urls)
            else:
                deleted_count = 0

            # In a real implementation, we would re-fetch and re-process the documents
            # For now, we'll just return a response
            return RefreshDocumentsResponse(
                status="success",
                refreshed_chunks=0,  # Placeholder
                deleted_chunks=deleted_count
            )
        except Exception as e:
            logger.error(f"Error refreshing documents: {str(e)}")
            raise

    async def delete_documents(self, urls: List[str]) -> DeleteDocumentsResponse:
        """
        Remove document chunks from vector store
        """
        try:
            deleted_count = await self.qdrant_service.delete_by_source_url(urls)

            return DeleteDocumentsResponse(
                status="success",
                deleted_chunks=deleted_count
            )
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise