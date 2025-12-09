from typing import List, Dict, Any, Optional
from app.models.rag import RAGQueryResponse, RAGSearchResult
from app.services.qdrant_service import QdrantService
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
import logging

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.qdrant_service = QdrantService()
        self.llm_service = LLMService()
        self.embedding_service = EmbeddingService()

    async def query(self, query: str, selected_text: str, context_window: int = 3,
                   session_id: Optional[str] = None, source_url: Optional[str] = None) -> RAGQueryResponse:
        """
        Process a RAG query with selected text context
        """
        try:
            # First, search for relevant chunks based on the query
            # For now, we'll use a simple embedding of the query
            query_embeddings = await self.llm_service.generate_embeddings([query])
            query_vector = query_embeddings[0] if query_embeddings else [0.0] * 768

            # Search for relevant chunks
            search_results = await self.qdrant_service.search_similar(
                query_vector=query_vector,
                top_k=5,  # Get top 5 relevant chunks
                source_url=source_url
            )

            # Build context from search results and selected text
            context_parts = []

            # Add selected text as primary context
            context_parts.append(f"Selected Text: {selected_text}")

            # Add relevant chunks as additional context
            for result in search_results:
                context_parts.append(f"Related Content: {result['content']}")

            context = "\n\n".join(context_parts)

            # Generate response using the LLM
            response = await self.llm_service.generate_response(query, context)

            # Count tokens used
            input_tokens = await self.llm_service.count_tokens(context + query)
            output_tokens = await self.llm_service.count_tokens(response)

            # Format sources
            sources = []
            for result in search_results:
                sources.append({
                    "url": result["source_url"],
                    "title": result["page_title"],
                    "content_snippet": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                    "similarity_score": result["similarity_score"]
                })

            return RAGQueryResponse(
                response=response,
                sources=sources,
                conversation_id=session_id,
                tokens_used={
                    "input": input_tokens,
                    "output": output_tokens
                },
                response_time_ms=0  # This will be set in the API endpoint
            )
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            raise

    async def search(self, query: str, top_k: int = 5, source_url: Optional[str] = None) -> List[RAGSearchResult]:
        """
        Search for relevant chunks without generating response
        """
        try:
            # Generate embedding for the query
            query_embeddings = await self.llm_service.generate_embeddings([query])
            query_vector = query_embeddings[0] if query_embeddings else [0.0] * 768

            # Search for relevant chunks
            search_results = await self.qdrant_service.search_similar(
                query_vector=query_vector,
                top_k=top_k,
                source_url=source_url
            )

            # Format results
            formatted_results = []
            for result in search_results:
                formatted_results.append(
                    RAGSearchResult(
                        id=result["id"],
                        content=result["content"],
                        source_url=result["source_url"],
                        page_title=result["page_title"],
                        similarity_score=result["similarity_score"]
                    )
                )

            return formatted_results
        except Exception as e:
            logger.error(f"Error in RAG search: {str(e)}")
            raise