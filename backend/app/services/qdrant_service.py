from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
from app.core.config import settings
import uuid
import logging

logger = logging.getLogger(__name__)


class QdrantService:
    def __init__(self):
        if settings.QDRANT_URL.startswith("https://"):
            # Use HTTPS with API key for Qdrant Cloud
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                prefer_grpc=False
            )
        else:
            # Use HTTP for local Qdrant
            self.client = QdrantClient(host="localhost", port=6333)

        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Ensure the collection exists with the proper configuration
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with appropriate configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768,  # Gemini embeddings are typically 768-dim
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    async def store_embeddings(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Store document chunks with their embeddings in Qdrant
        """
        try:
            points = []
            for chunk in chunks:
                point_id = str(uuid.uuid4())

                # Create payload with metadata
                payload = {
                    "content": chunk["content"],
                    "source_url": chunk["source_url"],
                    "page_title": chunk["page_title"],
                    "section": chunk.get("section", ""),
                    "chunk_order": chunk.get("chunk_order", 0),
                    "metadata": chunk.get("metadata", {})
                }

                # For now, we'll use a placeholder for the actual embedding
                # In a real implementation, this would be the actual vector
                vector = [0.0] * 768  # Placeholder - replace with actual embedding

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )

            # Upload points to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Stored {len(points)} chunks in Qdrant")
            return len(points)
        except Exception as e:
            logger.error(f"Error storing embeddings in Qdrant: {str(e)}")
            raise

    async def search_similar(self, query_vector: List[float], top_k: int = 5, source_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the vector database
        """
        try:
            # Prepare filters if source_url is provided
            filters = None
            if source_url:
                filters = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_url",
                            match=models.MatchValue(value=source_url)
                        )
                    ]
                )

            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filters,
                with_payload=True
            )

            # Format results
            results = []
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "content": hit.payload["content"],
                    "source_url": hit.payload["source_url"],
                    "page_title": hit.payload["page_title"],
                    "similarity_score": hit.score
                }
                results.append(result)

            logger.info(f"Found {len(results)} similar chunks")
            return results
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}")
            raise

    async def delete_by_source_url(self, urls: List[str]) -> int:
        """
        Delete chunks by source URL
        """
        try:
            # Find points to delete
            points_to_delete = []

            for url in urls:
                # Search for points with this URL
                search_results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_url",
                                match=models.MatchValue(value=url)
                            )
                        ]
                    ),
                    limit=10000  # Adjust based on expected number of chunks per URL
                )

                for point in search_results[0]:
                    points_to_delete.append(point.id)

            # Delete points if any found
            if points_to_delete:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=points_to_delete
                    )
                )

            logger.info(f"Deleted {len(points_to_delete)} chunks for URLs: {urls}")
            return len(points_to_delete)
        except Exception as e:
            logger.error(f"Error deleting from Qdrant: {str(e)}")
            raise

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.params.vectors_count,
                "vector_size": collection_info.config.params.vector_size,
                "distance": collection_info.config.params.distance
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise