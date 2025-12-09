from typing import List, Dict, Any, Optional
from app.models.conversation import ConversationResponse, ConversationDetailResponse, MessageResponse, FeedbackResponse, StatsResponse
from app.services.database_service import DatabaseService
import logging

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(self):
        self.db_service = DatabaseService()

    async def create_conversation(self, metadata: Optional[Dict[str, Any]] = None) -> ConversationResponse:
        """
        Create a new conversation session
        """
        try:
            result = await self.db_service.create_conversation(metadata)
            return ConversationResponse(
                conversation_id=result["conversation_id"],
                session_id=result["session_id"],
                created_at=result["created_at"],
                metadata=result["metadata"]
            )
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise

    async def get_conversation(self, session_id: str) -> ConversationDetailResponse:
        """
        Retrieve conversation history by session ID
        """
        try:
            result = await self.db_service.get_conversation(session_id)
            if not result:
                raise ValueError(f"Conversation with session_id {session_id} not found")

            return ConversationDetailResponse(
                conversation_id=result["conversation_id"],
                session_id=result["session_id"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
                messages=[
                    MessageResponse(
                        id=msg["id"],
                        role=msg["role"],
                        content=msg["content"],
                        timestamp=msg["timestamp"],
                        tokens_used=msg["tokens_used"],
                        source_chunks=msg["source_chunks"]
                    ) for msg in result["messages"]
                ],
                metadata=result["metadata"]
            )
        except Exception as e:
            logger.error(f"Error retrieving conversation: {str(e)}")
            raise

    async def add_message(self, session_id: str, role: str, content: str,
                         tokens_used: Optional[int] = None,
                         source_chunks: Optional[List[str]] = None) -> MessageResponse:
        """
        Add a message to a conversation
        """
        try:
            # First get the conversation to get its ID
            conv_result = await self.db_service.get_conversation(session_id)
            if not conv_result:
                # If conversation doesn't exist, create one
                conv = await self.create_conversation()
                conversation_id = conv.conversation_id
            else:
                conversation_id = conv_result["conversation_id"]

            message_id = await self.db_service.add_message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                tokens_used=tokens_used,
                source_chunks=source_chunks
            )

            # Return the created message
            return MessageResponse(
                id=message_id,
                role=role,
                content=content,
                timestamp=None,  # We'll need to fetch this from DB if needed
                tokens_used=tokens_used,
                source_chunks=source_chunks
            )
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            raise

    async def submit_feedback(self, message_id: str, score: int, comment: Optional[str] = None) -> FeedbackResponse:
        """
        Submit feedback for a specific message
        """
        try:
            success = await self.db_service.submit_feedback(message_id, score, comment)
            status = "success" if success else "message_not_found"
            return FeedbackResponse(status=status)
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            raise

    async def get_stats(self) -> StatsResponse:
        """
        Get system statistics
        """
        try:
            stats = await self.db_service.get_stats()
            return StatsResponse(**stats)
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            raise