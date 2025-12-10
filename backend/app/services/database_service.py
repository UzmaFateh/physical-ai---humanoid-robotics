from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSON
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()

# Create engine and session - using SQLite for development
import os

# Use database from environment settings or fallback to SQLite
db_url = os.getenv("DATABASE_URL", "sqlite:///./rag_chatbot.db")
# For SQLite, we need special handling
if db_url.startswith("sqlite"):
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False}  # Required for SQLite
    )
else:
    # For PostgreSQL or other databases
    engine = create_engine(db_url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    software_background = Column(String, nullable=True)
    hardware_background = Column(String, nullable=True)
    experience_level = Column(String, nullable=True)
    additional_info = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    meta_data = Column('metadata', JSON)  # JSONB for additional metadata


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role = Column(String, index=True)  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=func.now(), index=True)
    tokens_used = Column(Integer, default=0)
    source_chunks = Column(JSON)  # JSON array of relevant chunk IDs
    response_time_ms = Column(Integer, default=0)
    feedback_score = Column(Integer)  # 1-5 rating

    # Relationship
    conversation = relationship("Conversation", back_populates="messages")


Conversation.messages = relationship("Message", back_populates="conversation")


class DatabaseService:
    def __init__(self):
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)

    def get_db(self) -> Session:
        """
        Get database session
        """
        db = SessionLocal()
        try:
            return db
        except Exception:
            db.close()
            raise

    async def create_conversation(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new conversation record
        """
        db = self.get_db()
        try:
            conversation_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())

            conversation = Conversation(
                id=conversation_id,
                session_id=session_id,
                metadata=metadata or {}
            )

            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            logger.info(f"Created conversation: {conversation_id}")
            return {
                "conversation_id": conversation.id,
                "session_id": conversation.session_id,
                "created_at": conversation.created_at,
                "metadata": conversation.meta_data
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating conversation: {str(e)}")
            raise
        finally:
            db.close()

    async def get_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve conversation by session ID
        """
        db = self.get_db()
        try:
            conversation = db.query(Conversation).filter(Conversation.session_id == session_id).first()
            if not conversation:
                return None

            # Get messages for this conversation
            messages = db.query(Message).filter(Message.conversation_id == conversation.id).order_by(Message.timestamp).all()

            message_list = []
            for msg in messages:
                message_list.append({
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "tokens_used": msg.tokens_used,
                    "source_chunks": msg.source_chunks
                })

            return {
                "conversation_id": conversation.id,
                "session_id": conversation.session_id,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "messages": message_list,
                "metadata": conversation.meta_data
            }
        except Exception as e:
            logger.error(f"Error retrieving conversation: {str(e)}")
            raise
        finally:
            db.close()

    async def add_message(self, conversation_id: str, role: str, content: str, tokens_used: Optional[int] = None,
                         source_chunks: Optional[List[str]] = None, response_time_ms: Optional[int] = None) -> str:
        """
        Add a message to a conversation
        """
        db = self.get_db()
        try:
            message_id = str(uuid.uuid4())

            message = Message(
                id=message_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                tokens_used=tokens_used or 0,
                source_chunks=source_chunks or [],
                response_time_ms=response_time_ms or 0
            )

            db.add(message)
            db.commit()
            db.refresh(message)

            logger.info(f"Added message to conversation {conversation_id}: {message_id}")
            return message_id
        except Exception as e:
            db.rollback()
            logger.error(f"Error adding message: {str(e)}")
            raise
        finally:
            db.close()

    async def submit_feedback(self, message_id: str, score: int, comment: Optional[str] = None) -> bool:
        """
        Submit feedback for a specific message
        """
        db = self.get_db()
        try:
            message = db.query(Message).filter(Message.id == message_id).first()
            if not message:
                return False

            message.feedback_score = score
            db.commit()

            logger.info(f"Submitted feedback for message {message_id}: score {score}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error submitting feedback: {str(e)}")
            raise
        finally:
            db.close()

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics
        """
        db = self.get_db()
        try:
            total_conversations = db.query(Conversation).count()
            total_messages = db.query(Message).count()

            # Calculate active sessions (conversations updated in the last hour)
            from datetime import timedelta
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            active_sessions = db.query(Conversation).filter(Conversation.updated_at >= one_hour_ago).count()

            # Calculate average response time
            avg_response_time = db.query(func.avg(Message.response_time_ms)).scalar() or 0

            # Calculate queries today
            today = datetime.utcnow().date()
            queries_today = db.query(Message).filter(
                func.date(Message.timestamp) == today,
                Message.role == "user"
            ).count()

            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "active_sessions": active_sessions,
                "avg_response_time": float(avg_response_time),
                "queries_today": queries_today
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            raise
        finally:
            db.close()