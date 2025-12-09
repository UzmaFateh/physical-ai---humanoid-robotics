import google.generativeai as genai
from typing import List, Dict, Any, Optional
from app.core.config import settings
import logging
import time

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        # Configure the Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Initialize the model
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts using a compatible model
        Note: Gemini doesn't directly provide embeddings, so we'll use a workaround
        or recommend using a different embedding model like OpenAI's text-embedding-ada-002
        """
        # In a real implementation, you might use a different service for embeddings
        # or use a locally available embedding model
        # For now, returning placeholder embeddings
        embeddings = []
        for text in texts:
            # Placeholder - in real implementation, use actual embedding service
            embedding = [0.0] * 768  # Example embedding size
            embeddings.append(embedding)

        return embeddings

    async def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a response using the LLM with the given prompt and optional context
        """
        try:
            # Combine context and prompt if context is provided
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nPlease provide a helpful answer based on the context provided."
            else:
                full_prompt = prompt

            # Generate content using Gemini
            response = await self.model.generate_content_async(full_prompt)

            # Extract the text from the response
            if response and response.text:
                return response.text.strip()
            else:
                return "I couldn't generate a response. Please try again."

        except Exception as e:
            logger.error(f"Error generating response with Gemini: {str(e)}")
            # Return a fallback response
            return "I'm having trouble generating a response right now. Please try again later."

    async def chat_completion(self, messages: List[Dict[str, str]], context: Optional[str] = None) -> str:
        """
        Simulate OpenAI-compatible chat completion using Gemini
        """
        try:
            # Combine all messages into a single prompt
            conversation_text = ""
            for msg in messages:
                conversation_text += f"{msg['role']}: {msg['content']}\n"

            # Add context if provided
            if context:
                full_prompt = f"Context: {context}\n\nConversation:\n{conversation_text}\n\nAssistant:"
            else:
                full_prompt = f"Conversation:\n{conversation_text}\n\nAssistant:"

            # Generate response
            response = await self.model.generate_content_async(full_prompt)

            if response and response.text:
                return response.text.strip()
            else:
                return "I couldn't generate a response. Please try again."

        except Exception as e:
            logger.error(f"Error in chat completion with Gemini: {str(e)}")
            return "I'm having trouble generating a response right now. Please try again later."

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text
        """
        try:
            result = await self.model.count_tokens_async(text)
            return result.total_tokens
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            # Fallback token estimation (rough approximation)
            return len(text.split())