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
        Generate embeddings for the given texts.
        Using a local embedding model as Google's API doesn't have direct embeddings access in the generative library.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            # Using a lightweight but effective model for embeddings
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts).tolist()
            return embeddings
        except ImportError:
            logger.error("sentence_transformers not installed, installing now...")
            import subprocess
            import sys

            # Install sentence_transformers if not present
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])

            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts).tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return placeholder embeddings as final fallback
            embeddings = []
            for text in texts:
                embedding = [0.0] * 384  # Using 384 for MiniLM model
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