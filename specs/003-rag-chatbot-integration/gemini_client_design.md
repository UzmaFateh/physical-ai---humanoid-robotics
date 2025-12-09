# OpenAI-Style Client for Gemini Integration

## Overview

This document outlines the design for a Gemini API client that mimics the OpenAI API interface to allow for easy swapping between providers. The client will provide the same interface as OpenAI but use Google's Gemini API under the hood.

## Architecture

### Client Structure

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │───▶│   GeminiClient   │───▶│   Gemini API    │
│   (using OpenAI │    │   (OpenAI-style  │    │   (Google's     │
│   interface)    │    │   wrapper)       │    │   service)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Implementation Design

### Core Client Class

```python
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
import time
import logging

class GeminiClient:
    """
    A client that provides OpenAI-compatible interface for Google's Gemini API
    """
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        """
        Initialize the Gemini client with API key and default model
        """
        genai.configure(api_key=api_key)
        self.model_name = model
        self.client = genai.GenerativeModel(model_name=model)
        self.logger = logging.getLogger(__name__)

    def chat_completions_create(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Union['ChatCompletionResponse', 'StreamResponse']:
        """
        Create a chat completion using Gemini API with OpenAI-compatible interface
        """
        if stream:
            return self._stream_chat_completions(messages, temperature, max_tokens, top_p, **kwargs)
        else:
            return self._create_chat_completion(messages, temperature, max_tokens, top_p, **kwargs)

    def embeddings_create(
        self,
        input: Union[str, List[str]],
        model: str = "text-embedding-004",  # Placeholder - will use appropriate embedding method
    ) -> 'EmbeddingResponse':
        """
        Create embeddings using appropriate method (may need to use different service)
        """
        return self._create_embeddings(input, model)
```

### Message Conversion

Since Gemini uses a different message format than OpenAI, we need conversion logic:

```python
def _convert_messages_to_gemini(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert OpenAI-style messages to Gemini format
    """
    gemini_messages = []

    for message in messages:
        role = message.get('role', 'user')
        content = message.get('content', '')

        # Convert role from OpenAI to Gemini format
        if role == 'system':
            # Gemini doesn't have system role, so we'll add it as part of user message
            gemini_messages.append({
                'role': 'user',
                'parts': [f"System instruction: {content}"]
            })
        elif role in ['user', 'assistant']:
            gemini_messages.append({
                'role': 'model' if role == 'assistant' else 'user',
                'parts': [content]
            })

    return gemini_messages

def _convert_response_from_gemini(self, response, original_messages: List[Dict[str, str]]) -> 'ChatCompletionResponse':
    """
    Convert Gemini response to OpenAI-compatible format
    """
    # Extract the response text
    response_text = response.text

    # Create OpenAI-compatible response object
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=self.model_name,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        usage={
            "prompt_tokens": self._count_tokens(original_messages),
            "completion_tokens": self._count_tokens([{"role": "assistant", "content": response_text}]),
            "total_tokens": self._count_tokens(original_messages) +
                           self._count_tokens([{"role": "assistant", "content": response_text}])
        }
    )
```

### Pydantic Models

```python
from pydantic import BaseModel
from typing import List, Dict, Optional, Union

class ChatMessage(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class EmbeddingData(BaseModel):
    index: int
    embedding: List[float]
    object: str = "embedding"

class EmbeddingResponse(BaseModel):
    object: str
    data: List[EmbeddingData]
    model: str
    usage: Optional[Usage] = None
```

### Advanced Features

#### Streaming Support

```python
class StreamResponse:
    def __init__(self, response_stream):
        self.response_stream = response_stream

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.response_stream)
            return self._convert_stream_chunk(chunk)
        except StopIteration:
            raise StopIteration

    def _convert_stream_chunk(self, chunk):
        # Convert Gemini streaming chunk to OpenAI format
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk.text
                    },
                    "finish_reason": None
                }
            ]
        }
```

#### Error Handling

```python
class GeminiAPIError(Exception):
    """Custom exception for Gemini API errors"""
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

def _handle_gemini_error(self, error: Exception) -> None:
    """
    Handle Gemini API errors and convert to OpenAI-compatible format
    """
    if hasattr(error, 'args') and len(error.args) > 0:
        error_msg = str(error.args[0])
    else:
        error_msg = str(error)

    # Map common errors to appropriate responses
    if "quota" in error_msg.lower() or "rate" in error_msg.lower():
        raise GeminiAPIError("Rate limit exceeded", 429)
    elif "invalid" in error_msg.lower() or "api_key" in error_msg.lower():
        raise GeminiAPIError("Invalid API key", 401)
    else:
        raise GeminiAPIError(error_msg, 500)
```

### Service Integration

The client will be integrated into the RAG service:

```python
class LLMService:
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.client = GeminiClient(api_key=api_key, model=model)

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using the configured LLM
        """
        # Add context to messages if provided
        if context:
            context_message = {
                "role": "system",
                "content": f"Use the following context to answer the question: {context}"
            }
            messages = [context_message] + messages

        try:
            response = self.client.chat_completions_create(
                messages=messages,
                temperature=temperature
            )

            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            raise
```

### Configuration and Settings

```python
from pydantic import BaseSettings

class LLMSettings(BaseSettings):
    gemini_api_key: str
    gemini_model: str = "gemini-pro"
    gemini_temperature: float = 0.7
    gemini_max_tokens: int = 1000
    gemini_top_p: float = 1.0

    class Config:
        env_file = ".env"
        env_prefix = "GEMINI_"
```

### Testing Strategy

#### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestGeminiClient:
    def test_chat_completions_create(self):
        # Test that the client properly converts messages and calls Gemini API
        client = GeminiClient(api_key="test-key")

        messages = [
            {"role": "user", "content": "Hello"}
        ]

        with patch('google.generativeai.GenerativeModel.generate_content') as mock_generate:
            mock_response = Mock()
            mock_response.text = "Hi there!"
            mock_generate.return_value = mock_response

            response = client.chat_completions_create(messages=messages)

            # Verify the response is in OpenAI format
            assert response.object == "chat.completion"
            assert response.choices[0].message.content == "Hi there!"

    def test_message_conversion(self):
        # Test message conversion logic
        client = GeminiClient(api_key="test-key")

        openai_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]

        gemini_messages = client._convert_messages_to_gemini(openai_messages)

        # Verify conversion logic
        assert len(gemini_messages) == 3
        assert gemini_messages[0]['role'] == 'user'  # System converted to user
        assert gemini_messages[1]['role'] == 'user'
        assert gemini_messages[2]['role'] == 'model'  # Assistant converted to model
```

### Performance Considerations

#### Caching Strategy

```python
import hashlib
from functools import lru_cache

class CachedGeminiClient(GeminiClient):
    def __init__(self, api_key: str, model: str = "gemini-pro", cache_size: int = 1000):
        super().__init__(api_key, model)
        self.cache_size = cache_size

    @lru_cache(maxsize=1000)
    def _cached_completion(self, messages_hash: str, messages: tuple, temperature: float):
        # Convert tuple back to list for processing
        messages_list = list(messages)
        response = self.client.chat_completions_create(
            messages=messages_list,
            temperature=temperature
        )
        return response

    def chat_completions_create(self, messages: List[Dict[str, str]], temperature: float = 0.7, **kwargs):
        # Create a hash of the messages for caching
        messages_str = str(sorted([(m['role'], m['content']) for m in messages]))
        messages_hash = hashlib.md5(messages_str.encode()).hexdigest()

        # Convert messages to tuple for hashing
        messages_tuple = tuple((m['role'], m['content']) for m in messages)

        return self._cached_completion(messages_hash, messages_tuple, temperature)
```

### Security Considerations

#### API Key Management

- Store API keys in environment variables
- Use proper secrets management in production
- Implement rate limiting to prevent abuse
- Log API usage for monitoring but don't log sensitive content

#### Input Sanitization

- Sanitize user inputs before sending to LLM
- Implement content filtering to prevent prompt injection
- Validate message structure before processing

This design provides a clean abstraction layer that allows the application to use Google's Gemini API with an OpenAI-compatible interface, making it easy to switch between providers or even support multiple providers simultaneously.