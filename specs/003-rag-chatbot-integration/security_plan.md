# Security and Environment Configuration Plan

## Security Architecture

### 1. API Security

#### Authentication and Authorization
- **API Key Authentication**: All API endpoints require valid API keys
- **JWT Tokens**: For session management and user authentication (if needed)
- **Rate Limiting**: Per-API key and IP-based rate limiting
- **CORS Policy**: Restrict cross-origin requests to trusted domains

#### Security Headers
```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app = FastAPI()
app.add_middleware(SecurityHeadersMiddleware)
```

### 2. Input Validation and Sanitization

#### Request Validation
```python
from pydantic import BaseModel, validator
import html
import re

class RAGQueryRequest(BaseModel):
    query: str
    selected_text: str
    session_id: str = None
    source_url: str = None

    @validator('query', 'selected_text')
    def validate_text_content(cls, v):
        if len(v) > 2000:
            raise ValueError('Text content too long')
        # Remove potentially harmful content
        sanitized = html.escape(v)
        return sanitized

    @validator('source_url')
    def validate_url(cls, v):
        if v:
            # Basic URL validation
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            if not url_pattern.match(v):
                raise ValueError('Invalid URL format')
        return v
```

### 3. Environment Configuration (.env)

#### Environment Variables Structure
```
# API Configuration
API_KEY=your_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
DATABASE_URL=postgresql://user:password@host:port/dbname

# Service URLs
QDRANT_URL=https://your-qdrant-instance.qdrant.tech:6333
GEMINI_MODEL=gemini-pro

# Security Settings
SECRET_KEY=your_very_long_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
MAX_CONTENT_LENGTH=1048576  # 1MB in bytes

# Rate Limiting
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60  # seconds

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# Embedding Settings
EMBEDDING_MODEL=text-embedding-ada-002
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Database Settings
DB_POOL_SIZE=20
DB_POOL_TIMEOUT=30
DB_STATEMENT_TIMEOUT=30

# Cache Settings
CACHE_TTL=3600  # 1 hour
CACHE_MAX_SIZE=1000

# Monitoring
SENTRY_DSN=your_sentry_dsn_here  # Optional
```

#### .env Example File
```env
# RAG Chatbot Environment Configuration
# Copy this file to .env and update with your values

# API Keys and Secrets
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
DATABASE_URL=postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require

# Qdrant Configuration
QDRANT_URL=https://your-cluster-id.qdrant.tech:6333

# Security
SECRET_KEY=change_this_to_a_very_long_random_string_for_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application
DEBUG=false
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=10485760  # 10MB

# Rate Limiting
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,https://yourdomain.com,https://yourdomain.github.io

# Database
DB_POOL_SIZE=10
DB_POOL_TIMEOUT=30

# Embedding Pipeline
CHUNK_SIZE=512
CHUNK_OVERLAP=50
EMBEDDING_BATCH_SIZE=20

# Monitoring (Optional)
SENTRY_DSN=
PROMETHEUS_ENABLED=false
```

### 4. Secret Management

#### Secret Storage Best Practices
1. **Environment Variables**: Primary method for configuration
2. **Secret Management Services**: Use cloud provider secret managers in production
3. **File-based Secrets**: For local development only
4. **No Hardcoded Secrets**: Never commit secrets to version control

#### Configuration Class
```python
from pydantic import BaseSettings, validator
from typing import List
import secrets

class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str
    qdrant_api_key: str

    # URLs
    qdrant_url: str
    database_url: str

    # Security
    secret_key: str = secrets.token_urlsafe(32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Application
    debug: bool = False
    log_level: str = "INFO"

    # Rate Limiting
    rate_limit_requests: int = 60
    rate_limit_window: int = 60

    # CORS
    allowed_origins: List[str] = ["http://localhost:3000"]

    # Database
    db_pool_size: int = 10
    db_pool_timeout: int = 30

    # Embedding
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_batch_size: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v

settings = Settings()
```

### 5. Security Middleware

#### Rate Limiting Implementation
```python
import time
from collections import defaultdict, deque
from fastapi import Request, HTTPException
from functools import wraps

class RateLimiter:
    def __init__(self, max_requests: int, window_size: int):
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests = defaultdict(lambda: deque())

    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        requests = self.requests[identifier]

        # Remove old requests outside the window
        while requests and now - requests[0] > self.window_size:
            requests.popleft()

        if len(requests) >= self.max_requests:
            return False

        requests.append(now)
        return True

# Global rate limiter instance
rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_requests,
    window_size=settings.rate_limit_window
)

def rate_limit(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs.get('request') or args[0]
        client_ip = request.client.host

        if not rate_limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )

        return await func(*args, **kwargs)
    return wrapper
```

#### CORS Configuration
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Only allow credentials for trusted origins
    allow_origin_regex=r"https://.*\.yourdomain\.com"
)
```

### 6. Database Security

#### Connection Security
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

def create_secure_engine():
    return create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=settings.db_pool_size,
        pool_timeout=settings.db_pool_timeout,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=300,    # Recycle connections every 5 minutes
    )

engine = create_secure_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

#### SQL Injection Prevention
```python
from sqlalchemy.orm import Session
from sqlalchemy import text

def get_conversation_by_session_safe(db: Session, session_id: str):
    # Use parameterized queries to prevent SQL injection
    result = db.execute(
        text("SELECT * FROM conversations WHERE session_id = :session_id"),
        {"session_id": session_id}
    ).fetchone()
    return result
```

### 7. File Upload Security (if applicable)

#### Secure File Handling
```python
import magic
from pathlib import Path

def validate_uploaded_file(file_path: str) -> bool:
    # Check file type using python-magic
    file_type = magic.from_file(file_path, mime=True)

    allowed_types = [
        'text/plain',
        'text/html',
        'text/markdown',
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]

    return file_type in allowed_types

def secure_filename(filename: str) -> str:
    # Remove path traversal characters
    filename = Path(filename).name
    # Add timestamp to prevent conflicts
    name, ext = Path(filename).stem, Path(filename).suffix
    timestamp = int(time.time())
    return f"{name}_{timestamp}{ext}"
```

### 8. Logging and Monitoring Security

#### Secure Logging
```python
import logging
from logging.handlers import RotatingFileHandler
import re

class SecureFormatter(logging.Formatter):
    def format(self, record):
        # Sanitize sensitive information from logs
        msg = super().format(record)

        # Remove API keys from logs
        msg = re.sub(r'API_KEY=\w+', 'API_KEY=***', msg)
        msg = re.sub(r'Bearer [a-zA-Z0-9-_.]+', 'Bearer ***', msg)

        record.msg = msg
        return super().format(record)

def setup_secure_logging():
    logger = logging.getLogger()
    handler = RotatingFileHandler(
        "app.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    handler.setFormatter(SecureFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, settings.log_level.upper()))
```

### 9. Dependency Security

#### Security Scanning
```bash
# Requirements with security considerations
pip install safety
safety check -r requirements.txt

# Or use other tools
pip install bandit  # For code scanning
bandit -r .
```

#### Dependency Updates
- Regularly update dependencies
- Use dependabot or similar tools for automated updates
- Monitor for security vulnerabilities in dependencies

### 10. Production Security Checklist

- [ ] Use HTTPS in production
- [ ] Implement proper CORS policies
- [ ] Set secure headers
- [ ] Validate and sanitize all inputs
- [ ] Use parameterized queries to prevent SQL injection
- [ ] Implement rate limiting
- [ ] Store secrets securely (not in code)
- [ ] Use strong secret keys
- [ ] Implement proper authentication
- [ ] Monitor and log security events
- [ ] Regular security audits
- [ ] Update dependencies regularly
- [ ] Implement proper error handling (don't expose internal details)
- [ ] Use secure session management
- [ ] Implement CSRF protection if needed