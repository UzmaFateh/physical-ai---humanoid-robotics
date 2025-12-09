# Deployment Strategy for RAG Chatbot

## Overview

This document outlines the deployment strategy for the RAG Chatbot system, covering both backend and frontend components, infrastructure requirements, CI/CD pipeline, and operational considerations.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Docusaurus    │────│   FastAPI        │────│   Qdrant        │
│   Documentation │    │   Backend        │    │   Vector DB     │
│   Site          │    │   (Cloud Run)    │    │   (Cloud)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              │
                       ┌──────────────────┐
                       │   Neon Postgres  │
                       │   (Serverless)   │
                       └──────────────────┘
```

## Infrastructure Requirements

### Backend Infrastructure

#### Cloud Platform Options
1. **Google Cloud Run** (Recommended)
   - Serverless containers
   - Automatic scaling
   - Pay-per-use pricing
   - Easy integration with other Google services

2. **AWS Fargate**
   - Serverless compute for containers
   - Integration with ECS
   - Auto scaling

3. **Railway** (Alternative)
   - Developer-friendly platform
   - Built-in CI/CD
   - Good for startups

#### Resource Requirements
- **CPU**: Minimum 0.25 vCPU, typically 0.5-1 vCPU
- **Memory**: Minimum 512MB, typically 1-2GB
- **Concurrency**: Auto-scaling based on load
- **Storage**: Ephemeral storage for the container

### Database Infrastructure

#### Neon Postgres Configuration
- **Plan**: Serverless (Pay-as-you-go)
- **Region**: Match with backend deployment region
- **Connection Pool**: Transaction pooling recommended
- **Branching**: Use for development/staging environments

#### Qdrant Cloud Configuration
- **Plan**: Scale as needed based on vector count
- **Region**: Match with backend deployment region
- **Replicas**: 2 for high availability
- **Storage**: SSD for better performance

## Deployment Environments

### Development Environment
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Local Docs  │────│   Local FastAPI  │────│   Local Qdrant  │
│   (Docusaurus)  │    │   (Docker)       │    │   (Docker)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              │
                       ┌──────────────────┐
                       │   Local Postgres │
                       │   (Docker)       │
                       └──────────────────┘
```

**Components:**
- Local FastAPI server running on Docker
- Local Qdrant instance in Docker
- Local Postgres in Docker
- Docusaurus site with development API endpoint

### Staging Environment
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Staging Docs  │────│   Staging API    │────│   Qdrant        │
│   (Cloud)       │    │   (Cloud Run)    │    │   (Cloud)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              │
                       ┌──────────────────┐
                       │   Neon Staging   │
                       │   Postgres       │
                       └──────────────────┘
```

**Components:**
- Staging FastAPI deployment
- Staging Qdrant instance
- Staging Neon Postgres database
- Staging Docusaurus site

### Production Environment
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Production    │────│   Production API │────│   Production    │
│   Docs (CDN)    │    │   (Cloud Run)    │    │   Qdrant        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              │
                       ┌──────────────────┐
                       │   Production     │
                       │   Neon Postgres  │
                       └──────────────────┘
```

**Components:**
- Production FastAPI deployment (high availability)
- Production Qdrant cluster
- Production Neon Postgres database
- Production Docusaurus site with CDN

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy RAG Chatbot

on:
  push:
    branches: [ main, staging ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=app/

    - name: Security scan
      run: |
        pip install safety
        safety check -r requirements.txt

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'

    steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: gcr.io
        username: _json_key
        password: ${{ secrets.GCP_SA_KEY }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./backend/Dockerfile
        push: true
        tags: gcr.io/${{ secrets.GCP_PROJECT_ID }}/rag-chatbot:${{ github.sha }}

    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy rag-chatbot \
          --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/rag-chatbot:${{ github.sha }} \
          --region ${{ secrets.GCP_REGION }} \
          --platform managed \
          --set-env-vars GEMINI_API_KEY=${{ secrets.GEMINI_API_KEY }} \
          --set-env-vars QDRANT_API_KEY=${{ secrets.QDRANT_API_KEY }} \
          --set-env-vars DATABASE_URL=${{ secrets.DATABASE_URL }} \
          --set-env-vars QDRANT_URL=${{ secrets.QDRANT_URL }} \
          --memory 1Gi \
          --cpu 1 \
          --allow-unauthenticated
      env:
        GOOGLE_APPLICATION_CREDENTIALS: ${{ github.workspace }}/gcp-key.json
      working-directory: ./backend

  deploy-docs:
    needs: build-and-deploy
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'

    - name: Install dependencies
      run: npm ci

    - name: Build Docusaurus site
      run: npm run build

    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./build
```

### Docker Configuration

#### Backend Dockerfile
```dockerfile
# backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Backend requirements.txt
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
qdrant-client==1.8.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
pydantic==2.5.0
python-multipart==0.0.6
python-dotenv==1.0.0
google-generativeai==0.3.1
pyjwt==2.8.0
passlib[bcrypt]==1.7.4
python-jose[cryptography]==3.3.0
alembic==1.13.1
httpx==0.25.2
requests==2.31.0
```

## Deployment Scripts

### Deployment Configuration

#### Docker Compose for Local Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
      - QDRANT_URL=${QDRANT_URL}
      - DATABASE_URL=${DATABASE_URL}
      - DEBUG=true
    depends_on:
      - qdrant
      - postgres

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: rag_chatbot
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  qdrant_data:
  postgres_data:
```

#### Cloud Run Deployment Script
```bash
#!/bin/bash
# deploy.sh

set -e

# Configuration
PROJECT_ID=$1
REGION=${2:-us-central1}
SERVICE_NAME=rag-chatbot
IMAGE_NAME=gcr.io/$PROJECT_ID/$SERVICE_NAME

echo "Deploying RAG Chatbot to Cloud Run..."

# Build and push Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME ./backend
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --region $REGION \
  --platform managed \
  --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY \
  --set-env-vars QDRANT_API_KEY=$QDRANT_API_KEY \
  --set-env-vars QDRANT_URL=$QDRANT_URL \
  --set-env-vars DATABASE_URL=$DATABASE_URL \
  --set-env-vars SECRET_KEY=$SECRET_KEY \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 80 \
  --timeout 300s \
  --min-instances 0 \
  --max-instances 10 \
  --allow-unauthenticated

echo "Deployment completed successfully!"
echo "Service URL: $(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')"
```

## Monitoring and Observability

### Application Monitoring

#### Health Checks
```python
# app/health.py
from fastapi import APIRouter
import httpx
import logging

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint with dependency status"""
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": {
            "qdrant": await check_qdrant_health(),
            "database": await check_database_health(),
            "gemini_api": await check_gemini_health()
        }
    }

    # Overall status
    overall_healthy = all(checks["dependencies"].values())
    checks["overall_status"] = "healthy" if overall_healthy else "degraded"

    return checks

async def check_qdrant_health():
    try:
        # Check Qdrant connection
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.qdrant_url}/collections",
                headers={"api-key": settings.qdrant_api_key}
            )
            return response.status_code == 200
    except:
        return False

async def check_database_health():
    try:
        # Check database connection
        from app.db.session import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except:
        return False
```

#### Metrics Collection
```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_SESSIONS = Gauge(
    'active_sessions',
    'Number of active chat sessions'
)

# In your middleware
class MetricsMiddleware:
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start_time = time.time()
        method = scope["method"]
        path = scope["path"]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                REQUEST_COUNT.labels(method=method, endpoint=path, status=status).inc()
                duration = time.time() - start_time
                REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
            return await send(message)

        return await self.app(scope, receive, send_wrapper)
```

### Infrastructure Monitoring

#### Cloud Run Monitoring
- **CPU Utilization**: Monitor for scaling needs
- **Memory Usage**: Track memory consumption
- **Request Latency**: Monitor response times
- **Error Rates**: Track HTTP error rates
- **Traffic**: Monitor request volume

#### Database Monitoring
- **Connection Count**: Monitor active connections
- **Query Performance**: Track slow queries
- **Storage Usage**: Monitor database size
- **Replication Lag**: For multi-region setups

#### Vector Database Monitoring
- **Index Performance**: Monitor search latency
- **Storage Usage**: Track vector storage
- **Query Volume**: Monitor search requests
- **Memory Usage**: Track Qdrant memory consumption

## Backup and Recovery

### Database Backup Strategy

#### Neon Postgres Backup
- **Continuous Backup**: Neon provides built-in continuous backup
- **Point-in-Time Recovery**: Available for production plans
- **Branching**: Use Neon's branching for backup and restore

#### Backup Scripts
```bash
#!/bin/bash
# backup.sh

# Create database backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Upload to cloud storage
gsutil cp backup_*.sql gs://$BACKUP_BUCKET/backups/

# Clean up old backups (keep last 30 days)
find . -name "backup_*.sql" -mtime +30 -delete
```

### Disaster Recovery Plan

1. **Database Recovery**
   - Use Neon's point-in-time recovery
   - Restore from backup if needed

2. **Service Recovery**
   - Redeploy from latest successful build
   - Rollback to previous version if needed

3. **Data Recovery**
   - Restore vector embeddings from backup
   - Re-process documents if vectors are lost

## Security Considerations

### Network Security
- **VPC**: Use VPC for internal communication
- **Firewall Rules**: Restrict access to necessary ports
- **Private Connections**: Use private connections between services

### API Security
- **Rate Limiting**: Implement rate limiting per API key
- **Authentication**: Validate API keys for all endpoints
- **Input Validation**: Validate all user inputs
- **Output Sanitization**: Sanitize all API responses

### Secret Management
- **Cloud Secret Manager**: Store secrets in cloud provider
- **Environment Variables**: Inject secrets as environment variables
- **No Hardcoded Secrets**: Never commit secrets to code

## Performance Optimization

### Caching Strategy
```python
# app/cache.py
import redis
from functools import wraps
import json

class CacheManager:
    def __init__(self, redis_url):
        self.redis = redis.from_url(redis_url)

    def cached(self, ttl=3600):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

                # Try to get from cache
                cached_result = self.redis.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)

                # Execute function and cache result
                result = await func(*args, **kwargs)
                self.redis.setex(
                    cache_key,
                    ttl,
                    json.dumps(result, default=str)
                )
                return result
            return wrapper
        return decorator

cache = CacheManager(os.getenv("REDIS_URL"))
```

### CDN Strategy
- **Static Assets**: Serve frontend assets via CDN
- **API Gateway**: Use API gateway for request routing
- **Edge Caching**: Cache common responses at edge locations

## Rollback Strategy

### Deployment Rollback
1. **Version Tracking**: Keep track of deployed versions
2. **Health Monitoring**: Monitor health after deployment
3. **Automatic Rollback**: Rollback if health checks fail
4. **Manual Rollback**: Easy rollback to previous version

#### Rollback Script
```bash
#!/bin/bash
# rollback.sh

SERVICE_NAME=$1
PREVIOUS_REVISION=$2

gcloud run services update-traffic $SERVICE_NAME \
  --to-revisions $PREVIOUS_REVISION=100 \
  --region $REGION
```

## Environment Promotion

### Promotion Process
1. **Code Promotion**: Merge from development to staging to production
2. **Configuration Promotion**: Promote environment-specific configurations
3. **Database Migrations**: Apply database changes in sequence
4. **Data Promotion**: Promote necessary data between environments

### Environment-Specific Configurations

#### Development
```yaml
# config/development.yaml
debug: true
log_level: DEBUG
rate_limit_requests: 1000
max_content_length: 10485760  # 10MB
```

#### Staging
```yaml
# config/staging.yaml
debug: false
log_level: INFO
rate_limit_requests: 100
monitoring_enabled: true
```

#### Production
```yaml
# config/production.yaml
debug: false
log_level: WARNING
rate_limit_requests: 60
max_content_length: 5242880  # 5MB
monitoring_enabled: true
alert_thresholds:
  error_rate: 0.01  # 1%
  latency_p95: 2.0  # 2 seconds
```

## Operational Runbook

### Common Operations

#### Deploy New Version
```bash
./deploy.sh PROJECT_ID [REGION]
```

#### Check Service Health
```bash
curl https://SERVICE-NAME.REGION.run.app/health
```

#### View Logs
```bash
gcloud run services logs read rag-chatbot --region=REGION
```

#### Scale Service
```bash
gcloud run services update rag-chatbot \
  --region=REGION \
  --memory=2Gi \
  --cpu=2 \
  --max-instances=20
```

### Troubleshooting Guide

#### Common Issues and Solutions

1. **High Latency**
   - Check Qdrant performance
   - Verify Gemini API response times
   - Review embedding pipeline performance

2. **Service Unavailable**
   - Check all dependency health
   - Verify API keys and connections
   - Review error logs

3. **Database Connection Issues**
   - Check connection pool settings
   - Verify database credentials
   - Review network connectivity

4. **Memory Issues**
   - Review memory usage patterns
   - Optimize data processing
   - Increase memory allocation if needed

This deployment strategy provides a comprehensive approach to deploying the RAG Chatbot system with proper infrastructure, security, monitoring, and operational considerations.