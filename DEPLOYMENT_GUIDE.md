# RAG Chatbot Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the RAG Chatbot system with FastAPI backend, Qdrant vector database, and Neon Postgres for conversation storage. The system is designed to be deployed in production environments with scalability and security in mind.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Docusaurus    │────│   FastAPI        │────│   Qdrant        │
│   Documentation │    │   Backend        │    │   Vector DB     │
│   Site          │    │   (Production)   │    │   (Cloud/Hosted)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              │
                       ┌──────────────────┐
                       │   Neon Postgres  │
                       │   (Serverless)   │
                       └──────────────────┘
```

## Prerequisites

### Infrastructure Requirements
- **Backend Server**: Linux server with Python 3.9+, 4GB+ RAM, 2+ CPU cores
- **Qdrant**: Cloud instance or self-hosted (4GB+ RAM recommended)
- **Neon Postgres**: Serverless or dedicated Postgres instance
- **Domain**: SSL-enabled domain for production deployment

### API Keys and Services
- Google Gemini API key
- Qdrant Cloud cluster (or self-hosted instance)
- Neon Postgres database
- SSL certificate (for production)

## Backend Deployment

### 1. Server Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3.9 python3.9-venv python3.9-dev build-essential -y

# Install system dependencies
sudo apt install postgresql-client -y
```

### 2. Clone and Setup Application

```bash
# Clone the repository (or copy the backend code)
git clone <repository-url>
cd physical-ai-humanoid-robotics/backend

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file with production settings:

```bash
# backend/.env
DATABASE_URL=postgresql://username:password@neon-host.region.neon.tech/dbname
QDRANT_URL=https://your-cluster.your-region.qdrant.tech
QDRANT_API_KEY=your-qdrant-api-key
GEMINI_API_KEY=your-gemini-api-key
API_KEY=your-production-api-key
ENVIRONMENT=production
LOG_LEVEL=info
SECRET_KEY=your-very-long-secret-key-here
```

### 4. Database Migrations

The application will automatically create tables on first run. For Neon Postgres:

```bash
# Verify database connection
python -c "from app.services.database_service import engine; print('Database connection successful')"
```

### 5. Process Management with Gunicorn

Install Gunicorn for production:

```bash
pip install gunicorn

# Create gunicorn configuration
cat > gunicorn.conf.py << EOF
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
preload_app = True
EOF
```

### 6. Start the Application

```bash
gunicorn app.main:app -c gunicorn.conf.py
```

### 7. Set up Process Management with Systemd

Create a systemd service file:

```bash
sudo tee /etc/systemd/system/rag-chatbot.service << EOF
[Unit]
Description=RAG Chatbot API
After=network.target

[Service]
Type=notify
User=your-user
Group=your-user
WorkingDirectory=/path/to/physical-ai-humanoid-robotics/backend
EnvironmentFile=/path/to/physical-ai-humanoid-robotics/backend/.env
ExecStart=/path/to/physical-ai-humanoid-robotics/backend/venv/bin/gunicorn app.main:app -c gunicorn.conf.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable rag-chatbot
sudo systemctl start rag-chatbot
sudo systemctl status rag-chatbot
```

## Frontend Integration with Docusaurus

### 1. Build and Deploy Docusaurus Site

```bash
# In your Docusaurus project directory
npm run build

# Copy the static files to your web server
# The rag-chatbot-embed.js and rag-chatbot.css files should be available
cp static/rag-chatbot-embed.js build/
cp static/css/rag-chatbot.css build/css/
```

### 2. Update Docusaurus Configuration

Add to your `docusaurus.config.js`:

```js
module.exports = {
  // ... other config
  scripts: [
    {
      src: '/rag-chatbot-embed.js',
      async: true,
      defer: true
    }
  ],
  stylesheets: [
    {
      href: '/css/rag-chatbot.css',
      type: 'text/css'
    }
  ]
};
```

### 3. Configure Widget Settings

Add configuration script to your HTML template or docusaurus config:

```js
// Add this to your page template or as a separate script
window.RAG_CHATBOT_CONFIG = {
  apiEndpoint: 'https://your-backend-domain.com',
  apiKey: 'your-api-key',
  title: 'Documentation Assistant',
  placeholder: 'Ask about this documentation...'
};
```

## SSL and Reverse Proxy Setup

### Using Nginx

Install and configure Nginx:

```bash
sudo apt install nginx -y
sudo tee /etc/nginx/sites-available/rag-chatbot << EOF
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # WebSocket support for future features
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Serve static files for frontend integration
    location /static/ {
        alias /path/to/your/docusaurus/build/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/rag-chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Monitoring and Logging

### 1. Application Logs

Set up log rotation:

```bash
sudo tee /etc/logrotate.d/rag-chatbot << EOF
/var/log/rag-chatbot/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 your-user your-user
    postrotate
        systemctl reload rag-chatbot
    endscript
}
EOF
```

### 2. Health Checks

Add to your monitoring system:

```bash
# Health check endpoint
curl -f https://your-domain.com/health || exit 1
```

### 3. Performance Monitoring

Monitor these key metrics:
- Response times (should be < 5 seconds for 90% of requests)
- Error rates
- API usage
- Database connection pool usage
- Vector database query performance

## Security Configuration

### 1. API Rate Limiting

The application includes built-in rate limiting (60 requests per minute per API key).

### 2. SSL/TLS

Ensure SSL is properly configured with strong cipher suites.

### 3. API Key Management

- Use strong, randomly generated API keys
- Rotate keys regularly
- Monitor API key usage
- Implement key revocation procedures

### 4. Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

## Scaling Considerations

### Horizontal Scaling
- Multiple backend instances behind a load balancer
- Shared Qdrant and Neon Postgres instances
- Shared Redis for session management (if needed)

### Performance Optimization
- Qdrant collection optimization with proper indexing
- Neon Postgres connection pooling
- Caching for frequently accessed embeddings
- CDN for static assets

## Backup and Recovery

### 1. Database Backup (Neon Postgres)
Neon provides automatic backups, but you can also set up manual backups:

```bash
# Create backup
pg_dump "postgresql://username:password@host:port/dbname" > backup.sql
```

### 2. Qdrant Backup
Use Qdrant's snapshot functionality or implement a custom backup strategy for your vector collections.

## Troubleshooting

### Common Issues

1. **Application won't start**: Check logs with `journalctl -u rag-chatbot -f`
2. **Database connection errors**: Verify connection string and network access
3. **Qdrant connection errors**: Check API key and network access
4. **High response times**: Check database and vector store performance

### Log Locations

- Application logs: `/var/log/rag-chatbot/app.log`
- Error logs: `/var/log/rag-chatbot/error.log`
- System logs: `journalctl -u rag-chatbot`

## Maintenance Tasks

### 1. Regular Maintenance
- Monitor disk space and database size
- Rotate logs weekly
- Update dependencies monthly
- Review and clean up old conversations

### 2. Database Maintenance
```sql
-- Clean up old conversations (e.g., older than 30 days)
DELETE FROM conversations WHERE created_at < NOW() - INTERVAL '30 days';
```

## Rollback Procedure

To rollback to a previous version:

1. Stop the service: `sudo systemctl stop rag-chatbot`
2. Revert code changes
3. Restore from backup if needed
4. Start the service: `sudo systemctl start rag-chatbot`
5. Verify functionality

## Support and Monitoring

Set up alerts for:
- Service downtime
- High error rates (> 5%)
- Slow response times (> 10 seconds)
- Database connection issues
- Qdrant service issues