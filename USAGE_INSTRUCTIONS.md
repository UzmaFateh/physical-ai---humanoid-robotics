# RAG Chatbot Integration - Usage Instructions

## Overview

This document provides instructions for using the RAG (Retrieval-Augmented Generation) Chatbot that has been integrated into the Docusaurus documentation site. The chatbot allows users to select text from the documentation and ask questions about it, with responses generated using a Retrieval-Augmented Generation approach powered by Google Gemini, Qdrant vector database, and Neon Postgres for conversation storage.

## Features

1. **Contextual Q&A**: Select text on any documentation page and ask questions about it
2. **Conversation History**: View and continue previous conversations during the same session
3. **Source Attribution**: Responses include links to the source documentation pages
4. **Responsive Design**: Works on both desktop and mobile devices

## How to Use

### 1. Text Selection and Question Asking
- Highlight any text on the documentation page
- The chatbot widget will become more prominent (or click the floating button if not visible)
- Type your question in the input field
- Press Enter or click "Send" to submit

### 2. Viewing Conversation History
- All questions and answers from the current session are displayed in the chat interface
- Scroll up to review previous exchanges
- The conversation persists during your browsing session

### 3. Source References
- Each response includes source links to the documentation pages used
- Click on source links to navigate directly to referenced content

## Technical Architecture

### Backend Components
- **FastAPI**: Web framework for the backend API
- **Qdrant**: Vector database for document embeddings and retrieval
- **Neon Postgres**: Serverless Postgres for conversation history storage
- **Google Gemini**: LLM for generating responses

### Frontend Components
- **Web Component**: Embeddable chat widget using custom elements
- **Text Selection API**: Captures user-selected text
- **CSS Styling**: Responsive design that matches the documentation site

## API Endpoints

### RAG Endpoints
- `POST /api/v1/rag/query` - Main RAG query endpoint
- `POST /api/v1/rag/search` - Search for relevant chunks without generating response
- `GET /api/v1/rag/health` - Health check endpoint

### Embedding Endpoints
- `POST /api/v1/embeddings/process` - Process and store document chunks
- `POST /api/v1/embeddings/refresh` - Refresh embeddings for specific content
- `DELETE /api/v1/embeddings/documents` - Remove document chunks

### Conversation Endpoints
- `POST /api/v1/conversations` - Create new conversation session
- `GET /api/v1/conversations/{session_id}` - Retrieve conversation history
- `POST /api/v1/conversations/{session_id}/messages` - Add message to conversation
- `POST /api/v1/conversations/{session_id}/feedback` - Submit feedback
- `GET /api/v1/conversations/stats` - Get system statistics

## Environment Configuration

Create a `.env` file with the following variables:

```env
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/rag_chatbot

# Qdrant Configuration
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your-qdrant-api-key

# Gemini Configuration
GEMINI_API_KEY=your-gemini-api-key

# Application Configuration
API_KEY=your-api-key-for-authentication
ENVIRONMENT=development
LOG_LEVEL=info
```

## Deployment Guide

### Backend Deployment

1. **Prerequisites**:
   - Python 3.9+
   - Qdrant Cloud account or local Qdrant instance
   - Neon Serverless Postgres database
   - Google Gemini API key

2. **Setup**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r backend/requirements.txt

   # Set environment variables
   cp backend/.env.example backend/.env
   # Edit backend/.env with your actual values
   ```

3. **Run the application**:
   ```bash
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. **Using Docker**:
   ```bash
   cd backend
   docker-compose up -d
   ```

### Frontend Integration

1. **For Docusaurus sites**, add the following to your `docusaurus.config.js` in the `scripts` section:

   ```js
   module.exports = {
     // ... other config
     scripts: [
       // ... other scripts
       {
         src: '/rag-chatbot-embed.js',
         async: true,
         defer: true
       }
     ],
   };
   ```

2. **Alternative manual integration**: Add the following script to your HTML:

   ```html
   <script>
     window.RAG_CHATBOT_CONFIG = {
       apiEndpoint: 'https://your-backend-api.com',
       apiKey: 'your-api-key'
     };
   </script>
   <script src="/rag-chatbot-embed.js"></script>
   <link rel="stylesheet" href="/css/rag-chatbot.css">
   ```

### Configuration Options

The chatbot can be configured by setting the `RAG_CHATBOT_CONFIG` object:

```js
window.RAG_CHATBOT_CONFIG = {
  apiEndpoint: 'https://your-backend-api.com',  // Backend API URL
  apiKey: 'your-api-key',                       // API key for authentication
  title: 'Documentation Assistant',             // Widget title
  placeholder: 'Ask about this documentation...' // Input placeholder text
};
```

## Security Considerations

- API keys are required for all backend endpoints
- Rate limiting is implemented to prevent abuse
- Input sanitization is performed on all user inputs
- Use HTTPS in production environments

## Troubleshooting

### Common Issues

1. **Widget not appearing**: Check that the embed script is loaded and CSS is applied
2. **API errors**: Verify that the backend is running and API keys are correct
3. **Text selection not working**: Ensure no JavaScript errors are preventing the selection API from functioning

### Error Messages

- "Invalid API Key": Check your API key configuration
- "Service unavailable": Verify the backend service is running
- "Rate limit exceeded": Wait before making additional requests

## Development

To run the application in development mode:

1. Start the backend:
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. For the frontend, ensure the Docusaurus site is running:
   ```bash
   npm run start
   ```

## Monitoring and Maintenance

- Monitor API response times and error rates
- Regularly review conversation logs for quality assurance
- Update documentation embeddings as content changes
- Monitor database and vector store usage

## Support

For technical support, please contact the development team or refer to the project documentation.