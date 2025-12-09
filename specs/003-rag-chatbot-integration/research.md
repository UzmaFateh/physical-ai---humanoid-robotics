# Research Document: RAG Chatbot Implementation

## Executive Summary

This document provides research and analysis for the RAG Chatbot implementation, addressing key unknowns and technical decisions required for the project. The research covers Qdrant integration, Gemini API usage, Neon Postgres schema design, Docusaurus integration, and frontend widget architecture.

## 1. Qdrant Cloud Integration Research

### 1.1 Optimal Collection Schema

**Decision**: Use Qdrant collection with structured payload for document chunks

**Rationale**:
- Qdrant supports both dense and sparse vector storage with rich payload capabilities
- Structured payloads allow for efficient filtering and metadata retrieval
- Vector dimensions of 1536 (compatible with OpenAI embeddings) or 768 (compatible with Sentence Transformers) are standard

**Collection Configuration**:
```json
{
  "collection_name": "doc_chunks",
  "vector_size": 1536,
  "distance": "Cosine",
  "hnsw_config": {
    "m": 16,
    "ef_construct": 100,
    "full_scan_threshold": 10000
  },
  "optimizers_config": {
    "deleted_threshold": 0.2,
    "vacuum_min_vector_number": 1000
  }
}
```

**Payload Structure**:
```json
{
  "content": "Text content of the document chunk",
  "source_url": "URL where the content originated",
  "page_title": "Title of the source page",
  "section": "Section or heading name",
  "chunk_order": 0,
  "metadata": {
    "tags": ["tag1", "tag2"],
    "last_updated": "2025-12-09T00:00:00Z"
  }
}
```

### 1.2 Similarity Search Best Practices

**Decision**: Use cosine distance with filtering and re-ranking

**Rationale**:
- Cosine distance works well for semantic similarity in high-dimensional spaces
- Filtering by source URL can ensure relevance to specific documents
- Re-ranking with rerankers library can improve quality

**Search Parameters**:
- Top-k: 5-10 most similar chunks
- Score threshold: 0.5 minimum similarity score
- Prefiltering: Filter by specific document URLs if needed

### 1.3 Performance Characteristics

**Findings**:
- Qdrant Cloud offers horizontal scaling with multiple replicas
- Query performance: <100ms for typical similarity searches
- Vector storage efficiency: Can handle millions of vectors with proper indexing
- Rate limits: Typically high (thousands of requests per second) but vary by plan

## 2. Gemini API Integration Research

### 2.1 OpenAI-Compatible Interface Patterns

**Decision**: Use Google's Generative AI SDK with custom wrapper to mimic OpenAI interface

**Rationale**:
- Google's official SDK provides better integration than unofficial OpenAI-compatible libraries
- Custom wrapper allows for consistent interface across different LLM providers
- Official SDK has better error handling and documentation

**Implementation Pattern**:
```python
class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.GenerativeModel('gemini-pro')

    def chat_completions_create(self, messages: List[Dict], **kwargs):
        # Convert OpenAI-style messages to Gemini format
        # Process and return response in OpenAI-compatible format
        pass
```

### 2.2 Prompt Engineering for RAG

**Decision**: Use context-aware RAG prompting with selected text prioritization

**Rationale**:
- Selected text should be prioritized in the context window
- Relevant chunks from Qdrant should be included with clear source attribution
- Prompt engineering significantly affects response quality

**Prompt Template**:
```
You are a helpful assistant for documentation. Answer the user's question based on the provided context.

Selected Text: {selected_text}

Additional Context:
{retrieved_chunks_formatted}

Question: {user_question}

Please provide a helpful and accurate answer based on the provided context. If the context doesn't contain the information needed, say so clearly.
```

### 2.3 Token Limits and Cost Implications

**Findings**:
- Gemini Pro: 32,768 tokens input, 2,048 tokens output
- Pricing: ~$0.50/1M input tokens, ~$1.50/1M output tokens
- Rate limits: ~60 requests per minute per API key (varies by region)

**Cost Optimization Strategies**:
- Chunk documents to fit within context window efficiently
- Implement caching for frequently asked questions
- Use streaming responses to improve user experience

## 3. Neon Serverless Postgres Research

### 3.1 Optimal Schema for Conversation Storage

**Decision**: Use two related tables for conversations and messages

**Rationale**:
- Separation of concerns between conversation sessions and individual messages
- Efficient querying of conversation history
- Proper indexing for performance

**Schema Design**:
```sql
-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_chunks JSONB, -- IDs of chunks used in response
    tokens_used INTEGER
);

-- Indexes for performance
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_conversations_session_id ON conversations(session_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);
```

### 3.2 Connection Pooling and Performance

**Findings**:
- Neon Serverless automatically handles connection pooling
- Serverless compute scales based on demand
- Connection limits: Up to 20 connections for free tier, more for paid tiers
- Performance: Sub-millisecond response times for simple queries

### 3.3 Cost Implications

**Findings**:
- Compute: Charged per-second when active
- Storage: Charged per GB per month
- Data transfer: Charged based on data egress
- Free tier: 500MB storage, 10M compute seconds per month

## 4. Docusaurus Integration Research

### 4.1 Custom Script Injection Methods

**Decision**: Use Docusaurus plugin system with custom component injection

**Rationale**:
- Docusaurus provides clean plugin architecture for custom scripts
- Components can be injected via swizzling or plugin system
- Minimal impact on build process and performance

**Integration Approaches**:
1. **Docusaurus Plugin**: Create a dedicated plugin that injects the chat widget
2. **Custom Layout**: Modify the layout to include the widget
3. **MDX Components**: Use MDX components for specific page integration

**Recommended Approach**:
- Use plugin system to inject script tag in document head
- Add container div in layout for widget mounting
- Provide configuration options via plugin options

### 4.2 Text Selection API Implementation

**Decision**: Use browser's Selection API with custom event handling

**Rationale**:
- Native browser API provides reliable text selection detection
- Custom event handling allows for precise control over widget activation
- Cross-browser compatibility is good for modern browsers

**Implementation Pattern**:
```javascript
// Detect text selection
document.addEventListener('mouseup', handleTextSelection);

function handleTextSelection() {
  const selection = window.getSelection();
  if (selection.toString().trim().length > 0) {
    // Show chat widget with selected text context
    showChatWidget(selection.toString().trim());
  }
}
```

### 4.3 Performance Optimization

**Findings**:
- Widget should be loaded asynchronously to avoid blocking page rendering
- Lazy loading can improve initial page load times
- Caching strategies can reduce API calls for repeated content

## 5. Frontend Widget Architecture Research

### 5.1 Embeddable UI Component Best Practices

**Decision**: Use vanilla JavaScript with Web Components for maximum compatibility

**Rationale**:
- Works with any frontend framework
- Minimal dependencies
- Easy to embed in any HTML page
- Better performance than heavy frameworks for simple UI

**Architecture Pattern**:
```javascript
class RAGChatWidget extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    this.render();
    this.setupEventListeners();
  }

  render() {
    // Create widget UI using template literals or shadow DOM
  }
}
```

### 5.2 State Management Requirements

**Decision**: Use custom state management within the widget

**Rationale**:
- External state management libraries add unnecessary complexity
- Self-contained state management improves performance
- Easier to maintain and debug

**State Structure**:
```javascript
{
  isVisible: boolean,
  selectedText: string,
  conversation: Array,
  isLoading: boolean,
  error: string
}
```

### 5.3 Responsive Design Considerations

**Findings**:
- Widget should adapt to different screen sizes
- Mobile-first approach recommended for better UX
- Positioning should not interfere with page content
- Accessibility standards (WCAG) should be followed

## 6. Security Research

### 6.1 API Key Management

**Decision**: Store API keys in environment variables with runtime validation

**Rationale**:
- Environment variables are the standard approach for secrets
- Runtime validation prevents misconfigurations
- Separation of development and production keys

### 6.2 Input Validation and Sanitization

**Decision**: Implement comprehensive input validation on both frontend and backend

**Rationale**:
- Prevents injection attacks
- Ensures data integrity
- Improves system reliability

**Validation Strategy**:
- Sanitize all user inputs (queries, selected text)
- Validate URL parameters and request bodies
- Implement rate limiting to prevent abuse

## 7. Performance Optimization Research

### 7.1 Caching Strategies

**Decision**: Implement multi-layer caching (CDN, application, database)

**Rationale**:
- Reduces response times for common queries
- Decreases API costs
- Improves user experience

**Caching Layers**:
1. **CDN**: Cache static widget assets
2. **Application**: Cache conversation history and common responses
3. **Database**: Use Neon's built-in query caching

### 7.2 Load Balancing and Scaling

**Findings**:
- FastAPI with Uvicorn provides good performance for async operations
- Horizontal scaling possible with load balancer
- Qdrant Cloud handles vector search scaling automatically
- Neon Serverless scales compute automatically

## 8. Deployment Research

### 8.1 Backend Deployment Options

**Decision**: Deploy to cloud platform with containerization (Docker)

**Rationale**:
- Containerization ensures consistent deployment across environments
- Cloud platforms provide auto-scaling and monitoring
- Easy to manage dependencies and configurations

**Recommended Platforms**:
- Google Cloud Run (for cost-effective serverless)
- AWS Fargate
- Railway
- Render

### 8.2 Frontend Asset Optimization

**Findings**:
- Minimize and bundle JavaScript for smaller payload
- Use CDN for faster asset delivery
- Implement lazy loading for non-critical functionality
- Consider service workers for offline capabilities