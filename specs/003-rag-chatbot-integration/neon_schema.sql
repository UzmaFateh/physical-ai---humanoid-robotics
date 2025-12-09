# Neon Postgres Schema: RAG Chatbot

## Database Configuration

### Connection Settings
```sql
-- Database URL format for Neon
-- postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require

-- Recommended connection pooling settings
-- Max connections: 20 (for starter plan)
-- Connection timeout: 30 seconds
-- Idle timeout: 300 seconds
```

## Schema Design

### 1. Conversations Table
```sql
-- Table for storing conversation sessions
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    INDEX idx_conversations_session_id (session_id),
    INDEX idx_conversations_created_at (created_at),
    INDEX idx_conversations_updated_at (updated_at)
);

-- Add comments for documentation
COMMENT ON TABLE conversations IS 'Stores conversation sessions with metadata';
COMMENT ON COLUMN conversations.id IS 'Unique identifier for the conversation';
COMMENT ON COLUMN conversations.session_id IS 'Session identifier to group related conversations';
COMMENT ON COLUMN conversations.created_at IS 'Timestamp when conversation was created';
COMMENT ON COLUMN conversations.updated_at IS 'Timestamp when conversation was last updated';
COMMENT ON COLUMN conversations.metadata IS 'Additional metadata like user agent, source page, etc.';
```

### 2. Messages Table
```sql
-- Table for storing individual messages within conversations
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_chunks JSONB, -- IDs of chunks used in response
    tokens_used INTEGER,
    response_time_ms INTEGER,
    feedback_score INTEGER CHECK (feedback_score >= 1 AND feedback_score <= 5),
    INDEX idx_messages_conversation_id (conversation_id),
    INDEX idx_messages_timestamp (timestamp),
    INDEX idx_messages_role (role),
    INDEX idx_messages_feedback_score (feedback_score)
);

-- Add comments for documentation
COMMENT ON TABLE messages IS 'Stores individual messages within conversations';
COMMENT ON COLUMN messages.id IS 'Unique identifier for the message';
COMMENT ON COLUMN messages.conversation_id IS 'Reference to the parent conversation';
COMMENT ON COLUMN messages.role IS 'Role of the message sender (user/assistant)';
COMMENT ON COLUMN messages.content IS 'Content of the message';
COMMENT ON COLUMN messages.timestamp IS 'Timestamp when message was created';
COMMENT ON COLUMN messages.source_chunks IS 'JSON array of chunk IDs used to generate response';
COMMENT ON COLUMN messages.tokens_used IS 'Number of tokens in the message';
COMMENT ON COLUMN messages.response_time_ms IS 'Response time in milliseconds';
COMMENT ON COLUMN messages.feedback_score IS 'User feedback score (1-5)';
```

### 3. User Feedback Table (Optional - for advanced analytics)
```sql
-- Table for detailed feedback beyond simple scores
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_feedback TEXT,
    is_positive BOOLEAN,
    category VARCHAR(50), -- 'accuracy', 'relevance', 'helpfulness', etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX idx_feedback_message_id (message_id),
    INDEX idx_feedback_conversation_id (conversation_id),
    INDEX idx_feedback_category (category),
    INDEX idx_feedback_created_at (created_at)
);

COMMENT ON TABLE feedback IS 'Detailed feedback for messages';
COMMENT ON COLUMN feedback.user_feedback IS 'Text feedback from user';
COMMENT ON COLUMN feedback.is_positive IS 'Whether feedback is positive (true/false)';
COMMENT ON COLUMN feedback.category IS 'Category of feedback';
```

### 4. Document Metadata Table (Optional - for tracking document processing)
```sql
-- Table for tracking processed documents
CREATE TABLE document_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url VARCHAR(2000) NOT NULL UNIQUE,
    title VARCHAR(500),
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    chunk_count INTEGER,
    word_count INTEGER,
    status VARCHAR(20) DEFAULT 'processed', -- 'pending', 'processed', 'failed'
    processing_error TEXT,
    metadata JSONB,
    INDEX idx_document_metadata_source_url (source_url),
    INDEX idx_document_metadata_status (status),
    INDEX idx_document_metadata_processed_at (processed_at)
);

COMMENT ON TABLE document_metadata IS 'Metadata for processed documents';
COMMENT ON COLUMN document_metadata.source_url IS 'URL of the source document';
COMMENT ON COLUMN document_metadata.title IS 'Title of the document';
COMMENT ON COLUMN document_metadata.processed_at IS 'When the document was processed';
COMMENT ON COLUMN document_metadata.chunk_count IS 'Number of chunks created';
COMMENT ON COLUMN document_metadata.word_count IS 'Total word count of the document';
COMMENT ON COLUMN document_metadata.status IS 'Processing status';
COMMENT ON COLUMN document_metadata.processing_error IS 'Error message if processing failed';
```

## Indexing Strategy

### Primary Indexes
- `conversations.session_id`: For efficient session retrieval
- `messages.conversation_id`: For fetching messages by conversation
- `messages.timestamp`: For time-based queries and ordering

### Secondary Indexes
- `conversations.updated_at`: For finding recent conversations
- `messages.role`: For filtering by message role
- `messages.feedback_score`: For analytics on feedback
- `document_metadata.status`: For processing queue management

## Sample Queries

### 1. Retrieve conversation history
```sql
SELECT
    c.id as conversation_id,
    c.session_id,
    c.created_at,
    c.updated_at,
    c.metadata,
    json_agg(
        json_build_object(
            'id', m.id,
            'role', m.role,
            'content', m.content,
            'timestamp', m.timestamp,
            'tokens_used', m.tokens_used,
            'source_chunks', m.source_chunks
        ) ORDER BY m.timestamp
    ) as messages
FROM conversations c
LEFT JOIN messages m ON c.id = m.conversation_id
WHERE c.session_id = $1
GROUP BY c.id, c.session_id, c.created_at, c.updated_at, c.metadata;
```

### 2. Get conversation analytics
```sql
SELECT
    c.session_id,
    COUNT(m.id) as message_count,
    MAX(m.timestamp) as last_message_time,
    AVG(m.tokens_used) as avg_tokens_per_message,
    AVG(m.feedback_score) as avg_feedback_score
FROM conversations c
LEFT JOIN messages m ON c.id = m.conversation_id
WHERE c.created_at >= NOW() - INTERVAL '7 days'
GROUP BY c.session_id
ORDER BY last_message_time DESC;
```

### 3. Find messages with low feedback
```sql
SELECT
    m.id,
    m.content,
    m.feedback_score,
    m.source_chunks,
    c.session_id
FROM messages m
JOIN conversations c ON m.conversation_id = c.id
WHERE m.feedback_score <= 2
    AND m.role = 'assistant'
    AND m.timestamp >= NOW() - INTERVAL '1 day'
ORDER BY m.timestamp DESC;
```

## Security Considerations

### Row-Level Security (RLS)
```sql
-- Enable RLS if multi-tenancy is required
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Create policies for tenant isolation
CREATE POLICY conversations_tenant_isolation ON conversations
    FOR ALL TO authenticated_users
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY messages_tenant_isolation ON messages
    FOR ALL TO authenticated_users
    USING (
        conversation_id IN (
            SELECT id FROM conversations
            WHERE tenant_id = current_setting('app.current_tenant')::UUID
        )
    );
```

### Data Retention Policy
```sql
-- Create a function to archive old conversations
CREATE OR REPLACE FUNCTION archive_old_conversations()
RETURNS void AS $$
BEGIN
    -- Move conversations older than 90 days to archive table
    INSERT INTO conversations_archive
    SELECT * FROM conversations
    WHERE created_at < NOW() - INTERVAL '90 days';

    -- Delete from main table
    DELETE FROM conversations
    WHERE created_at < NOW() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- Create a scheduled job to run the archiving function
-- This would be set up in Neon's scheduled jobs feature
```

## Performance Optimization

### Connection Pooling
- Use Neon's built-in connection pooling
- Implement application-level connection pooling with SQLAlchemy
- Set appropriate timeout values to handle connection issues

### Partitioning Strategy
```sql
-- Example of time-based partitioning for messages table
-- This would be implemented if the table grows very large

CREATE TABLE messages_partitioned (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_chunks JSONB,
    tokens_used INTEGER,
    response_time_ms INTEGER,
    feedback_score INTEGER CHECK (feedback_score >= 1 AND feedback_score <= 5)
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE messages_2025_01 PARTITION OF messages_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE messages_2025_02 PARTITION OF messages_partitioned
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
-- ... continue for each month
```

## Backup and Recovery

### Automated Backups
- Leverage Neon's built-in continuous backup
- Set up point-in-time recovery (PITR) for production environments
- Regular export of critical data for disaster recovery

### Data Validation Queries
```sql
-- Check for orphaned messages (messages without valid conversations)
SELECT COUNT(*) FROM messages m
LEFT JOIN conversations c ON m.conversation_id = c.id
WHERE c.id IS NULL;

-- Check for conversations with no messages
SELECT COUNT(*) FROM conversations c
LEFT JOIN messages m ON c.id = m.conversation_id
WHERE m.id IS NULL;
```