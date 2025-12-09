/**
 * API Integration Layer for RAG Chatbot
 * Handles all communication with the backend API
 */

class APIIntegration {
    constructor(apiEndpoint, apiKey) {
        this.apiEndpoint = apiEndpoint || 'http://localhost:8000';
        this.apiKey = apiKey;
    }

    // Set API key after initialization
    setApiKey(apiKey) {
        this.apiKey = apiKey;
    }

    // Make an authenticated API request
    async makeRequest(endpoint, method = 'GET', data = null) {
        const url = `${this.apiEndpoint}/api/v1${endpoint}`;
        const config = {
            method,
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            }
        };

        if (data) {
            config.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request error:', error);
            throw error;
        }
    }

    // RAG Query endpoint
    async queryRAG(query, selectedText, sessionId = null, sourceUrl = null) {
        const payload = {
            query,
            selected_text: selectedText,
            session_id: sessionId,
            source_url: sourceUrl
        };

        return await this.makeRequest('/rag/query', 'POST', payload);
    }

    // RAG Search endpoint
    async searchRAG(query, topK = 5, sourceUrl = null) {
        const payload = {
            query,
            top_k: topK,
            source_url: sourceUrl
        };

        return await this.makeRequest('/rag/search', 'POST', payload);
    }

    // Health check endpoint
    async healthCheck() {
        return await this.makeRequest('/rag/health', 'GET');
    }

    // Process embeddings endpoint
    async processDocuments(documents, chunkSize = 512, overlap = 50) {
        const payload = {
            documents,
            chunk_size: chunkSize,
            overlap
        };

        return await this.makeRequest('/embeddings/process', 'POST', payload);
    }

    // Refresh embeddings endpoint
    async refreshDocuments(urls, forceRecreate = false) {
        const payload = {
            urls,
            force_recreate: forceRecreate
        };

        return await this.makeRequest('/embeddings/refresh', 'POST', payload);
    }

    // Delete documents endpoint
    async deleteDocuments(urls) {
        const payload = { urls };
        return await this.makeRequest('/embeddings/documents', 'DELETE', payload);
    }

    // Create conversation endpoint
    async createConversation(metadata = null) {
        const payload = { metadata };
        return await this.makeRequest('/conversations', 'POST', payload);
    }

    // Get conversation endpoint
    async getConversation(sessionId) {
        return await this.makeRequest(`/conversations/${sessionId}`, 'GET');
    }

    // Add message to conversation endpoint
    async addMessageToConversation(sessionId, role, content, tokensUsed = null, sourceChunks = null) {
        const payload = {
            role,
            content,
            tokens_used: tokensUsed,
            source_chunks: sourceChunks
        };

        return await this.makeRequest(`/conversations/${sessionId}/messages`, 'POST', payload);
    }

    // Submit feedback endpoint
    async submitFeedback(sessionId, messageId, score, comment = null) {
        const payload = {
            message_id: messageId,
            score,
            comment
        };

        return await this.makeRequest(`/conversations/${sessionId}/feedback`, 'POST', payload);
    }

    // Get stats endpoint
    async getStats() {
        return await this.makeRequest('/conversations/stats', 'GET');
    }
}

// Export as a global variable or module
if (typeof module !== 'undefined' && module.exports) {
    module.exports = APIIntegration;
} else {
    window.APIIntegration = APIIntegration;
}