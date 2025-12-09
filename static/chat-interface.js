/**
 * Chat Interface Component for RAG Chatbot
 * Provides the UI for the chat interface
 */

class ChatInterface {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.options = {
            title: options.title || 'Documentation Assistant',
            placeholder: options.placeholder || 'Ask a question about the selected text...',
            position: options.position || 'bottom-right',
            onClosed: options.onClosed || null,  // Callback when chat is closed
            onOpened: options.onOpened || null,  // Callback when chat is opened
            ...options
        };
        this.isOpen = false;
        this.messages = [];
        this.currentSessionId = null;

        this.init();
    }

    init() {
        this.createChatInterface();
        this.attachEventListeners();
    }

    createChatInterface() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        const chatHtml = `
            <div id="rag-chat-widget" class="rag-chat-widget hidden">
                <div class="rag-chat-header">
                    <div class="rag-chat-title">${this.options.title}</div>
                    <button id="rag-chat-close" class="rag-chat-close-btn">&times;</button>
                </div>
                <div id="rag-chat-messages" class="rag-chat-messages">
                    <div class="rag-welcome-message">
                        Select text on the page and ask questions about it!
                    </div>
                </div>
                <div class="rag-chat-input-area">
                    <textarea
                        id="rag-chat-input"
                        class="rag-chat-input"
                        placeholder="${this.options.placeholder}"
                        rows="1"
                    ></textarea>
                    <button id="rag-chat-send" class="rag-chat-send-btn">Send</button>
                </div>
                <div id="rag-chat-loading" class="rag-chat-loading hidden">
                    <div class="rag-loading-spinner"></div>
                    <span>Thinking...</span>
                </div>
            </div>
        `;

        container.innerHTML = chatHtml;
    }

    attachEventListeners() {
        // Close button
        document.getElementById('rag-chat-close').addEventListener('click', () => {
            this.close();
        });

        // Send button
        document.getElementById('rag-chat-send').addEventListener('click', () => {
            this.sendMessage();
        });

        // Input textarea
        const input = document.getElementById('rag-chat-input');
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        input.addEventListener('input', () => {
            this.autoResizeTextarea(input);
        });
    }

    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    open() {
        const widget = document.getElementById('rag-chat-widget');
        widget.classList.remove('hidden');
        this.isOpen = true;

        // Focus the input
        setTimeout(() => {
            document.getElementById('rag-chat-input').focus();
        }, 100);

        // Call the onOpened callback if provided
        if (this.options.onOpened) {
            this.options.onOpened();
        }
    }

    close() {
        const widget = document.getElementById('rag-chat-widget');
        widget.classList.add('hidden');
        this.isOpen = false;

        // Call the onClosed callback if provided
        if (this.options.onClosed) {
            this.options.onClosed();
        }
    }

    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }

    addMessage(role, content, sources = null) {
        const messagesContainer = document.getElementById('rag-chat-messages');

        // Remove welcome message if it's the first message
        if (this.messages.length === 0) {
            const welcomeMsg = messagesContainer.querySelector('.rag-welcome-message');
            if (welcomeMsg) welcomeMsg.remove();
        }

        const messageElement = document.createElement('div');
        messageElement.className = `rag-message rag-message-${role}`;

        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = '<div class="rag-sources"><strong>Sources:</strong><ul>';
            sources.forEach(source => {
                sourcesHtml += `<li><a href="${source.url}" target="_blank">${source.title}</a></li>`;
            });
            sourcesHtml += '</ul></div>';
        }

        messageElement.innerHTML = `
            <div class="rag-message-content">
                <div class="rag-message-text">${this.escapeHtml(content)}</div>
                ${sourcesHtml}
            </div>
        `;

        messagesContainer.appendChild(messageElement);
        this.scrollToBottom();

        // Add to internal messages array
        this.messages.push({ role, content, sources });
    }

    showLoading() {
        const loadingElement = document.getElementById('rag-chat-loading');
        loadingElement.classList.remove('hidden');
    }

    hideLoading() {
        const loadingElement = document.getElementById('rag-chat-loading');
        loadingElement.classList.add('hidden');
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('rag-chat-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    async sendMessage() {
        const input = document.getElementById('rag-chat-input');
        const message = input.value.trim();

        if (!message) return;

        // Clear input and resize
        input.value = '';
        input.style.height = 'auto';

        // Add user message to UI
        this.addMessage('user', message);

        // Show loading indicator
        this.showLoading();

        try {
            // This will be implemented with actual API call
            const response = await this.callAPI(message);

            // Add assistant response to UI
            this.addMessage('assistant', response.response, response.sources || []);
        } catch (error) {
            this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
            console.error('Chat error:', error);
        } finally {
            this.hideLoading();
        }
    }

    async callAPI(message) {
        // Placeholder for actual API call
        // In a real implementation, this would call the backend API
        // For now, returning a mock response
        return {
            response: `This is a mock response to your question: "${message}". In a real implementation, this would come from the RAG system.`,
            sources: []
        };
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    setCurrentSessionId(sessionId) {
        this.currentSessionId = sessionId;
    }

    getCurrentSessionId() {
        return this.currentSessionId;
    }
}

// Export as a global variable or module
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatInterface;
} else {
    window.ChatInterface = ChatInterface;
}