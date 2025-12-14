/**
 * RAG Chat Widget - Web Component
 * Main entry point for the embeddable chat widget
 */

class RAGChatWidget extends HTMLElement {
    constructor() {
        super();
        this.apiEndpoint = this.getAttribute('api-endpoint') || 'http://localhost:8000';
        this.apiKey = this.getAttribute('api-key') || '';
        this.selectedText = '';
        this.sessionId = null;
    }

    connectedCallback() {
        this.render();
        this.initWidget();
    }

    render() {
        this.innerHTML = `
            <style>
                /* Include the CSS styles here */
                ${this.getCSS()}
            </style>
            <div id="rag-chat-container"></div>
        `;
    }

    getCSS() {
        // In a real implementation, this would load from the external CSS file
        // For now, we'll include a basic version
        return `
            .rag-chat-widget {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 380px;
                height: 500px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                display: flex;
                flex-direction: column;
                z-index: 10000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                overflow: hidden;
                border: 1px solid #e1e5e9;
            }

            .rag-chat-widget.hidden {
                display: none;
            }

            .rag-chat-header {
                background: #4f46e5;
                color: white;
                padding: 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #e1e5e9;
            }

            .rag-chat-title {
                font-weight: 600;
                font-size: 16px;
            }

            .rag-chat-close-btn {
                background: none;
                border: none;
                color: white;
                font-size: 24px;
                cursor: pointer;
                padding: 0;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .rag-chat-close-btn:hover {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 50%;
            }

            .rag-chat-messages {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 12px;
                background: #f9fafb;
            }

            .rag-welcome-message {
                text-align: center;
                color: #6b7280;
                font-style: italic;
                padding: 20px 0;
                font-size: 14px;
            }

            .rag-message {
                max-width: 85%;
                padding: 12px 16px;
                border-radius: 18px;
                font-size: 14px;
                line-height: 1.5;
                position: relative;
            }

            .rag-message-user {
                align-self: flex-end;
                background: #4f46e5;
                color: white;
                border-bottom-right-radius: 4px;
            }

            .rag-message-assistant {
                align-self: flex-start;
                background: white;
                color: #374151;
                border: 1px solid #e5e7eb;
                border-bottom-left-radius: 4px;
            }

            .rag-message-content {
                word-wrap: break-word;
            }

            .rag-sources {
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #e5e7eb;
                font-size: 12px;
            }

            .rag-sources ul {
                margin: 4px 0 0 0;
                padding-left: 16px;
            }

            .rag-sources li {
                margin-bottom: 4px;
            }

            .rag-sources a {
                color: #4f46e5;
                text-decoration: none;
            }

            .rag-sources a:hover {
                text-decoration: underline;
            }

            .rag-chat-input-area {
                padding: 12px 16px;
                background: white;
                border-top: 1px solid #e1e5e9;
                display: flex;
                gap: 8px;
            }

            .rag-chat-input {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #d1d5db;
                border-radius: 24px;
                resize: none;
                font-family: inherit;
                font-size: 14px;
                outline: none;
                max-height: 150px;
            }

            .rag-chat-input:focus {
                border-color: #4f46e5;
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
            }

            .rag-chat-send-btn {
                background: #4f46e5;
                color: white;
                border: none;
                border-radius: 24px;
                padding: 12px 20px;
                cursor: pointer;
                font-weight: 500;
                font-size: 14px;
            }

            .rag-chat-send-btn:hover {
                background: #4338ca;
            }

            .rag-chat-send-btn:disabled {
                background: #d1d5db;
                cursor: not-allowed;
            }

            .rag-chat-loading {
                padding: 16px;
                display: flex;
                align-items: center;
                gap: 12px;
                color: #6b7280;
                font-size: 14px;
            }

            .rag-loading-spinner {
                width: 20px;
                height: 20px;
                border: 2px solid #e5e7eb;
                border-top: 2px solid #4f46e5;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .rag-chat-toggle-btn {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 60px;
                height: 60px;
                background: #4f46e5;
                color: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                z-index: 10000;
                font-size: 24px;
                border: none;
            }

            .rag-chat-toggle-btn:hover {
                background: #4338ca;
                transform: scale(1.05);
            }

            @media (max-width: 480px) {
                .rag-chat-widget {
                    width: calc(100% - 40px);
                    height: 50vh;
                    bottom: 10px;
                    right: 10px;
                    left: 10px;
                }

                .rag-message {
                    max-width: 90%;
                }
            }
        `;
    }

    async initWidget() {
        // Create the chat interface
        this.chatInterface = new ChatInterface('rag-chat-container', {
            title: 'Documentation Assistant'
        });

        // Initialize text selection
        this.textSelection = new TextSelectionAPI();
        this.textSelection.addListener(this.handleTextSelection.bind(this));

        // Create toggle button if widget is closed
        this.createToggleButton();

        // Initialize session
        await this.createSession();
    }

    createToggleButton() {
        if (document.getElementById('rag-chat-toggle-btn')) return;

        const toggleBtn = document.createElement('button');
        toggleBtn.id = 'rag-chat-toggle-btn';
        toggleBtn.className = 'rag-chat-toggle-btn';
        toggleBtn.innerHTML = '💬';
        toggleBtn.title = 'Open Documentation Assistant';
        toggleBtn.onclick = () => {
            this.chatInterface.toggle();
            if (this.chatInterface.isOpen) {
                toggleBtn.style.display = 'none';
            }
        };

        document.body.appendChild(toggleBtn);
    }

    handleTextSelection(selection) {
        if (selection.text && selection.text.trim() !== '') {
            this.selectedText = selection.text.trim();

            // If the chat is not open, show a subtle indicator
            if (!this.chatInterface.isOpen) {
                // Show a small notification or highlight the toggle button
                const toggleBtn = document.getElementById('rag-chat-toggle-btn');
                if (toggleBtn) {
                    toggleBtn.style.transform = 'scale(1.1)';
                    setTimeout(() => {
                        if (toggleBtn) toggleBtn.style.transform = 'scale(1)';
                    }, 500);
                }
            }
        }
    }

    async createSession() {
        try {
            const response = await fetch(`${this.apiEndpoint}/api/v1/conversations`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`
                },
                body: JSON.stringify({
                    metadata: {
                        source_page: window.location.href,
                        user_agent: navigator.userAgent
                    }
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.sessionId = data.session_id;
                this.chatInterface.setCurrentSessionId(this.sessionId);
            }
        } catch (error) {
            console.error('Error creating session:', error);
        }
    }

    async initWidget() {
        // Create the chat interface
        this.chatInterface = new ChatInterface('rag-chat-container', {
            title: 'Documentation Assistant'
        });

        // Override the chat interface's callAPI method to use actual API
        // This is the key change to ensure real API calls are made
        this.chatInterface.callAPI = this.callAPI.bind(this);

        // Initialize text selection
        this.textSelection = new TextSelectionAPI();
        this.textSelection.addListener(this.handleTextSelection.bind(this));

        // Create toggle button if widget is closed
        this.createToggleButton();

        // Initialize session
        await this.createSession();
    }

    async callAPI(message) {
        try {
            const selectedText = this.selectedText || window.selectedTextForRAG || message;
            const response = await fetch(`${this.apiEndpoint}/api/v1/rag/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`
                },
                body: JSON.stringify({
                    query: message,
                    selected_text: selectedText,
                    session_id: this.sessionId,
                    source_url: window.location.href,
                    context_window: 3
                })
            });

            if (response.ok) {
                return await response.json();
            } else {
                console.error(`API request failed with status ${response.status}`);
                const errorText = await response.text();
                console.error('Error details:', errorText);
                throw new Error(`API request failed with status ${response.status}`);
            }
        } catch (error) {
            console.error('API call error:', error);
            return {
                response: 'Sorry, I encountered an error connecting to the service. Please try again.',
                sources: []
            };
        }
    }
}

// Register the custom element
customElements.define('rag-chat-widget', RAGChatWidget);

// Also make it available globally for non-web component usage
if (typeof window !== 'undefined') {
    window.RAGChatWidget = RAGChatWidget;
}