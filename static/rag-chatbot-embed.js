/**
 * RAG Chatbot Embed Script for Docusaurus
 * This script should be added to your Docusaurus site to enable the chatbot widget
 */

// First load the API integration and chat interface scripts if not already loaded
function loadScript(src) {
    return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) {
            resolve();
            return;
        }

        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// Wait for the page to load
document.addEventListener('DOMContentLoaded', async function() {
    try {
        // Load required scripts
        await Promise.allSettled([
            loadScript('/api-integration.js').catch(() => {}), // Try to load from root
            loadScript('/chat-interface.js').catch(() => {})   // Try to load from root
        ]);

        // Also try to load from static directory if needed
        if (typeof APIIntegration === 'undefined') {
            const apiScript = document.createElement('script');
            apiScript.src = '/api-integration.js';
            document.head.appendChild(apiScript);
        }

        if (typeof ChatInterface === 'undefined') {
            const chatScript = document.createElement('script');
            chatScript.src = '/chat-interface.js';
            document.head.appendChild(chatScript);
        }

        // Wait a bit for scripts to load if they were just added
        await new Promise(resolve => setTimeout(resolve, 100));

        // Load the CSS
        const cssLink = document.createElement('link');
        cssLink.rel = 'stylesheet';
        cssLink.type = 'text/css';
        cssLink.href = '/css/rag-chatbot.css'; // Adjust path as needed
        document.head.appendChild(cssLink);

        // Create the container for the widget
        const container = document.createElement('div');
        container.id = 'rag-chatbot-container';
        document.body.appendChild(container);

        // Initialize the chat interface
        const chatInterface = new ChatInterface('rag-chatbot-container', {
            title: 'Textbook Assistant',
            placeholder: 'Ask a question about this textbook content...',
            onClosed: function() {
                // Show the toggle button when chat is closed
                const toggleBtn = document.getElementById('rag-chat-toggle-btn');
                if (toggleBtn) {
                    toggleBtn.style.display = 'flex';
                }
            },
            onOpened: function() {
                // Hide the toggle button when chat is opened
                const toggleBtn = document.getElementById('rag-chat-toggle-btn');
                if (toggleBtn) {
                    toggleBtn.style.display = 'none';
                }
            }
        });

        // Create API integration instance
        const apiIntegration = new APIIntegration(
            window.RAG_CHATBOT_CONFIG?.apiEndpoint || 'http://localhost:8000',
            window.RAG_CHATBOT_CONFIG?.apiKey || ''
        );

        // Override the callAPI method to use actual backend
        chatInterface.callAPI = async function(message) {
            // Get selected text from the page
            const selectedText = window.getSelection().toString().trim();
            const sourceUrl = window.location.href;

            try {
                const response = await apiIntegration.queryRAG(
                    message,
                    selectedText,
                    chatInterface.getCurrentSessionId(),
                    sourceUrl
                );
                return response;
            } catch (error) {
                console.error('API call error:', error);
                return {
                    response: 'Sorry, I encountered an error connecting to the backend. Please try again.',
                    sources: []
                };
            }
        };

        // Store references for later use
        window.ragChatInterface = chatInterface;
        window.ragApiIntegration = apiIntegration;

        // Function to extract page content for context
        function extractPageContent() {
            // Get main content from Docusaurus docs
            const mainContent = document.querySelector('main') || document.querySelector('.container');
            if (!mainContent) return '';

            // Remove code blocks, buttons, and other non-content elements
            const clone = mainContent.cloneNode(true);
            const elementsToRemove = clone.querySelectorAll('pre, code, button, nav, .pagination-nav, .theme-admonition, .theme-edit-this-page');
            elementsToRemove.forEach(el => el.remove());

            return clone.textContent.trim();
        }

        // Add text selection functionality
        document.addEventListener('mouseup', function() {
            const selectedText = window.getSelection().toString().trim();
            if (selectedText && selectedText.length > 10) { // Only show if meaningful selection
                // Optionally show a hint to ask about the selected text
                console.log('Text selected:', selectedText.substring(0, 100) + '...');
            }
        });

        // Function to send page content to backend for indexing (when needed)
        window.indexCurrentPage = async function() {
            const pageContent = extractPageContent();
            const currentUrl = window.location.href;
            const pageTitle = document.title;

            if (pageContent.length < 50) return; // Skip if not enough content

            try {
                const documentChunk = {
                    content: pageContent,
                    source_url: currentUrl,
                    page_title: pageTitle,
                    section: 'main_content',
                    metadata: {
                        timestamp: new Date().toISOString(),
                        url: currentUrl,
                        title: pageTitle
                    }
                };

                // Process the document for RAG
                const result = await apiIntegration.processDocuments([documentChunk]);
                console.log('Page indexed successfully:', result);
            } catch (error) {
                console.error('Error indexing page:', error);
            }
        };

        // Index the current page when the chat widget is opened (optional)
        const originalToggle = chatInterface.toggle;
        chatInterface.toggle = function() {
            originalToggle.call(this);
            if (!this.isOpen) {
                // When opening the chat, ensure the current page is indexed
                setTimeout(() => {
                    window.indexCurrentPage();
                }, 500);
            }
        };

        // Add widget toggle button if needed
        if (!document.getElementById('rag-chat-toggle-btn')) {
            const toggleBtn = document.createElement('button');
            toggleBtn.id = 'rag-chat-toggle-btn';
            toggleBtn.className = 'rag-chat-toggle-btn';
            toggleBtn.innerHTML = '💬';
            toggleBtn.title = 'Open Textbook Assistant';
            toggleBtn.style.cssText = `
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
                font-family: inherit;
            `;
            toggleBtn.onclick = () => {
                chatInterface.toggle();
                if (chatInterface.isOpen) {
                    toggleBtn.style.display = 'none';
                } else {
                    toggleBtn.style.display = 'flex';
                }
            };

            // Initially show the toggle button since the chat starts closed
            document.body.appendChild(toggleBtn);
            // Make sure it's visible when initially created (chat starts closed)
            setTimeout(() => {
                toggleBtn.style.display = 'flex';
            }, 100);
        }
    } catch (error) {
        console.error('Error initializing RAG chatbot:', error);
    }
});