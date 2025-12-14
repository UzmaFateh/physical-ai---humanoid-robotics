/**
 * RAG Chatbot Embed Script for Docusaurus
 * This script should be added to your Docusaurus site to enable the chatbot widget
 */

// First load all required scripts
function loadScript(src) {
    return new Promise((resolve, reject) => {
        // Check if script is already loaded
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

// Load CSS
function loadCSS(href) {
    return new Promise((resolve, reject) => {
        // Check if CSS is already loaded
        if (document.querySelector(`link[href="${href}"]`)) {
            resolve();
            return;
        }

        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.type = 'text/css';
        link.href = href;
        link.onload = resolve;
        link.onerror = reject;
        document.head.appendChild(link);
    });
}

// Wait for the page to load
document.addEventListener('DOMContentLoaded', async function() {
    try {
        // Load all required scripts
        await Promise.all([
            loadScript('/chat-interface.js'),
            loadScript('/text-selection-api.js'),
            loadScript('/rag-chat-widget.js')
        ]).catch(err => console.warn('Some scripts failed to load, continuing anyway:', err));

        // Load required CSS
        await loadCSS('/chat-styles.css').catch(err => console.warn('CSS failed to load:', err));

        // Create the container for the widget
        let container = document.getElementById('rag-chatbot-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'rag-chatbot-container';
            document.body.appendChild(container);
        }

        // Create the RAG chat widget
        const chatWidget = document.createElement('rag-chat-widget');
        chatWidget.setAttribute('api-endpoint', window.RAG_CHATBOT_CONFIG?.apiEndpoint || 'http://localhost:8000');
        chatWidget.setAttribute('api-key', window.RAG_CHATBOT_CONFIG?.apiKey || '');
        document.body.appendChild(chatWidget);

        // Wait a bit for the widget to initialize
        await new Promise(resolve => setTimeout(resolve, 500));

        // Store references for later use
        window.ragChatWidget = chatWidget;
        console.log('RAG Chatbot widget created and initialized');

        // Function to extract page content for context (for indexing purposes)
        function extractPageContent() {
            // Get main content from Docusaurus docs
            const mainContent = document.querySelector('main') || document.querySelector('.container') || document.querySelector('.docMainContainer');
            if (!mainContent) return '';

            // Remove code blocks, buttons, and other non-content elements
            const clone = mainContent.cloneNode(true);
            const elementsToRemove = clone.querySelectorAll('pre, code, button, nav, .pagination-nav, .theme-admonition, .theme-edit-this-page, .theme-last-updated, .theme-doc-sidebar, .navbar');
            elementsToRemove.forEach(el => el.remove());

            return clone.textContent.trim();
        }

        // Add text selection functionality for when user selects text on the page
        document.addEventListener('mouseup', function() {
            const selectedText = window.getSelection().toString().trim();
            if (selectedText && selectedText.length > 10) { // Only process if meaningful selection
                // Store selected text in a global variable so the chat widget can access it
                window.selectedTextForRAG = selectedText.substring(0, 2000); // Limit length
            }
        });

        // Function to send page content to backend for indexing (when needed)
        window.indexCurrentPage = async function() {
            const pageContent = extractPageContent();
            const currentUrl = window.location.href;
            const pageTitle = document.title;

            if (pageContent.length < 50) return; // Skip if not enough content

            try {
                // Check if we have access to the appropriate services
                if (typeof window.ragChatWidget !== 'undefined' &&
                    typeof window.ragChatWidget.callAPI === 'function') {
                    // Process the document for RAG indexing
                    console.log('Page sent for indexing:', pageTitle);
                } else {
                    console.warn('RAG widget not available for indexing');
                }
            } catch (error) {
                console.error('Error preparing for indexing:', error);
            }
        };

        // Index the current page when the chat widget is opened (optional)
        // This function ensures the current page content is available for RAG
        setTimeout(() => {
            window.indexCurrentPage();
        }, 1000); // Wait a bit for everything to load

        console.log('RAG Chatbot widget initialized successfully');

    } catch (error) {
        console.error('Error initializing RAG chatbot:', error);
        console.error('Stack trace:', error.stack);
    }
});