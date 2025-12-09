# Docusaurus Embed Widget Implementation Plan

## Overview

The Docusaurus embed widget will provide a seamless chatbot experience integrated directly into the documentation site. The widget will appear when users select text and ask questions about it, providing contextual answers based on the documentation content.

## Widget Architecture

### Frontend Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Docusaurus Site                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Documentation Content                │   │
│  │                                                     │   │
│  │  Selected Text → [?] Widget appears here            │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Chat Widget Interface                │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │              Conversation Area              │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │              Input Area                     │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Components

### 1. Widget Entry Point (JavaScript)

```javascript
// rag-chat-widget.js
class RAGChatWidget extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.isVisible = false;
    this.selectedText = '';
    this.conversation = [];
    this.isLoading = false;
    this.apiEndpoint = process.env.RAG_API_ENDPOINT || '/api/v1';
    this.sessionId = this._generateSessionId();
  }

  connectedCallback() {
    this.render();
    this._setupEventListeners();
  }

  render() {
    this.shadowRoot.innerHTML = `
      <style>
        ${this._getStyles()}
      </style>
      <div id="widget-container" class="widget-hidden">
        <div id="widget-header">
          <h3>Documentation Assistant</h3>
          <button id="close-btn">×</button>
        </div>
        <div id="conversation-area">
          <div id="messages-container"></div>
        </div>
        <div id="input-area">
          <textarea id="user-input" placeholder="Ask about the selected text..."></textarea>
          <button id="send-btn">Send</button>
        </div>
        <div id="loading-indicator" class="hidden">Thinking...</div>
      </div>
      <div id="activation-btn" class="hidden">
        <span>?</span>
      </div>
    `;
    this._attachEventListeners();
  }

  _getStyles() {
    return `
      :host {
        all: initial;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }

      #widget-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 400px;
        height: 500px;
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        display: flex;
        flex-direction: column;
        z-index: 10000;
      }

      .widget-hidden {
        display: none;
      }

      #widget-header {
        padding: 16px;
        background: #f8f9fa;
        border-bottom: 1px solid #e0e0e0;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      #close-btn {
        background: none;
        border: none;
        font-size: 24px;
        cursor: pointer;
        color: #666;
      }

      #conversation-area {
        flex: 1;
        overflow-y: auto;
        padding: 16px;
      }

      .message {
        margin-bottom: 12px;
        padding: 8px 12px;
        border-radius: 8px;
        max-width: 80%;
      }

      .user-message {
        background: #e3f2fd;
        margin-left: auto;
      }

      .assistant-message {
        background: #f5f5f5;
        margin-right: auto;
      }

      #input-area {
        padding: 16px;
        border-top: 1px solid #e0e0e0;
        display: flex;
        gap: 8px;
      }

      #user-input {
        flex: 1;
        padding: 8px 12px;
        border: 1px solid #ddd;
        border-radius: 4px;
        resize: none;
        height: 60px;
      }

      #send-btn {
        padding: 8px 16px;
        background: #1976d2;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
      }

      #activation-btn {
        position: absolute;
        background: #1976d2;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        z-index: 10001;
        font-size: 18px;
        font-weight: bold;
      }

      .hidden {
        display: none !important;
      }

      #loading-indicator {
        padding: 8px 16px;
        color: #666;
        font-style: italic;
      }
    `;
  }

  _setupEventListeners() {
    // Text selection detection
    document.addEventListener('mouseup', this._handleTextSelection.bind(this));

    // Widget resize/scroll handling
    window.addEventListener('resize', this._handleResize.bind(this));
    window.addEventListener('scroll', this._handleScroll.bind(this));
  }

  _attachEventListeners() {
    this.shadowRoot.getElementById('close-btn').addEventListener('click', this._closeWidget.bind(this));
    this.shadowRoot.getElementById('send-btn').addEventListener('click', this._handleSendMessage.bind(this));
    this.shadowRoot.getElementById('user-input').addEventListener('keypress', this._handleKeyPress.bind(this));
  }

  _handleTextSelection() {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    if (selectedText.length > 0) {
      this.selectedText = selectedText;
      this._showActivationButton(selection);
    } else {
      this._hideActivationButton();
    }
  }

  _showActivationButton(selection) {
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();

    const activationBtn = this.shadowRoot.getElementById('activation-btn');
    activationBtn.classList.remove('hidden');

    // Position the button near the selection
    activationBtn.style.top = `${rect.top + window.scrollY - 50}px`;
    activationBtn.style.left = `${rect.left + window.scrollX + rect.width/2 - 20}px`;

    // Add click handler to activation button
    activationBtn.onclick = () => {
      this._showWidget();
      this._hideActivationButton();

      // Pre-fill the input with context about the selected text
      const userInput = this.shadowRoot.getElementById('user-input');
      userInput.focus();
      userInput.placeholder = `Ask about: "${this.selectedText.substring(0, 50)}${this.selectedText.length > 50 ? '...' : ''}"`;
    };
  }

  _hideActivationButton() {
    const activationBtn = this.shadowRoot.getElementById('activation-btn');
    activationBtn.classList.add('hidden');
  }

  _showWidget() {
    const widgetContainer = this.shadowRoot.getElementById('widget-container');
    widgetContainer.classList.remove('widget-hidden');
    this.isVisible = true;

    // Add overlay to improve focus
    this._createOverlay();
  }

  _closeWidget() {
    const widgetContainer = this.shadowRoot.getElementById('widget-container');
    widgetContainer.classList.add('widget-hidden');
    this.isVisible = false;

    // Remove overlay
    this._removeOverlay();
  }

  _createOverlay() {
    if (!this.shadowRoot.querySelector('#widget-overlay')) {
      const overlay = document.createElement('div');
      overlay.id = 'widget-overlay';
      overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.1);
        z-index: 9999;
      `;
      this.shadowRoot.appendChild(overlay);

      // Clicking overlay closes widget
      overlay.addEventListener('click', this._closeWidget.bind(this));
    }
  }

  _removeOverlay() {
    const overlay = this.shadowRoot.getElementById('widget-overlay');
    if (overlay) {
      overlay.remove();
    }
  }

  async _handleSendMessage() {
    const userInput = this.shadowRoot.getElementById('user-input');
    const message = userInput.value.trim();

    if (!message) return;

    // Add user message to conversation
    this._addMessageToUI('user', message);

    // Clear input
    userInput.value = '';

    // Show loading state
    this._setLoadingState(true);

    try {
      // Call the RAG API
      const response = await this._callRAGAPI(message);

      // Add assistant response to conversation
      this._addMessageToUI('assistant', response.response);

      // Store in conversation history
      this.conversation.push({ role: 'user', content: message });
      this.conversation.push({ role: 'assistant', content: response.response });
    } catch (error) {
      this._addMessageToUI('assistant', 'Sorry, I encountered an error. Please try again.');
      console.error('RAG API Error:', error);
    } finally {
      this._setLoadingState(false);
    }
  }

  _handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this._handleSendMessage();
    }
  }

  async _callRAGAPI(userQuery) {
    const response = await fetch(`${this.apiEndpoint}/rag/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.RAG_API_KEY}` // Or however auth is handled
      },
      body: JSON.stringify({
        query: userQuery,
        selected_text: this.selectedText,
        session_id: this.sessionId,
        source_url: window.location.href
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    return await response.json();
  }

  _addMessageToUI(role, content) {
    const messagesContainer = this.shadowRoot.getElementById('messages-container');

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${role}-message`);
    messageDiv.textContent = content;

    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  _setLoadingState(isLoading) {
    const loadingIndicator = this.shadowRoot.getElementById('loading-indicator');
    const sendBtn = this.shadowRoot.getElementById('send-btn');

    if (isLoading) {
      loadingIndicator.classList.remove('hidden');
      sendBtn.disabled = true;
      this.isLoading = true;
    } else {
      loadingIndicator.classList.add('hidden');
      sendBtn.disabled = false;
      this.isLoading = false;
    }
  }

  _generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  _handleResize() {
    // Re-position activation button if needed
    if (!this.isVisible) {
      this._hideActivationButton();
    }
  }

  _handleScroll() {
    // Hide activation button when scrolling
    if (!this.isVisible) {
      this._hideActivationButton();
    }
  }
}

// Register the custom element
customElements.define('rag-chat-widget', RAGChatWidget);

// Initialize the widget when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  if (!document.querySelector('rag-chat-widget')) {
    const widget = document.createElement('rag-chat-widget');
    document.body.appendChild(widget);
  }
});
```

### 2. Docusaurus Integration

#### Plugin Structure

```
docusaurus-rag-chat-plugin/
├── src/
│   ├── index.js              # Plugin entry point
│   ├── theme/
│   │   └── RagChatWidget.js  # Widget component
│   └── utils/
│       └── config.js         # Configuration utilities
├── package.json
└── README.md
```

#### Plugin Implementation

```javascript
// docusaurus-rag-chat-plugin/src/index.js
const path = require('path');

module.exports = function (context, options) {
  const { siteConfig } = context;
  const config = {
    enabled: options.enabled !== false, // enabled by default
    apiEndpoint: options.apiEndpoint || '/api/v1',
    position: options.position || 'bottom-right', // 'bottom-right', 'bottom-left'
    zIndex: options.zIndex || 10000,
    ...options
  };

  return {
    name: 'docusaurus-rag-chat-plugin',

    getClientModules() {
      if (!config.enabled) return [];
      return [path.resolve(__dirname, './theme/RagChatWidget.js')];
    },

    configureWebpack(config, isServer, utils) {
      if (!config.enabled) return {};

      return {
        resolve: {
          alias: {
            '@rag-chat/config': path.resolve(__dirname, './utils/config.js')
          }
        }
      };
    },

    async contentLoaded({ actions }) {
      if (!config.enabled) return;

      const { setGlobalData } = actions;
      setGlobalData(config);
    }
  };
};
```

#### Docusaurus Configuration

```javascript
// docusaurus.config.js
module.exports = {
  // ... other config
  plugins: [
    [
      'docusaurus-rag-chat-plugin',
      {
        enabled: true,
        apiEndpoint: process.env.RAG_API_ENDPOINT || 'http://localhost:8000/api/v1',
        position: 'bottom-right',
        zIndex: 10000
      }
    ]
  ],
  themeConfig: {
    // ... other theme config
  }
};
```

### 3. Widget Configuration Options

#### Configuration Schema

```javascript
// Configuration options for the widget
const WidgetConfigSchema = {
  apiEndpoint: {
    type: 'string',
    description: 'Base URL for the RAG API',
    default: '/api/v1'
  },
  position: {
    type: 'string',
    enum: ['bottom-right', 'bottom-left'],
    default: 'bottom-right'
  },
  zIndex: {
    type: 'number',
    default: 10000
  },
  welcomeMessage: {
    type: 'string',
    default: 'Ask me anything about this documentation!'
  },
  showActivationButton: {
    type: 'boolean',
    default: true
  },
  enableLogging: {
    type: 'boolean',
    default: false
  }
};
```

### 4. Styling and Theming

#### CSS Variables for Theming

```css
/* Custom properties for easy theming */
:root {
  --rag-widget-primary-color: #1976d2;
  --rag-widget-secondary-color: #f5f5f5;
  --rag-widget-text-color: #333;
  --rag-widget-border-color: #e0e0e0;
  --rag-widget-shadow: 0 4px 12px rgba(0,0,0,0.15);
  --rag-widget-radius: 8px;
}
```

### 5. Widget Installation

#### Method 1: NPM Package

```bash
npm install docusaurus-rag-chat-plugin
```

#### Method 2: Direct Script Injection

```html
<!-- Add to your Docusaurus site -->
<script src="/path/to/rag-chat-widget.js"></script>
```

### 6. Advanced Features

#### Conversation History Persistence

```javascript
// Implement local storage for conversation history
class ConversationManager {
  constructor(sessionId) {
    this.sessionId = sessionId;
    this.storageKey = `rag_chat_conversation_${sessionId}`;
  }

  save(conversation) {
    localStorage.setItem(this.storageKey, JSON.stringify(conversation));
  }

  load() {
    const stored = localStorage.getItem(this.storageKey);
    return stored ? JSON.parse(stored) : [];
  }

  clear() {
    localStorage.removeItem(this.storageKey);
  }
}
```

#### Analytics and Feedback

```javascript
// Add analytics tracking
class AnalyticsTracker {
  static trackEvent(eventType, properties = {}) {
    // Track widget usage
    if (typeof window.gtag !== 'undefined') {
      gtag('event', `rag_${eventType}`, properties);
    }

    // Or send to custom analytics endpoint
    fetch('/api/analytics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event: eventType,
        properties: { ...properties, timestamp: Date.now() }
      })
    }).catch(console.error);
  }
}
```

### 7. Security Considerations

#### Content Security Policy

```javascript
// Ensure widget follows CSP
const CSP_HEADERS = {
  'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';"
};
```

#### Input Sanitization

```javascript
// Sanitize user inputs before sending to API
function sanitizeInput(input) {
  // Remove potentially harmful content
  return input
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .trim();
}
```

### 8. Performance Optimization

#### Lazy Loading

```javascript
// Implement lazy loading for the widget
const loadWidget = async () => {
  if (!window.customElements.get('rag-chat-widget')) {
    // Dynamically import the widget code
    await import('./rag-chat-widget.js');
  }

  // Initialize widget
  const widget = document.createElement('rag-chat-widget');
  document.body.appendChild(widget);
};
```

#### Caching Strategies

```javascript
// Implement caching for API responses
class ResponseCache {
  constructor(maxSize = 100) {
    this.cache = new Map();
    this.maxSize = maxSize;
  }

  set(key, value) {
    if (this.cache.size >= this.maxSize) {
      // Remove oldest entry
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  get(key) {
    return this.cache.get(key);
  }

  has(key) {
    return this.cache.has(key);
  }
}
```

This implementation provides a comprehensive embeddable chat widget that integrates seamlessly with Docusaurus documentation sites, allowing users to select text and ask contextual questions about the content.