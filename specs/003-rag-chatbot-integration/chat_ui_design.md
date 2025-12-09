# Chat UI and Text-Selection API Design

## Overview

This document outlines the design for the chat user interface and text selection API that will power the RAG chatbot integration. The design focuses on providing an intuitive user experience for selecting text and asking contextual questions about the documentation.

## Text Selection API

### Browser Text Selection Detection

#### Core Text Selection Handler

```javascript
class TextSelectionAPI {
  constructor() {
    this.selectionTimeout = null;
    this.lastSelection = null;
    this.callbacks = {
      onTextSelected: [],
      onSelectionCleared: []
    };
  }

  /**
   * Initialize text selection detection
   */
  init() {
    document.addEventListener('mouseup', this._handleTextSelection.bind(this));
    document.addEventListener('keyup', (e) => {
      if (e.key === 'Escape') {
        this.clearSelection();
      }
    });
  }

  /**
   * Handle text selection event
   */
  _handleTextSelection() {
    clearTimeout(this.selectionTimeout);

    // Use setTimeout to ensure selection is complete
    this.selectionTimeout = setTimeout(() => {
      const selection = window.getSelection();
      const selectedText = selection.toString().trim();

      if (selectedText.length > 0) {
        // Get the bounding rectangle for positioning
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();

        const selectionData = {
          text: selectedText,
          rect: {
            top: rect.top,
            left: rect.left,
            width: rect.width,
            height: rect.height
          },
          position: {
            x: rect.left + rect.width / 2,
            y: rect.top
          },
          range: range,
          sourceElement: selection.anchorNode?.parentElement || null
        };

        this.lastSelection = selectionData;
        this._triggerCallbacks('onTextSelected', selectionData);
      } else if (this.lastSelection) {
        this.lastSelection = null;
        this._triggerCallbacks('onSelectionCleared');
      }
    }, 100);
  }

  /**
   * Subscribe to text selection events
   */
  onTextSelected(callback) {
    this.callbacks.onTextSelected.push(callback);
  }

  /**
   * Subscribe to selection cleared events
   */
  onSelectionCleared(callback) {
    this.callbacks.onSelectionCleared.push(callback);
  }

  /**
   * Clear current selection
   */
  clearSelection() {
    window.getSelection().removeAllRanges();
    if (this.lastSelection) {
      this.lastSelection = null;
      this._triggerCallbacks('onSelectionCleared');
    }
  }

  /**
   * Get current selection
   */
  getCurrentSelection() {
    return this.lastSelection;
  }

  /**
   * Trigger callbacks for an event
   */
  _triggerCallbacks(event, data) {
    this.callbacks[event].forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error in ${event} callback:`, error);
      }
    });
  }
}

// Global instance
const textSelectionAPI = new TextSelectionAPI();
textSelectionAPI.init();
```

### Text Selection Validation

```javascript
class TextSelectionValidator {
  /**
   * Validate if the selected text is appropriate for RAG
   */
  static validateSelection(selectionData) {
    const { text, rect } = selectionData;

    // Check minimum length
    if (text.length < 10) {
      return {
        valid: false,
        reason: 'Text selection too short. Please select more content.',
        severity: 'info'
      };
    }

    // Check maximum length
    if (text.length > 1000) {
      return {
        valid: false,
        reason: 'Text selection too long. Please select a smaller portion.',
        severity: 'warning'
      };
    }

    // Check for non-text content
    if (this._containsNonTextContent(selectionData.sourceElement)) {
      return {
        valid: false,
        reason: 'Cannot process this type of content.',
        severity: 'info'
      };
    }

    return {
      valid: true,
      reason: 'Valid selection for RAG processing',
      severity: 'success'
    };
  }

  /**
   * Check if element contains non-text content
   */
  static _containsNonTextContent(element) {
    if (!element) return false;

    const nonTextSelectors = [
      'img', 'video', 'audio', 'canvas', 'svg',
      'input', 'textarea', 'select', 'button'
    ];

    return nonTextSelectors.some(selector =>
      element.matches?.(selector) || element.querySelector?.(selector)
    );
  }
}
```

## Chat UI Components

### 1. Main Chat Interface

```javascript
// chat-interface.js
class ChatInterface {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.options = {
      theme: options.theme || 'light',
      maxHeight: options.maxHeight || '500px',
      width: options.width || '400px',
      position: options.position || 'fixed',
      ...options
    };

    this.conversationHistory = [];
    this.currentSessionId = this._generateSessionId();
    this.isLoading = false;

    this._init();
  }

  _init() {
    this._createUI();
    this._attachEventListeners();
    this._loadConversationHistory();
  }

  _createUI() {
    this.container.innerHTML = `
      <div id="chat-container" class="rag-chat-container ${this.options.theme}">
        <div id="chat-header" class="rag-chat-header">
          <div class="chat-title">
            <span class="chat-icon">💬</span>
            <h3>Documentation Assistant</h3>
          </div>
          <button id="chat-close" class="chat-close-btn">✕</button>
        </div>

        <div id="chat-messages" class="rag-chat-messages">
          <div class="welcome-message">
            <p>Select text on the page and ask me about it!</p>
          </div>
        </div>

        <div id="chat-input-area" class="rag-chat-input-area">
          <div id="context-indicator" class="context-indicator hidden">
            <span class="context-text"></span>
            <button id="clear-context" class="clear-context-btn">×</button>
          </div>

          <div class="input-container">
            <textarea
              id="chat-input"
              class="chat-input"
              placeholder="Ask about the selected text..."
              rows="1"
            ></textarea>
            <button id="chat-send" class="chat-send-btn" disabled>
              <span class="send-icon">➤</span>
            </button>
          </div>

          <div id="chat-loading" class="chat-loading hidden">
            <span>Thinking...</span>
          </div>
        </div>
      </div>
    `;
  }

  _attachEventListeners() {
    // Send button
    this.container.querySelector('#chat-send').addEventListener('click', () => {
      this._handleSendMessage();
    });

    // Input events
    const input = this.container.querySelector('#chat-input');
    input.addEventListener('input', this._handleInputResize.bind(this));
    input.addEventListener('keydown', this._handleInputKeydown.bind(this));

    // Close button
    this.container.querySelector('#chat-close').addEventListener('click', () => {
      this.hide();
    });

    // Clear context
    this.container.querySelector('#clear-context').addEventListener('click', () => {
      this._clearContext();
    });

    // Text selection API integration
    textSelectionAPI.onTextSelected((selectionData) => {
      this._handleTextSelection(selectionData);
    });

    textSelectionAPI.onSelectionCleared(() => {
      this._handleSelectionCleared();
    });
  }

  _handleTextSelection(selectionData) {
    const validation = TextSelectionValidator.validateSelection(selectionData);

    if (validation.valid) {
      this._showContext(selectionData.text);
      this._focusInput();
    } else {
      // Show validation message if needed
      console.log(`Selection validation: ${validation.reason}`);
    }
  }

  _handleSelectionCleared() {
    this._clearContext();
  }

  _showContext(selectedText) {
    const contextIndicator = this.container.querySelector('#context-indicator');
    const contextText = this.container.querySelector('.context-text');

    // Truncate long text for display
    const displayText = selectedText.length > 100
      ? selectedText.substring(0, 100) + '...'
      : selectedText;

    contextText.textContent = `"${displayText}"`;
    contextIndicator.classList.remove('hidden');
  }

  _clearContext() {
    const contextIndicator = this.container.querySelector('#context-indicator');
    contextIndicator.classList.add('hidden');
  }

  _handleInputResize(event) {
    const input = event.target;
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  }

  _handleInputKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this._handleSendMessage();
    }
  }

  async _handleSendMessage() {
    const input = this.container.querySelector('#chat-input');
    const message = input.value.trim();

    if (!message || this.isLoading) return;

    // Get selected text context if available
    const selectedText = textSelectionAPI.getCurrentSelection()?.text || '';

    // Add user message to UI
    this._addMessageToUI('user', message);

    // Clear input and reset height
    input.value = '';
    input.style.height = 'auto';

    // Show loading state
    this._setLoadingState(true);

    try {
      // Call the RAG API
      const response = await this._callRAGAPI(message, selectedText);

      // Add assistant response to UI
      this._addMessageToUI('assistant', response.response);

      // Store in conversation history
      this._addMessageToHistory('user', message);
      this._addMessageToHistory('assistant', response.response);

      // Save conversation
      this._saveConversationHistory();

      // Clear context after successful query
      this._clearContext();

    } catch (error) {
      console.error('Chat API Error:', error);
      this._addMessageToUI('assistant', 'Sorry, I encountered an error. Please try again.');
    } finally {
      this._setLoadingState(false);
    }
  }

  async _callRAGAPI(userQuery, selectedText) {
    const response = await fetch('/api/v1/rag/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.RAG_API_KEY || ''}`
      },
      body: JSON.stringify({
        query: userQuery,
        selected_text: selectedText,
        session_id: this.currentSessionId,
        source_url: window.location.href
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  _addMessageToUI(role, content) {
    const messagesContainer = this.container.querySelector('#chat-messages');

    // Remove welcome message if it's the first real message
    const welcomeMessage = messagesContainer.querySelector('.welcome-message');
    if (welcomeMessage && messagesContainer.children.length <= 1) {
      welcomeMessage.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}-message`;
    messageDiv.innerHTML = `
      <div class="message-avatar">${role === 'user' ? '👤' : '🤖'}</div>
      <div class="message-content">${this._escapeHtml(content)}</div>
      <div class="message-timestamp">${new Date().toLocaleTimeString()}</div>
    `;

    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  _addMessageToHistory(role, content) {
    this.conversationHistory.push({
      id: this._generateMessageId(),
      role,
      content,
      timestamp: new Date().toISOString()
    });
  }

  _setLoadingState(isLoading) {
    const sendBtn = this.container.querySelector('#chat-send');
    const loadingIndicator = this.container.querySelector('#chat-loading');

    this.isLoading = isLoading;

    if (isLoading) {
      sendBtn.disabled = true;
      loadingIndicator.classList.remove('hidden');
    } else {
      sendBtn.disabled = false;
      loadingIndicator.classList.add('hidden');
    }
  }

  _focusInput() {
    const input = this.container.querySelector('#chat-input');
    setTimeout(() => input.focus(), 100);
  }

  _escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  _generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  _generateMessageId() {
    return 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  _loadConversationHistory() {
    try {
      const saved = localStorage.getItem(`rag_chat_history_${this.currentSessionId}`);
      if (saved) {
        this.conversationHistory = JSON.parse(saved);
        // Render conversation history
        this.conversationHistory.forEach(msg => {
          this._addMessageToUI(msg.role, msg.content);
        });
      }
    } catch (error) {
      console.error('Error loading conversation history:', error);
    }
  }

  _saveConversationHistory() {
    try {
      localStorage.setItem(
        `rag_chat_history_${this.currentSessionId}`,
        JSON.stringify(this.conversationHistory)
      );
    } catch (error) {
      console.error('Error saving conversation history:', error);
    }
  }

  show() {
    const container = this.container.querySelector('#chat-container');
    container.classList.remove('hidden');
  }

  hide() {
    const container = this.container.querySelector('#chat-container');
    container.classList.add('hidden');
  }

  destroy() {
    // Cleanup event listeners and resources
    this.container.innerHTML = '';
  }
}
```

### 2. CSS Styling

```css
/* chat-styles.css */
.rag-chat-container {
  width: var(--chat-width, 400px);
  max-height: var(--chat-max-height, 500px);
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--chat-bg, #ffffff);
  border: 1px solid var(--chat-border, #e0e0e0);
}

.rag-chat-header {
  background: var(--chat-header-bg, #f8f9fa);
  padding: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--chat-border, #e0e0e0);
}

.chat-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  color: var(--chat-text, #333333);
}

.chat-icon {
  font-size: 18px;
}

.chat-close-btn {
  background: none;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: var(--chat-text-light, #666666);
  padding: 4px;
  border-radius: 4px;
  transition: background-color 0.2s;
}

.chat-close-btn:hover {
  background: var(--chat-hover-bg, #e0e0e0);
}

.rag-chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: var(--chat-messages-bg, #fafafa);
}

.welcome-message {
  text-align: center;
  padding: 20px;
  color: var(--chat-text-light, #666666);
  font-style: italic;
}

.chat-message {
  display: flex;
  gap: 12px;
  max-width: 90%;
  animation: fadeIn 0.3s ease-in;
}

.user-message {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.user-message .message-content {
  background: var(--chat-user-bg, #e3f2fd);
  border-radius: 18px 4px 18px 18px;
}

.assistant-message .message-content {
  background: var(--chat-assistant-bg, #ffffff);
  border-radius: 4px 18px 18px 18px;
}

.message-avatar {
  align-self: flex-end;
  font-size: 14px;
  min-width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: var(--chat-avatar-bg, #e0e0e0);
}

.message-content {
  padding: 12px 16px;
  border-radius: 18px;
  line-height: 1.5;
  word-wrap: break-word;
  white-space: pre-wrap;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.message-timestamp {
  font-size: 11px;
  color: var(--chat-text-light, #999999);
  margin-top: 4px;
  align-self: flex-end;
}

.rag-chat-input-area {
  padding: 16px;
  border-top: 1px solid var(--chat-border, #e0e0e0);
  background: var(--chat-input-bg, #ffffff);
}

.context-indicator {
  background: var(--chat-context-bg, #e8f5e9);
  border: 1px solid var(--chat-context-border, #c8e6c9);
  border-radius: 16px;
  padding: 8px 12px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.context-text {
  font-size: 14px;
  color: var(--chat-context-text, #2e7d32);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-right: 8px;
}

.clear-context-btn {
  background: none;
  border: none;
  font-size: 16px;
  cursor: pointer;
  color: var(--chat-context-text, #2e7d32);
  padding: 2px 6px;
  border-radius: 50%;
}

.clear-context-btn:hover {
  background: var(--chat-hover-bg, #c8e6c9);
}

.input-container {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}

.chat-input {
  flex: 1;
  border: 1px solid var(--chat-input-border, #ddd);
  border-radius: 20px;
  padding: 12px 16px;
  resize: none;
  overflow: hidden;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.4;
  max-height: 120px;
  outline: none;
  transition: border-color 0.2s;
}

.chat-input:focus {
  border-color: var(--chat-primary, #1976d2);
}

.chat-send-btn {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: none;
  background: var(--chat-primary, #1976d2);
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.2s;
}

.chat-send-btn:disabled {
  background: var(--chat-disabled, #cccccc);
  cursor: not-allowed;
}

.chat-send-btn:not(:disabled):hover {
  background: var(--chat-primary-dark, #1565c0);
}

.chat-loading {
  text-align: center;
  padding: 8px;
  color: var(--chat-text-light, #666666);
  font-style: italic;
}

.hidden {
  display: none !important;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Responsive design */
@media (max-width: 768px) {
  .rag-chat-container {
    width: calc(100vw - 32px);
    max-height: 50vh;
    position: fixed;
    bottom: 0;
    left: 16px;
    right: 16px;
    border-radius: 16px 16px 0 0;
  }
}
```

### 3. API Integration Layer

```javascript
// api-integration.js
class RAGChatAPI {
  constructor(config) {
    this.config = {
      baseUrl: config.baseUrl || '/api/v1',
      apiKey: config.apiKey,
      timeout: config.timeout || 30000,
      ...config
    };
  }

  /**
   * Send a query to the RAG API
   */
  async query(userQuery, selectedText, options = {}) {
    const requestBody = {
      query: userQuery,
      selected_text: selectedText,
      session_id: options.sessionId,
      source_url: options.sourceUrl || window.location.href,
      context_window: options.contextWindow || 3,
      ...options.additionalParams
    };

    const response = await fetch(`${this.config.baseUrl}/rag/query`, {
      method: 'POST',
      headers: this._getHeaders(),
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(this.config.timeout)
    });

    if (!response.ok) {
      throw new Error(`RAG API Error: ${response.status} - ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Process document embeddings
   */
  async processDocument(documentData) {
    const response = await fetch(`${this.config.baseUrl}/embeddings/process`, {
      method: 'POST',
      headers: this._getHeaders(),
      body: JSON.stringify(documentData)
    });

    if (!response.ok) {
      throw new Error(`Embedding API Error: ${response.status} - ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Get conversation history
   */
  async getConversation(sessionId) {
    const response = await fetch(`${this.config.baseUrl}/conversations/${sessionId}`, {
      method: 'GET',
      headers: this._getHeaders()
    });

    if (!response.ok) {
      if (response.status === 404) {
        return { messages: [] }; // New conversation
      }
      throw new Error(`Conversation API Error: ${response.status} - ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Submit feedback for a response
   */
  async submitFeedback(messageId, feedback) {
    const response = await fetch(`${this.config.baseUrl}/conversations/feedback`, {
      method: 'POST',
      headers: this._getHeaders(),
      body: JSON.stringify({ message_id: messageId, feedback })
    });

    if (!response.ok) {
      throw new Error(`Feedback API Error: ${response.status} - ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Get health status of the RAG service
   */
  async healthCheck() {
    const response = await fetch(`${this.config.baseUrl}/rag/health`, {
      method: 'GET'
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status} - ${response.statusText}`);
    }

    return await response.json();
  }

  _getHeaders() {
    const headers = {
      'Content-Type': 'application/json'
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    return headers;
  }
}
```

### 4. Utility Functions

```javascript
// utils.js
class ChatUtils {
  /**
   * Format message content for display
   */
  static formatMessageContent(content) {
    // Convert markdown-like formatting to HTML
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold
      .replace(/\*(.*?)\*/g, '<em>$1</em>')              // Italic
      .replace(/`(.*?)`/g, '<code>$1</code>')            // Code
      .replace(/\n/g, '<br>');                           // Line breaks
  }

  /**
   * Estimate token count for a message
   */
  static estimateTokenCount(text) {
    // Rough estimation: 1 token ≈ 4 characters
    return Math.ceil(text.length / 4);
  }

  /**
   * Validate user input
   */
  static validateInput(input) {
    if (!input || input.trim().length === 0) {
      return { valid: false, error: 'Message cannot be empty' };
    }

    if (input.length > 2000) {
      return { valid: false, error: 'Message too long (max 2000 characters)' };
    }

    return { valid: true };
  }

  /**
   * Generate a readable timestamp
   */
  static formatTimestamp(date) {
    const now = new Date();
    const diff = now - new Date(date);
    const minutes = Math.floor(diff / 60000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`;
    return new Date(date).toLocaleDateString();
  }

  /**
   * Sanitize user input to prevent XSS
   */
  static sanitizeInput(input) {
    const div = document.createElement('div');
    div.textContent = input;
    return div.innerHTML;
  }
}
```

## Implementation Plan

### Phase 1: Basic Text Selection
- Implement text selection detection
- Create basic activation button
- Integrate with existing API

### Phase 2: Chat Interface
- Build responsive chat UI
- Implement conversation history
- Add loading states and error handling

### Phase 3: Advanced Features
- Add context indicators
- Implement local storage for conversations
- Add analytics and feedback mechanisms

### Phase 4: Optimization
- Performance optimization
- Accessibility improvements
- Mobile responsiveness