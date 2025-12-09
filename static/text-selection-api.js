/**
 * Text Selection API for RAG Chatbot
 * Provides functionality to capture user-selected text
 */

class TextSelectionAPI {
    constructor() {
        this.selectedText = '';
        this.selectionRect = null;
        this.listeners = [];

        this.init();
    }

    init() {
        // Listen for text selection
        document.addEventListener('mouseup', this.handleTextSelection.bind(this));
        document.addEventListener('touchend', this.handleTextSelection.bind(this));
    }

    handleTextSelection() {
        const selection = window.getSelection();
        if (selection.toString().trim() !== '') {
            this.selectedText = selection.toString().trim();
            this.selectionRect = this.getSelectionRect(selection);

            // Notify listeners
            this.notifyListeners();
        }
    }

    getSelectionRect(selection) {
        if (!selection.rangeCount) return null;

        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        return {
            x: rect.left + window.scrollX,
            y: rect.top + window.scrollY,
            width: rect.width,
            height: rect.height
        };
    }

    getSelectedText() {
        return this.selectedText;
    }

    getSelectionPosition() {
        return this.selectionRect;
    }

    addListener(callback) {
        this.listeners.push(callback);
    }

    removeListener(callback) {
        const index = this.listeners.indexOf(callback);
        if (index > -1) {
            this.listeners.splice(index, 1);
        }
    }

    notifyListeners() {
        this.listeners.forEach(callback => {
            callback({
                text: this.selectedText,
                position: this.selectionRect
            });
        });
    }

    clearSelection() {
        this.selectedText = '';
        this.selectionRect = null;
        window.getSelection().removeAllRanges();
    }
}

// Export as a global variable or module
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TextSelectionAPI;
} else {
    window.TextSelectionAPI = TextSelectionAPI;
}