// Auth functionality for the layout wrapper
// This file is loaded to ensure auth system is available globally

document.addEventListener('DOMContentLoaded', function() {
  // Initialize auth system if not already available
  if (!window.authSystem) {
    loadAuthSystem();
  }
});

function loadAuthSystem() {
  // Check if auth system is already being loaded
  if (document.querySelector('script[src*="auth-system"]')) {
    return;
  }

  // Create script element for auth system
  const script = document.createElement('script');
  script.src = '/auth/auth-system.js';  // Docusaurus serves static files from root
  script.async = true;
  document.head.appendChild(script);

  script.onload = () => {
    // Set up the auth change callback once auth system is loaded
    if (window.authSystem) {
      window.authSystem.onAuthChange = function() {
        // Dispatch auth change event for the React component to catch
        window.dispatchEvent(new CustomEvent('authChange'));
      };
    }
  };
}

// Make auth functions available globally for the React component
window.showSignupForm = function() {
  if (window.authSystem) {
    window.authSystem.showSignupForm();
  } else {
    loadAuthSystem();
    // Wait a bit and try again
    setTimeout(() => {
      if (window.authSystem) {
        window.authSystem.showSignupForm();
      }
    }, 200);
  }
};

window.showSigninForm = function() {
  if (window.authSystem) {
    window.authSystem.showSigninForm();
  } else {
    loadAuthSystem();
    // Wait a bit and try again
    setTimeout(() => {
      if (window.authSystem) {
        window.authSystem.showSigninForm();
      }
    }, 200);
  }
};

window.signOut = function() {
  if (window.authSystem) {
    window.authSystem.signOut();
  } else {
    // Manual sign out if auth system not loaded
    localStorage.removeItem('authToken');
    localStorage.removeItem('currentUser');
  }

  // Dispatch auth change event for the React component to catch
  window.dispatchEvent(new CustomEvent('authChange'));
};