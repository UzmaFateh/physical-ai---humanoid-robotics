import React, { useState, useEffect } from 'react';

const AuthButtons = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userEmail, setUserEmail] = useState('');

  useEffect(() => {
    // Check authentication status on component mount
    checkAuthStatus();

    // Listen for auth events
    const handleAuthChange = () => {
      checkAuthStatus();
    };

    window.addEventListener('authChange', handleAuthChange);

    return () => {
      window.removeEventListener('authChange', handleAuthChange);
    };
  }, []);

  const checkAuthStatus = () => {
    const token = localStorage.getItem('authToken');
    if (token) {
      setIsAuthenticated(true);
      // Try to get user email from stored user data
      const storedUser = localStorage.getItem('currentUser');
      if (storedUser) {
        try {
          const user = JSON.parse(storedUser);
          setUserEmail(user.email);
        } catch (e) {
          console.error('Error parsing stored user:', e);
        }
      }
    } else {
      setIsAuthenticated(false);
      setUserEmail('');
    }
  };

  const showSignupForm = () => {
    if (window.authSystem) {
      window.authSystem.showSignupForm();
    } else {
      // Load the auth system if not available
      loadAuthSystem();
    }
  };

  const showSigninForm = () => {
    if (window.authSystem) {
      window.authSystem.showSigninForm();
    } else {
      // Load the auth system if not available
      loadAuthSystem();
    }
  };

  const signOut = () => {
    if (window.authSystem) {
      window.authSystem.signOut();
      setIsAuthenticated(false);
      setUserEmail('');

      // Dispatch auth change event
      window.dispatchEvent(new CustomEvent('authChange'));
    }
  };

  const loadAuthSystem = () => {
    // Create script element for auth system
    const script = document.createElement('script');
    script.src = '/auth/auth-system.js';  // Docusaurus serves static files from root
    script.async = true;
    document.head.appendChild(script);

    script.onload = () => {
      // Set up the auth change callback
      window.authSystem.onAuthChange = function() {
        checkAuthStatus();
        window.dispatchEvent(new CustomEvent('authChange'));
      };
    };
  };

  if (isAuthenticated) {
    return (
      <div className="auth-buttons">
        <span className="user-greeting">Hello, {userEmail}</span>
        <button
          className="button button--secondary button--sm"
          onClick={signOut}
          style={{ marginLeft: '10px' }}
        >
          Sign Out
        </button>
      </div>
    );
  } else {
    return (
      <div className="auth-buttons">
        <button
          className="button button--secondary button--sm"
          onClick={showSigninForm}
          style={{ marginRight: '5px' }}
        >
          Sign In
        </button>
        <button
          className="button button--primary button--sm"
          onClick={showSignupForm}
        >
          Sign Up
        </button>
      </div>
    );
  }
};

export default AuthButtons;