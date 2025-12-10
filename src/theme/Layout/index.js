import React, { useState, useEffect } from 'react';
import OriginalLayout from '@theme-original/Layout';
import { useLocation } from '@docusaurus/router';

export default function Layout(props) {
  const { pathname } = useLocation();
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

    // Also check auth status when localStorage changes (another tab might have changed auth state)
    const handleStorageChange = (e) => {
      if (e.key === 'authToken') {
        checkAuthStatus();
      }
    };

    window.addEventListener('storage', handleStorageChange);

    return () => {
      window.removeEventListener('authChange', handleAuthChange);
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [pathname]);

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
      // Set up the auth change callback
      window.authSystem.onAuthChange = function() {
        checkAuthStatus();
        window.dispatchEvent(new CustomEvent('authChange'));
      };
    };
  };

  return (
    <>
      <OriginalLayout {...props}>
        {props.children}
        {/* Auth buttons and theme toggle in top right of the page */}
        <div style={{
          position: 'fixed',
          top: '20px',
          right: '20px',
          zIndex: '9999',
          display: 'flex',
          gap: '8px',
          alignItems: 'center'
        }}>
          {!isAuthenticated ? (
            <>
              <button
                onClick={showSigninForm}
                style={{
                  padding: '6px 12px',
                  backgroundColor: '#fff',
                  color: '#2e8555',
                  border: '1px solid #2e8555',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  height: '32px'
                }}
                title="Sign In"
              >
                Sign In
              </button>
              <button
                onClick={showSignupForm}
                style={{
                  padding: '6px 12px',
                  backgroundColor: '#2e8555',
                  color: '#fff',
                  border: '1px solid #2e8555',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  height: '32px'
                }}
                title="Sign Up"
              >
                Sign Up
              </button>
            </>
          ) : (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '4px 8px',
              backgroundColor: '#f8f9fa',
              border: '1px solid #dee2e6',
              borderRadius: '4px',
              fontSize: '14px',
              height: '32px'
            }}>
              <span style={{ fontSize: '13px' }}>Hi, {userEmail.split('@')[0]}</span>
              <button
                onClick={signOut}
                style={{
                  padding: '2px 8px',
                  backgroundColor: '#6c757d',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px',
                  height: '24px'
                }}
                title="Sign Out"
              >
                Sign Out
              </button>
            </div>
          )}
        </div>
      </OriginalLayout>
    </>
  );
}