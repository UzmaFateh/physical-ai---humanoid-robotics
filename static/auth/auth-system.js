// static/auth/auth-system.js
// Custom authentication system with background questions

class AuthSystem {
  constructor() {
    this.currentUser = null;
    // Using direct API paths instead of the variable since we've updated the methods
  }

  // Show signup form with background questions
  async showSignupForm() {
    const formHtml = `
      <div id="auth-modal" class="auth-modal">
        <div class="auth-container">
          <button class="auth-close-btn" id="close-auth-modal">&times;</button>
          <h2>Create Account</h2>
          <form id="signup-form">
            <div class="form-group">
              <label for="email">Email:</label>
              <input type="email" id="email" name="email" required>
            </div>

            <div class="form-group">
              <label for="password">Password:</label>
              <input type="password" id="password" name="password" required>
            </div>

            <div class="form-group">
              <label for="software-background">Software Background:</label>
              <select id="software-background" name="softwareBackground">
                <option value="">Select your software background</option>
                <option value="beginner">Beginner (Learning)</option>
                <option value="student">Student</option>
                <option value="developer">Software Developer</option>
                <option value="data-scientist">Data Scientist</option>
                <option value="engineer">Software Engineer</option>
                <option value="architect">Software Architect</option>
                <option value="other">Other</option>
              </select>
            </div>

            <div class="form-group">
              <label for="hardware-background">Hardware Background:</label>
              <select id="hardware-background" name="hardwareBackground">
                <option value="">Select your hardware background</option>
                <option value="none">No Hardware Background</option>
                <option value="student">Student (Hardware/Robotics)</option>
                <option value="engineer">Hardware Engineer</option>
                <option value="robotics">Robotics Specialist</option>
                <option value="mechanical">Mechanical Engineer</option>
                <option value="electronics">Electronics Engineer</option>
                <option value="other">Other</option>
              </select>
            </div>

            <div class="form-group">
              <label for="experience-level">Experience Level:</label>
              <select id="experience-level" name="experienceLevel">
                <option value="">Select your experience level</option>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
                <option value="expert">Expert</option>
              </select>
            </div>

            <div class="form-group">
              <label for="additional-info">Additional Information (Optional):</label>
              <textarea id="additional-info" name="additionalInfo" placeholder="Tell us more about your background and interests..."></textarea>
            </div>

            <button type="submit">Sign Up</button>
          </form>

          <div class="auth-switch">
            Already have an account? <a href="#" id="show-signin">Sign In</a>
          </div>
        </div>
      </div>
    `;

    document.body.insertAdjacentHTML('beforeend', formHtml);

    // Add event listeners
    document.getElementById('signup-form').addEventListener('submit', (e) => this.handleSignup(e));
    document.getElementById('show-signin').addEventListener('click', (e) => {
      e.preventDefault();
      this.showSigninForm();
    });

    // Add modal close button functionality
    document.getElementById('close-auth-modal').addEventListener('click', () => {
      this.closeModal();
    });

    // Add modal close functionality
    document.getElementById('auth-modal').addEventListener('click', (e) => {
      if (e.target.id === 'auth-modal') {
        this.closeModal();
      }
    });
  }

  // Show sign-in form
  async showSigninForm() {
    // Remove existing modal if present
    const existingModal = document.getElementById('auth-modal');
    if (existingModal) existingModal.remove();

    const formHtml = `
      <div id="auth-modal" class="auth-modal">
        <div class="auth-container">
          <button class="auth-close-btn" id="close-auth-modal">&times;</button>
          <h2>Sign In</h2>
          <form id="signin-form">
            <div class="form-group">
              <label for="email">Email:</label>
              <input type="email" id="signin-email" name="email" required>
            </div>

            <div class="form-group">
              <label for="password">Password:</label>
              <input type="password" id="signin-password" name="password" required>
            </div>

            <button type="submit">Sign In</button>
          </form>

          <div class="auth-switch">
            Don't have an account? <a href="#" id="show-signup">Sign Up</a>
          </div>
        </div>
      </div>
    `;

    document.body.insertAdjacentHTML('beforeend', formHtml);

    // Add event listeners
    document.getElementById('signin-form').addEventListener('submit', (e) => this.handleSignin(e));
    document.getElementById('show-signup').addEventListener('click', (e) => {
      e.preventDefault();
      this.showSignupForm();
    });

    // Add modal close button functionality
    document.getElementById('close-auth-modal').addEventListener('click', () => {
      this.closeModal();
    });

    // Add modal close functionality
    document.getElementById('auth-modal').addEventListener('click', (e) => {
      if (e.target.id === 'auth-modal') {
        this.closeModal();
      }
    });
  }

  // Handle signup
  async handleSignup(event) {
    event.preventDefault();
    const formData = new FormData(event.target);

    const userData = {
      email: formData.get('email'),
      password: formData.get('password'),
      softwareBackground: formData.get('softwareBackground'),
      hardwareBackground: formData.get('hardwareBackground'),
      experienceLevel: formData.get('experienceLevel'),
      additionalInfo: formData.get('additionalInfo')
    };

    try {
      // Use the API endpoint with configurable backend URL
      const backendUrl = window.RAG_CHATBOT_CONFIG?.apiEndpoint || 'http://localhost:8000';
      const apiUrl = backendUrl.startsWith('http') ? `${backendUrl}/api/v1/auth/signup` : `http://localhost:8000/api/v1/auth/signup`;

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      if (response.ok) {
        const result = await response.json();
        this.currentUser = result.user;
        localStorage.setItem('authToken', result.token);
        this.storeUser(result.user); // Store user details
        this.closeModal();
        this.onAuthSuccess('signup');
        alert('Account created successfully!');
      } else {
        const errorResult = await response.json().catch(() => ({}));
        const errorMessage = errorResult.detail || errorResult.error || 'Signup failed';
        alert(errorMessage);
        console.error('Signup error response:', errorResult);
      }
    } catch (error) {
      console.error('Signup error:', error);
      alert('An error occurred during signup. Please check the console for details.');
    }
  }

  // Handle sign-in
  async handleSignin(event) {
    event.preventDefault();
    const formData = new FormData(event.target);

    const credentials = {
      email: formData.get('email'),
      password: formData.get('password')
    };

    try {
      // Use the API endpoint with configurable backend URL
      const backendUrl = window.RAG_CHATBOT_CONFIG?.apiEndpoint || 'http://localhost:8000';
      const apiUrl = backendUrl.startsWith('http') ? `${backendUrl}/api/v1/auth/signin` : `http://localhost:8000/api/v1/auth/signin`;

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      if (response.ok) {
        const result = await response.json();
        this.currentUser = result.user;
        localStorage.setItem('authToken', result.token);
        this.storeUser(result.user); // Store user details
        this.closeModal();
        this.onAuthSuccess('signin');
      } else {
        const errorResult = await response.json().catch(() => ({}));
        const errorMessage = errorResult.detail || errorResult.error || 'Signin failed';
        alert(errorMessage);
        console.error('Signin error response:', errorResult);
      }
    } catch (error) {
      console.error('Signin error:', error);
      alert('An error occurred during signin. Please check the console for details.');
    }
  }

  // Handle sign-out
  async signOut() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('currentUser');
    this.currentUser = null;
    this.onAuthChange();
  }

  // Check if user is authenticated
  isAuthenticated() {
    const token = localStorage.getItem('authToken');
    // In a real implementation, you'd validate the token
    return !!token && !this.isTokenExpired(token);
  }

  // Check if token is expired
  isTokenExpired(token) {
    try {
      // Simple implementation - in real app, decode JWT and check expiration
      // For now just return false, but we should properly decode the token
      // to check if it's expired
      if (!token) return true;

      // If the token is a JWT, we can decode it to check expiration
      if (token.split('.').length === 3) { // JWT has 3 parts separated by dots
        const payload = JSON.parse(atob(token.split('.')[1]));
        const currentTime = Math.floor(Date.now() / 1000);
        return payload.exp < currentTime; // Check if token is expired
      }

      return false; // If not a JWT, assume it's not expired
    } catch (e) {
      console.error('Error checking token expiration:', e);
      return true; // If there's an error checking the token, treat it as expired
    }
  }

  // Get current user
  getCurrentUser() {
    if (this.isAuthenticated()) {
      // If we don't have the user in memory, try to get it from the token
      if (!this.currentUser) {
        const token = localStorage.getItem('authToken');
        if (token) {
          // In a real implementation, you would decode the JWT to get user info
          // For now, we'll try to get it from localStorage or make an API call
          const storedUser = localStorage.getItem('currentUser');
          if (storedUser) {
            try {
              this.currentUser = JSON.parse(storedUser);
            } catch (e) {
              console.error('Error parsing stored user:', e);
            }
          }
        }
      }
      return this.currentUser;
    }
    return null;
  }

  // Store user information
  storeUser(user) {
    this.currentUser = user;
    localStorage.setItem('currentUser', JSON.stringify(user));
  }

  // Close modal
  closeModal() {
    const modal = document.getElementById('auth-modal');
    if (modal) {
      modal.remove();
    }
  }

  // Event handlers - to be overridden by consumers
  onAuthSuccess(action) {
    console.log(`${action} successful`);
    this.onAuthChange();

    // Dispatch custom event for React component to catch
    window.dispatchEvent(new CustomEvent('authChange'));
  }

  onAuthChange() {
    // Update UI based on auth state
    const authButtons = document.getElementById('auth-buttons');
    if (authButtons) {
      if (this.isAuthenticated()) {
        authButtons.innerHTML = `
          <span>Welcome, ${this.getCurrentUser()?.email || 'User'}!</span>
          <button onclick="authSystem.signOut()">Sign Out</button>
        `;
      } else {
        authButtons.innerHTML = `
          <button onclick="authSystem.showSigninForm()">Sign In</button>
          <button onclick="authSystem.showSignupForm()">Sign Up</button>
        `;
      }
    }

    // Dispatch custom event for React component to catch
    window.dispatchEvent(new CustomEvent('authChange'));
  }
}

// Initialize auth system
const authSystem = new AuthSystem();

// Add CSS for auth modal
const authStyles = `
.auth-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.auth-container {
  background: white;
  padding: 2rem;
  border-radius: 8px;
  width: 90%;
  max-width: 500px;
  max-height: 90vh;
  overflow-y: auto;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: bold;
}

.form-group input,
.form-group select,
.form-group textarea {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  box-sizing: border-box;
}

.form-group textarea {
  height: 80px;
  resize: vertical;
}

button[type="submit"] {
  width: 100%;
  padding: 0.75rem;
  background-color: #007cba;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
}

button[type="submit"]:hover {
  background-color: #005a87;
}

.auth-switch {
  margin-top: 1rem;
  text-align: center;
}

.auth-switch a {
  color: #007cba;
  text-decoration: none;
}

.auth-switch a:hover {
  text-decoration: underline;
}

/* Close button for modal */
.auth-close-btn {
  position: absolute;
  top: 10px;
  right: 15px;
  font-size: 24px;
  cursor: pointer;
  background: none;
  border: none;
  color: #999;
}
.auth-close-btn:hover {
  color: #333;
}
`;

// Add styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = authStyles;
document.head.appendChild(styleSheet);

// Initialize auth state on page load
document.addEventListener('DOMContentLoaded', () => {
  authSystem.onAuthChange();
});

// Handle ESC key to close modal
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    const modal = document.getElementById('auth-modal');
    if (modal) {
      authSystem.closeModal();
    }
  }
});

// Export for global use
window.authSystem = authSystem;