/**
 * Global error handler for CourseGen AI application (No jQuery dependency)
 */
(function() {
  // Error state tracking
  const errorState = {
      hasActiveError: false,
      lastErrorTime: null
  };

  /**
   * Display error message to user with appropriate styling
   */
  function showErrorMessage(message, context = 'general', targetElement = null) {
      // Prevent error message spam
      if (errorState.hasActiveError && 
          (Date.now() - errorState.lastErrorTime < 5000)) {
          return;
      }
      
      errorState.hasActiveError = true;
      errorState.lastErrorTime = Date.now();
      
      // Create or find error container
      let errorContainer;
      
      if (targetElement) {
          // Create inline error near the specific element
          errorContainer = document.createElement('div');
          errorContainer.className = 'alert alert-danger mt-2 error-message';
          errorContainer.setAttribute('role', 'alert');
          targetElement.parentNode.insertBefore(errorContainer, targetElement.nextSibling);
      } else {
          // Create or use global error container
          errorContainer = document.getElementById('global-error-container');
          
          if (!errorContainer) {
              errorContainer = document.createElement('div');
              errorContainer.id = 'global-error-container';
              errorContainer.className = 'alert alert-danger alert-dismissible fade show fixed-top mx-auto mt-3 error-message';
              errorContainer.style.maxWidth = '80%';
              errorContainer.style.width = '600px';
              errorContainer.style.zIndex = '9999';
              errorContainer.setAttribute('role', 'alert');
              
              // Add close button
              const closeButton = document.createElement('button');
              closeButton.type = 'button';
              closeButton.className = 'btn-close';
              closeButton.setAttribute('data-bs-dismiss', 'alert');
              closeButton.setAttribute('aria-label', 'Close');
              errorContainer.appendChild(closeButton);
              
              document.body.appendChild(errorContainer);
          }
      }
      
      // Set error message with context
      errorContainer.innerHTML = `<strong>Error${context !== 'general' ? ' (' + context + ')' : ''}:</strong> ${message}`;
      
      if (!targetElement) {
          // Auto dismiss global errors after 5 seconds
          setTimeout(() => {
              if (errorContainer && errorContainer.parentNode) {
                  errorContainer.remove();
                  errorState.hasActiveError = false;
              }
          }, 5000);
      }
  }
  
  /**
   * Handle AJAX errors consistently without jQuery dependency
   */
  function handleAjaxError(error, context = 'server', targetElement = null) {
      console.error('AJAX Error:', error);
      
      let errorMessage = 'An unexpected error occurred.';
      
      // Extract meaningful error message based on error type
      if (error.response) {
          try {
              const contentType = error.response.headers.get('content-type');
              if (contentType && contentType.includes('application/json')) {
                  error.response.json().then(data => {
                      errorMessage = data.error || errorMessage;
                      showErrorMessage(errorMessage, context, targetElement);
                  }).catch(() => {
                      showErrorMessage(errorMessage, context, targetElement);
                  });
                  return;
              }
          } catch (e) {
              console.error('Error parsing response:', e);
          }
      }
      
      // Handle other error types
      if (error.message) {
          errorMessage = error.message;
      } else if (typeof error === 'string') {
          errorMessage = error;
      }
      
      showErrorMessage(errorMessage, context, targetElement);
  }
  
  /**
   * Setup global error handling with Fetch API support
   */
  function setupGlobalErrorHandling() {
      // Handle global JS errors
      window.addEventListener('error', function(event) {
          console.error('Global error:', event.error);
          // Only show UI errors for user interactions, not for script loading
          if (event.filename && !event.filename.includes('static/js/')) {
              showErrorMessage('JavaScript error: ' + (event.error ? event.error.message : 'Unknown error'));
          }
      });
      
      // Handle unhandled promise rejections
      window.addEventListener('unhandledrejection', function(event) {
          console.error('Unhandled promise rejection:', event.reason);
          if (event.reason.message) {
              showErrorMessage('Promise error: ' + event.reason.message);
          }
      });
      
      // Optional: Enhance fetch API with error handling if needed
      const originalFetch = window.fetch;
      window.fetch = function(...args) {
          return originalFetch.apply(this, args).catch(error => {
              handleAjaxError(error);
              throw error;
          });
      };
  }
  
  // Initialize error handling when DOM is ready
  document.addEventListener('DOMContentLoaded', function() {
      setupGlobalErrorHandling();
  });
  
  // Export public API
  window.ErrorHandler = {
      showError: showErrorMessage,
      handleAjaxError: handleAjaxError
  };
})();