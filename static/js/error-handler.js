// static/js/error-handler.js
class SkillstackErrorHandler {
    constructor() {
      this.errorTypes = {
        API_ERROR: 'api',
        VALIDATION_ERROR: 'validation',
        AUTH_ERROR: 'auth',
        RESOURCE_ERROR: 'resource',
        SYSTEM_ERROR: 'system'
      };
      
      window.addEventListener('error', this.handleGlobalError.bind(this));
      window.addEventListener('unhandledrejection', this.handlePromiseRejection.bind(this));
    }
    
    handleGlobalError(event) {
      console.error('Global error:', event.error);
      this.showErrorToast('An unexpected error occurred. Please try again.');
      return false;
    }
    
    handlePromiseRejection(event) {
      console.error('Unhandled promise rejection:', event.reason);
      if (event.reason && event.reason.status === 429) {
        this.showErrorToast('API rate limit exceeded. Please try again in a few minutes.');
      }
      return false;
    }
    
    showErrorToast(message, type = 'error', duration = 5000) {
      let container = document.getElementById('skillstack-toast-container');
      if (!container) {
        container = document.createElement('div');
        container.id = 'skillstack-toast-container';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
      }
      
      const toastId = `toast-${Date.now()}`;
      const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type === 'error' ? 'danger' : 'warning'} border-0" role="alert" aria-live="assertive" aria-atomic="true">
          <div class="d-flex">
            <div class="toast-body">
              <i class="bi bi-exclamation-triangle-fill me-2"></i>
              ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
          </div>
        </div>
      `;
      
      container.insertAdjacentHTML('beforeend', toastHtml);
      
      const toastEl = document.getElementById(toastId);
      const toast = new bootstrap.Toast(toastEl, { delay: duration });
      toast.show();
      
      toastEl.addEventListener('hidden.bs.toast', () => {
        toastEl.remove();
      });
    }
    
    validateApiKey(key) {
      if (!key) {
        return { 
          isValid: false, 
          message: 'API key is required'
        };
      }
      
      // Check for project-scoped keys (incompatible format)
      if (key.startsWith('sk-proj-')) {
        return { 
          isValid: false, 
          message: 'Project-scoped API keys (sk-proj-*) are not compatible with this service. Please use a standard API key that starts with "sk-".',
          keyType: 'projectScoped'
        };
      }
      
      // Check for session keys (also incompatible)
      if (key.startsWith('sess-')) {
        return { 
          isValid: false, 
          message: 'Session-scoped keys are not supported. Please use a standard API key.',
          keyType: 'sessionScoped'
        };
      }
      
      // Standard key validation
      if (!key.startsWith('sk-') || key.length < 35) {
        return { 
          isValid: false, 
          message: 'Invalid API key format. API keys should start with "sk-" followed by at least 32 characters.',
          keyType: 'invalid'
        };
      }
      
      return { isValid: true, keyType: 'standard' };
    }
    
    showApiKeyError(inputElement, validationResult) {
      const formGroup = inputElement.closest('.mb-3, .mb-4, .form-group');
      if (!formGroup) return;
      
      inputElement.classList.add('is-invalid');
      
      const existingError = formGroup.querySelector('.api-key-error-container');
      if (existingError) existingError.remove();
      
      const errorContainer = document.createElement('div');
      errorContainer.className = 'api-key-error-container alert alert-danger mt-2';
      
      if (validationResult.keyType === 'projectScoped') {
        errorContainer.innerHTML = `
          <div class="d-flex">
            <div class="me-3">
              <i class="bi bi-shield-exclamation fs-3"></i>
            </div>
            <div>
              <strong>Project-scoped Key Not Supported</strong>
              <p class="mb-1 mt-1">Project-scoped keys (sk-proj-*) cannot be used with this service. Please generate a standard API key.</p>
              <a href="https://platform.openai.com/account/api-keys" class="btn btn-sm btn-outline-light mt-1" target="_blank">
                <i class="bi bi-key me-1"></i> Get Standard API Key
              </a>
            </div>
          </div>
        `;
      } else {
        errorContainer.innerHTML = `
          <div class="d-flex align-items-center">
            <i class="bi bi-exclamation-triangle-fill me-2"></i>
            <span>${validationResult.message}</span>
          </div>
        `;
      }
      
      const inputGroup = formGroup.querySelector('.input-group');
      inputGroup.insertAdjacentElement('afterend', errorContainer);
      
      if (validationResult.keyType === 'projectScoped') {
        this.showErrorToast('Project-scoped API keys are not supported. Please use a standard API key.');
      }
    }
    
    clearApiKeyError(inputElement) {
      inputElement.classList.remove('is-invalid');
      const formGroup = inputElement.closest('.mb-3, .mb-4, .form-group');
      if (formGroup) {
        const errorContainer = formGroup.querySelector('.api-key-error-container');
        if (errorContainer) errorContainer.remove();
      }
    }
  }
  
  // Initialize the error handler
  document.addEventListener('DOMContentLoaded', () => {
    window.errorHandler = new SkillstackErrorHandler();
    
    // Add API key validation to all API key inputs
    document.querySelectorAll('.api-key-input').forEach(input => {
      input.addEventListener('blur', function() {
        const value = this.value.trim();
        if (value) {
          const result = window.errorHandler.validateApiKey(value);
          if (!result.isValid) {
            window.errorHandler.showApiKeyError(this, result);
          } else {
            window.errorHandler.clearApiKeyError(this);
          }
        }
      });
    });
  });