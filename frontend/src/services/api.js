import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API service functions
export const apiService = {
  // Health check
  healthCheck: () => api.get('/health'),
  
  // Dashboard stats
  getDashboardStats: () => api.get('/dashboard/stats'),
  
  // Plans management
  getPlans: (params = {}) => api.get('/plans', { params }),
  getPlanDetails: (planId) => api.get(`/plan/${planId}`),
  getFilterOptions: () => api.get('/filters/options'),
  
  // Optimization
  startOptimization: (data) => api.post('/optimize', data),
  getJobStatus: (jobId) => api.get(`/job/${jobId}`),
  getResults: (jobId, params = {}) => api.get(`/results/${jobId}`, { params }),
};

export default api;
