import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`[API Request] ${config.method.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`[API Response] ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('[API Error]', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const analyzeLoan = async (loanData) => {
  const response = await api.post('/analyze', loanData);
  return response.data;
};

export const getHistory = async (limit = 20) => {
  const response = await api.get(`/history?limit=${limit}`);
  // API returns { success: true, data: [...], ... } - extract the array
  return response.data?.data || response.data?.decisions || [];
};

export const compareLoans = async (params) => {
  const { income, loan_amount, credit_score } = params;
  const response = await api.get('/compare', {
    params: { income, loan_amount, credit_score }
  });
  return normalizeCompareResponse(response.data);
};

export const getModels = async () => {
  const response = await api.get('/api/models');
  return response.data;
};

export const getModelComparison = async () => {
  const response = await api.get('/api/models/comparison');
  return response.data;
};

export const compareAllModels = async (applicantData) => {
  const response = await api.post('/api/analyze/compare', applicantData);
  return response.data;
};

export const switchModel = async (modelName) => {
  const response = await api.post(`/api/models/switch/${modelName}`);
  return response.data;
};

export const getFeatureImportance = async (modelName) => {
  const response = await api.get(`/api/models/feature-importance/${modelName}`);
  return response.data;
};

export const explainDecision = async (applicantData, modelName = 'xgboost') => {
  const response = await api.post(`/api/explain?model=${modelName}`, applicantData);
  return response.data;
};

export const explainCompare = async (applicantData) => {
  const response = await api.post('/api/explain/compare', applicantData);
  return response.data;
};

export const explainWhatIf = async (applicantData, changes, modelName = 'xgboost') => {
  const response = await api.post(`/api/explain/what-if?model=${modelName}`, {
    applicant: applicantData,
    changes
  });
  return response.data;
};

// =============================================================================
// API Response Normalizers
// =============================================================================

const normalizeDecision = (decision) => {
  if (!decision || typeof decision !== 'string') return 'pending';
  return decision.toLowerCase();
};

const normalizeRiskLevel = (riskLevel) => {
  if (!riskLevel || typeof riskLevel !== 'string') return 'unknown';
  return riskLevel.toLowerCase();
};

const normalizeScenarioItem = (item) => {
  if (!item || typeof item !== 'object') {
    return {
      loan_amount: 0,
      decision: 'pending',
      emi: 0,
      risk_level: 'unknown',
      default_probability: 0,
      worst_case_emi: 0
    };
  }
  return {
    loan_amount: item.loan_amount ?? 0,
    decision: normalizeDecision(item.decision),
    emi: item.emi ?? 0,
    risk_level: normalizeRiskLevel(item.risk_level),
    default_probability: item.default_probability ?? 0,
    worst_case_emi: item.worst_case_emi ?? 0
  };
};

const normalizeCompareResponse = (data) => {
  if (!data || typeof data !== 'object') {
    return {
      income: 0,
      credit_score: 0,
      comparison: {
        low: normalizeScenarioItem(null),
        medium: normalizeScenarioItem(null),
        high: normalizeScenarioItem(null)
      }
    };
  }

  const comparison = data.comparison || {};

  return {
    income: data.income ?? 0,
    credit_score: data.credit_score ?? 0,
    comparison: {
      low: normalizeScenarioItem(comparison.low),
      medium: normalizeScenarioItem(comparison.medium),
      high: normalizeScenarioItem(comparison.high)
    }
  };
};

export default api;