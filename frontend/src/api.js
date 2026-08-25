const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

function getToken() {
  try { return localStorage.getItem('mortgage_token'); } catch { return null; }
}

export async function fetchApi(endpoint, options = {}) {
  const { timeout = 15000, ...fetchOptions } = options;
  const token = getToken();
  
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...fetchOptions,
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        ...fetchOptions.headers,
      },
      signal: controller.signal,
    });
    
    clearTimeout(id);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      if (response.status === 401) {
        localStorage.removeItem('mortgage_token');
        localStorage.removeItem('mortgage_user');
      }
      throw new Error(errorData.detail || errorData.message || `API error: ${response.status}`);
    }
    
    const data = await response.json();
    return data.success !== undefined ? data.data : data;
  } catch (err) {
    clearTimeout(id);
    if (err.name === 'AbortError') {
      throw new Error('Request timed out. Please try again.');
    }
    throw err;
  }
}

export const api = {
  // Core
  predict: (data, options) => fetchApi('/analyze', { method: 'POST', body: JSON.stringify(data), ...options }),
  health: () => fetchApi('/health'),
  history: (limit = 50) => fetchApi(`/history?limit=${limit}`),

  // Auth
  login: (username, password) => fetchApi('/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) }),
  logout: () => fetchApi('/auth/logout', { method: 'POST' }),
  me: () => fetchApi('/auth/me'),
  users: () => fetchApi('/auth/users'),
  createUser: (data) => fetchApi('/auth/users', { method: 'POST', body: JSON.stringify(data) }),

  // Audit (admin only)
  audit: (params = {}) => {
    const qs = new URLSearchParams(Object.entries(params).filter(([,v]) => v != null)).toString();
    return fetchApi(`/audit?${qs}`);
  },
  auditStats: () => fetchApi('/audit/stats'),

  // What-If Simulator
  whatif: (data) => fetchApi('/whatif', { method: 'POST', body: JSON.stringify(data), timeout: 30000 }),

  // Fairness Analytics (admin)
  fairness: () => fetchApi('/api/analytics/fairness'),

  // Data Retention (admin)
  oldCount: (days) => fetchApi(`/api/data/old-count?older_than_days=${days}`),
  purge: (days) => fetchApi(`/api/data/purge?older_than_days=${days}`, { method: 'DELETE' }),
};

// ─── CSV Export Helper ───────────────────────────────────────────────────────
export function exportToCSV(rows, filename = 'mortgage_history.csv') {
  if (!rows || rows.length === 0) return;
  const headers = Object.keys(rows[0]);
  const csv = [
    headers.join(','),
    ...rows.map(row => headers.map(h => {
      const val = row[h] ?? '';
      const str = String(val).replace(/"/g, '""');
      return str.includes(',') || str.includes('"') || str.includes('\n') ? `"${str}"` : str;
    }).join(','))
  ].join('\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
