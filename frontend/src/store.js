import { create } from 'zustand';

// Try to load initial user from localStorage
let initialUser = null;
try {
  const storedUser = localStorage.getItem('mortgage_user');
  if (storedUser) {
    initialUser = JSON.parse(storedUser);
  }
} catch (e) {
  console.warn('Failed to parse stored user');
}

export const useAppStore = create((set) => ({
  // Auth state
  user: initialUser,
  token: localStorage.getItem('mortgage_token') || null,
  
  login: (userData, tokenData) => {
    localStorage.setItem('mortgage_token', tokenData);
    localStorage.setItem('mortgage_user', JSON.stringify(userData));
    set({ user: userData, token: tokenData });
  },
  
  logout: () => {
    localStorage.removeItem('mortgage_token');
    localStorage.removeItem('mortgage_user');
    set({ user: null, token: null });
  },

  // Global states
  isSidebarCollapsed: false,
  toggleSidebar: () => set((state) => ({ isSidebarCollapsed: !state.isSidebarCollapsed })),

  // Prediction flow state
  currentPrediction: null,
  isPredicting: false,
  predictionError: null,
  
  setPredictionState: (updates) => set((state) => ({ ...state, ...updates })),
  clearPrediction: () => set({ currentPrediction: null, isPredicting: false, predictionError: null }),
}));
