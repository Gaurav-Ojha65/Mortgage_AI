import { useState, useCallback, useRef } from 'react';

/**
 * Custom hook for API calls with loading, error, and abort support.
 * Additive only - does not refactor existing code.
 *
 * @param {Function} apiFn - The API function to call
 * @returns {Object} { data, loading, error, execute, abort, reset }
 */
export const useApi = (apiFn) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortControllerRef = useRef(null);

  const abort = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    abort();
    setData(null);
    setLoading(false);
    setError(null);
  }, [abort]);

  const execute = useCallback(async (...args) => {
    abort();
    abortControllerRef.current = new AbortController();

    setLoading(true);
    setError(null);

    try {
      const result = await apiFn(...args, {
        signal: abortControllerRef.current.signal
      });
      setData(result);
      return { success: true, data: result };
    } catch (err) {
      if (err.name === 'AbortError') {
        return { success: false, aborted: true };
      }
      const errorMessage = err.response?.data?.detail || err.message || 'Request failed';
      setError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  }, [apiFn, abort]);

  return {
    data,
    loading,
    error,
    execute,
    abort,
    reset
  };
};

export default useApi;
