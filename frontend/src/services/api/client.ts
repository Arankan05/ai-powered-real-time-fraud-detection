import axios, { AxiosError } from 'axios'
import type { AxiosResponse } from 'axios'
import {
  getAccessToken,
  getRefreshToken,
  setTokens,
  clearTokens,
} from './tokenStorage'

/**
 * Reusable Axios instance configured with the API base URL.
 * Includes request interceptor (Bearer token) and response interceptor
 * (automatic 401 refresh + 403 preservation).
 */
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// ── Request interceptor ────────────────────────────────────────────
// Attaches the Bearer token to every outgoing request when available.
apiClient.interceptors.request.use((config) => {
  const token = getAccessToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// ── Response interceptor ───────────────────────────────────────────
// Shared state to prevent concurrent refresh requests.
let refreshPromise: Promise<string> | null = null

apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config
    if (!originalRequest) return Promise.reject(error)

    // ── 401 handling: token refresh + single retry ─────────────
    if (
      error.response?.status === 401 &&
      !(originalRequest as unknown as Record<string, unknown>)._retry &&
      // Never retry the refresh endpoint itself
      !originalRequest.url?.includes('/auth/refresh')
    ) {
      // Mark this request so it is never retried a second time
      (originalRequest as unknown as Record<string, unknown>)._retry = true

      const refreshToken = getRefreshToken()
      if (!refreshToken) {
        clearTokens()
        return Promise.reject(error)
      }

      try {
        // Re-use a single in-flight refresh call for concurrent 401s
        if (!refreshPromise) {
          refreshPromise = (async () => {
            // Raw axios call — bypasses apiClient interceptors entirely
            const response = await axios.post(
              `${import.meta.env.VITE_API_BASE_URL}/auth/refresh`,
              { refresh_token: refreshToken },
            )
            const { access_token, refresh_token } = response.data
            setTokens(access_token, refresh_token)
            return access_token as string
          })()
        }

        const newAccessToken = await refreshPromise
        refreshPromise = null

        // Retry the original request with the fresh token
        originalRequest.headers.Authorization = `Bearer ${newAccessToken}`
        return apiClient(originalRequest)
      } catch {
        // Refresh failed — clear auth state, no infinite loop
        refreshPromise = null
        clearTokens()
        return Promise.reject(error)
      }
    }

    // ── 403 handling: preserve and propagate ───────────────────
    // The full error (including detail / error_code) propagates to
    // callers so UI layers can handle unauthorised access later.
    // 403 is intentionally NOT converted to 401.

    return Promise.reject(error)
  },
)

export default apiClient
