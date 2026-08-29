import {
  createContext,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react'
import type { ReactNode } from 'react'
import type { User, LoginRequest, RegisterRequest } from '@/types/auth'
import * as authApi from '@/services/api/authApi'
import {
  getAccessToken,
  setTokens,
  clearTokens,
} from '@/services/api/tokenStorage'

/** Shape of the value provided by AuthProvider */
export interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  isInitializing: boolean
  login: (data: LoginRequest) => Promise<void>
  register: (data: RegisterRequest) => Promise<void>
  logout: () => void
}

export const AuthContext = createContext<AuthContextType | null>(null)

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isInitializing, setIsInitializing] = useState(true)

  // ── Initialise auth state on mount ─────────────────────────────
  useEffect(() => {
    let cancelled = false

    async function initializeAuth() {
      const token = getAccessToken()
      if (!token) {
        setIsInitializing(false)
        return
      }

      try {
        const currentUser = await authApi.getCurrentUser()
        if (!cancelled) {
          setUser(currentUser)
        }
      } catch {
        // Token invalid or refresh failed — interceptor already cleared tokens
        if (!cancelled) {
          setUser(null)
          clearTokens()
        }
      } finally {
        if (!cancelled) {
          setIsInitializing(false)
        }
      }
    }

    initializeAuth()

    return () => {
      cancelled = true
    }
  }, [])

  // ── Actions ────────────────────────────────────────────────────
  const login = useCallback(async (data: LoginRequest) => {
    setIsLoading(true)
    try {
      const response = await authApi.login(data)
      setTokens(response.access_token, response.refresh_token)
      const currentUser = await authApi.getCurrentUser()
      setUser(currentUser)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const register = useCallback(async (data: RegisterRequest) => {
    setIsLoading(true)
    try {
      await authApi.register(data)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const logout = useCallback(() => {
    clearTokens()
    setUser(null)
  }, [])

  const value = useMemo<AuthContextType>(
    () => ({
      user,
      isAuthenticated: user !== null,
      isLoading,
      isInitializing,
      login,
      register,
      logout,
    }),
    [user, isLoading, isInitializing, login, register, logout],
  )

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}
