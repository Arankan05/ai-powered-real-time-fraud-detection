/** User role enum — matches API contract */
export type UserRole = 'customer' | 'fraud_analyst' | 'admin'

/** POST /api/v1/auth/login — request body */
export interface LoginRequest {
  email: string
  password: string
}

/** POST /api/v1/auth/login — response body (200) */
export interface LoginResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

/** POST /api/v1/auth/register — request body */
export interface RegisterRequest {
  email: string
  password: string
  first_name: string
  last_name: string
  phone: string
  date_of_birth: string
  address: string
}

/** POST /api/v1/auth/register — response body (201) */
export interface RegisterResponse {
  id: string
  email: string
  first_name: string
  last_name: string
  role: UserRole
  customer_id: string
}

/** POST /api/v1/auth/refresh — request body */
export interface RefreshRequest {
  refresh_token: string
}

/** POST /api/v1/auth/refresh — response body (200) */
export interface RefreshResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

/** GET /api/v1/auth/me — response body (200) */
export interface User {
  id: string
  email: string
  first_name: string
  last_name: string
  role: UserRole
  customer_id: string | null
  is_active: boolean
  created_at: string
}

/** Internal auth state managed by AuthContext */
export interface AuthState {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  isInitializing: boolean
}

/** Standard API error response — matches API contract */
export interface ApiError {
  detail: string
  error_code: string
  errors?: Array<{ field: string; message: string }>
}
