import apiClient from './client'
import type {
  User,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  RegisterResponse,
  RefreshRequest,
  RefreshResponse,
} from '@/types/auth'

/** POST /api/v1/auth/login — public */
export async function login(data: LoginRequest): Promise<LoginResponse> {
  const response = await apiClient.post<LoginResponse>('/auth/login', data)
  return response.data
}

/** POST /api/v1/auth/register — public */
export async function register(data: RegisterRequest): Promise<RegisterResponse> {
  const response = await apiClient.post<RegisterResponse>('/auth/register', data)
  return response.data
}

/** POST /api/v1/auth/refresh — auth required */
export async function refresh(data: RefreshRequest): Promise<RefreshResponse> {
  const response = await apiClient.post<RefreshResponse>('/auth/refresh', data)
  return response.data
}

/** GET /api/v1/auth/me — auth required */
export async function getCurrentUser(): Promise<User> {
  const response = await apiClient.get<User>('/auth/me')
  return response.data
}
