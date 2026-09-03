import apiClient from './client'
import type {
  DashboardQueryParams,
  DashboardResponse,
} from '@/types/dashboard'

/** GET /api/v1/analytics/dashboard — auth required (fraud_analyst, admin) */
export async function getDashboard(
  params?: DashboardQueryParams,
): Promise<DashboardResponse> {
  const response = await apiClient.get<DashboardResponse>(
    '/analytics/dashboard',
    { params },
  )
  return response.data
}
