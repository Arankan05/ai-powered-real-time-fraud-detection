import apiClient from './client'
import type {
  AlertListResponse,
  AlertQueryParams,
  AlertDetail,
  AlertUpdateRequest,
  AlertUpdateResponse,
} from '@/types/alert'

/** GET /api/v1/alerts — auth required (fraud_analyst, admin) */
export async function getAlerts(
  params?: AlertQueryParams,
): Promise<AlertListResponse> {
  const response = await apiClient.get<AlertListResponse>(
    '/alerts',
    { params },
  )
  return response.data
}

/** GET /api/v1/alerts/:id — auth required (fraud_analyst, admin) */
export async function getAlert(
  id: string,
): Promise<AlertDetail> {
  const response = await apiClient.get<AlertDetail>(
    `/alerts/${id}`,
  )
  return response.data
}

/** PATCH /api/v1/alerts/:id — auth required (fraud_analyst, admin) */
export async function updateAlert(
  id: string,
  data: AlertUpdateRequest,
): Promise<AlertUpdateResponse> {
  const response = await apiClient.patch<AlertUpdateResponse>(
    `/alerts/${id}`,
    data,
  )
  return response.data
}
