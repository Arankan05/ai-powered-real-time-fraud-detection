import apiClient from './client'
import type {
  TransactionRequest,
  TransactionResponse,
  TransactionQueryParams,
  TransactionListResponse,
} from '@/types/transaction'

/** POST /api/v1/transactions — auth required (customer) */
export async function createTransaction(
  data: TransactionRequest,
): Promise<TransactionResponse> {
  const response = await apiClient.post<TransactionResponse>(
    '/transactions',
    data,
  )
  return response.data
}

/** GET /api/v1/transactions — customers see own, analysts/admins see all */
export async function getTransactions(
  params?: TransactionQueryParams,
): Promise<TransactionListResponse> {
  const response = await apiClient.get<TransactionListResponse>(
    '/transactions',
    { params },
  )
  return response.data
}

/** GET /api/v1/transactions/:id — full detail including fraud analysis */
export async function getTransaction(
  id: string,
): Promise<TransactionResponse> {
  const response = await apiClient.get<TransactionResponse>(
    `/transactions/${id}`,
  )
  return response.data
}
