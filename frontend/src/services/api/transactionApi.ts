import apiClient from './client'
import type { TransactionRequest, TransactionResponse } from '@/types/transaction'

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
