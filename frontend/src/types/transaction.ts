/** Transaction type enum — matches API contract */
export type TransactionType = 'purchase' | 'transfer' | 'withdrawal'

/** Device type enum — matches API contract */
export type DeviceType = 'mobile' | 'desktop' | 'pos'

/** Transaction status enum — matches API contract */
export type TransactionStatus = 'PENDING' | 'COMPLETED' | 'FAILED'

/**
 * POST /api/v1/transactions — request body.
 * Used for the transaction form schema.
 */
export interface TransactionRequest {
  amount: number
  currency: string
  merchant_name: string
  merchant_category: string
  transaction_type: TransactionType
  location_country: string
  location_city: string
  device_fingerprint: string
  device_type: DeviceType
  ip_address: string
}
