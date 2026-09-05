/** Transaction type enum — matches API contract */
export type TransactionType = 'purchase' | 'transfer' | 'withdrawal'

/** Device type enum — matches API contract */
export type DeviceType = 'mobile' | 'desktop' | 'pos'

/** Transaction status enum — matches API contract */
export type TransactionStatus = 'PENDING' | 'COMPLETED' | 'FAILED'

/** Decision enum — derived from risk scoring thresholds */
export type Decision = 'APPROVE' | 'VERIFY' | 'HOLD'

/** Risk level enum — matches API contract */
export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH'

/**
 * POST /api/v1/transactions — request body.
 * Used for the transaction form schema.
 * NOTE: customer_id is NOT included — the backend derives it from the JWT.
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

// ── Response sub-types ──────────────────────────────────────────────

/** ML top factor in the explanation */
export interface MlTopFactor {
  feature: string
  importance: number
}

/** Behaviour signal in the explanation */
export interface BehaviourSignal {
  signal: string
  severity: number
}

/** Rule triggered in the explanation */
export interface RuleTriggered {
  rule: string
  contribution: number
}

/** Full fraud explanation object */
export interface TransactionExplanation {
  ml_top_factors: MlTopFactor[]
  behaviour_signals: BehaviourSignal[]
  rules_triggered: RuleTriggered[]
}

/** Alert summary — present only when decision is HOLD */
export interface TransactionAlertSummary {
  id: string
  status: string
  created_at: string
}

// ── List / pagination types ─────────────────────────────────────────

/** Query parameters for GET /api/v1/transactions */
export interface TransactionQueryParams {
  page?: number
  per_page?: number
  status?: TransactionStatus
  risk_level?: RiskLevel
  from_date?: string
  to_date?: string
}

/** Single item in the transaction list response */
export interface TransactionSummaryItem {
  id?: string | null
  transaction_id?: string | null
  customer_id: string
  merchant_name: string
  amount: number
  currency: string
  transaction_type: TransactionType
  timestamp: string
  status: TransactionStatus
  risk_score: number
  risk_level: RiskLevel
  decision: Decision
}

/** Paginated response from GET /api/v1/transactions */
export interface TransactionListResponse {
  items: TransactionSummaryItem[]
  total: number
  page: number
  per_page: number
}

// ── Response type ───────────────────────────────────────────────────

/** POST /api/v1/transactions — response body (201) */
export interface TransactionResponse {
  id?: string | null
  transaction_id?: string | null
  customer_id?: string | null
  merchant_id?: string | null
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
  timestamp: string | number
  status: TransactionStatus
  ml_score: number
  behaviour_score: number
  rule_score: number
  risk_score: number
  risk_level: RiskLevel
  decision: Decision
  explanation: TransactionExplanation
  risk_factors: string[]
  model_version: string
  alert?: TransactionAlertSummary | null
}

/** Helper function to get canonical transaction ID from API response or summary */
export function getTransactionId(tx: {
  transaction_id?: string | null
  id?: string | null
}): string {
  return tx.transaction_id || tx.id || ''
}
