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

// ── Response type ───────────────────────────────────────────────────

/** POST /api/v1/transactions — response body (201) */
export interface TransactionResponse {
  id: string
  customer_id: string
  merchant_id: string
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
  timestamp: string
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
  alert: TransactionAlertSummary | null
}
