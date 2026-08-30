import type { RiskLevel, Decision, TransactionExplanation } from './transaction'

/** Alert status enum — matches API contract (alert_status) */
export type AlertStatus = 'OPEN' | 'IN_REVIEW' | 'RESOLVED' | 'DISMISSED'

// ── List / pagination types ─────────────────────────────────────────

/** Nested transaction summary inside an alert list item */
export interface AlertTransactionSummary {
  amount: number
  currency: string
  merchant_name: string
  transaction_type: string
  customer_email: string
  timestamp: string
}

/** Single item in the GET /api/v1/alerts list response */
export interface AlertSummaryItem {
  id: string
  transaction_id: string
  risk_score: number
  risk_level: RiskLevel
  decision: Decision
  status: AlertStatus
  analyst_id: string | null
  notes: string | null
  created_at: string
  resolved_at: string | null
  transaction_summary: AlertTransactionSummary
}

/** Paginated response from GET /api/v1/alerts */
export interface AlertListResponse {
  items: AlertSummaryItem[]
  total: number
  page: number
  per_page: number
}

/** Query parameters for GET /api/v1/alerts */
export interface AlertQueryParams {
  page?: number
  per_page?: number
  status?: AlertStatus
  risk_level?: RiskLevel
}

// ── Detail types ────────────────────────────────────────────────────

/** Full transaction object returned inside GET /api/v1/alerts/{id} */
export interface AlertTransaction {
  id: string
  customer_id: string
  amount: number
  currency: string
  merchant_name: string
  transaction_type: string
  location_country: string
  location_city: string
  device_type: string
  timestamp: string
  ml_score: number
  behaviour_score: number
  rule_score: number
}

/** Response from GET /api/v1/alerts/{id} — full alert with explanation and transaction */
export interface AlertDetail {
  id: string
  transaction_id: string
  risk_score: number
  risk_level: RiskLevel
  decision: Decision
  explanation: TransactionExplanation
  risk_factors: string[]
  status: AlertStatus
  analyst_id: string | null
  notes: string | null
  created_at: string
  resolved_at: string | null
  transaction: AlertTransaction
}

// ── Mutation types ──────────────────────────────────────────────────

/** Request body for PATCH /api/v1/alerts/{id} — both fields optional, at least one required */
export interface AlertUpdateRequest {
  status?: AlertStatus
  notes?: string
}

/** Response from PATCH /api/v1/alerts/{id} — does NOT include transaction_summary */
export interface AlertUpdateResponse {
  id: string
  transaction_id: string
  risk_score: number
  risk_level: RiskLevel
  decision: Decision
  status: AlertStatus
  analyst_id: string | null
  notes: string | null
  created_at: string
  resolved_at: string | null
}
