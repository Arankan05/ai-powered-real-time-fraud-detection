/** Query parameters for GET /api/v1/analytics/dashboard */
export interface DashboardQueryParams {
  from_date?: string
  to_date?: string
}

/** Risk distribution counts — matches API contract */
export interface RiskDistribution {
  LOW: number
  MEDIUM: number
  HIGH: number
}

/** Top risk factor — matches API contract */
export interface TopRiskFactor {
  factor: string
  count: number
}

/** Transaction trend data point — matches API contract */
export interface TransactionOverTime {
  date: string
  total: number
  flagged: number
}

/** GET /api/v1/analytics/dashboard — response body (200) */
export interface DashboardResponse {
  from_date: string
  to_date: string
  total_transactions: number
  flagged_transactions: number
  alerts_open: number
  alerts_resolved: number
  risk_distribution: RiskDistribution
  top_risk_factors: TopRiskFactor[]
  transactions_over_time: TransactionOverTime[]
}
