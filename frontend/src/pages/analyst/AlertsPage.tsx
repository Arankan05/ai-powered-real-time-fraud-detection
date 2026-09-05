import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AxiosError } from 'axios'
import * as alertApi from '@/services/api/alertApi'
import type {
  AlertSummaryItem,
  AlertListResponse,
  AlertStatus,
} from '@/types/alert'
import type { RiskLevel } from '@/types/transaction'
import { Button } from '@/components/ui/button'

const riskStyles: Record<string, string> = {
  LOW: 'bg-emerald-500/10 text-emerald-700',
  MEDIUM: 'bg-amber-500/10 text-amber-700',
  HIGH: 'bg-red-500/10 text-red-700',
}

const decisionStyles: Record<string, string> = {
  APPROVE: 'bg-emerald-500/10 text-emerald-700',
  VERIFY: 'bg-amber-500/10 text-amber-700',
  HOLD: 'bg-red-500/10 text-red-700',
}

const statusStyles: Record<string, string> = {
  OPEN: 'bg-red-500/10 text-red-700',
  IN_REVIEW: 'bg-blue-500/10 text-blue-700',
  RESOLVED: 'bg-emerald-500/10 text-emerald-700',
  DISMISSED: 'bg-gray-500/10 text-gray-600',
}

function formatDate(iso: string): string {
  if (!iso) return 'N/A'
  try {
    return new Date(iso).toLocaleString(undefined, {
      dateStyle: 'short',
      timeStyle: 'short',
    })
  } catch {
    return iso
  }
}

function formatAmount(amount?: number | null, currency?: string | null): string {
  const val = amount ?? 0
  const curr = currency || 'USD'
  try {
    return new Intl.NumberFormat(undefined, {
      style: 'currency',
      currency: curr,
    }).format(val)
  } catch {
    return `${curr} ${val.toFixed(2)}`
  }
}

function getMerchantName(alert: AlertSummaryItem): string {
  return alert.transaction_summary?.merchant_name ?? alert.merchant_name ?? 'N/A'
}

function getAmount(alert: AlertSummaryItem): number {
  return alert.transaction_summary?.amount ?? alert.amount ?? 0
}

function getCurrency(alert: AlertSummaryItem): string {
  return alert.transaction_summary?.currency ?? alert.currency ?? 'USD'
}

function AlertsPage() {
  const navigate = useNavigate()
  const [alerts, setAlerts] = useState<AlertSummaryItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [perPage] = useState(20)
  const [statusFilter, setStatusFilter] = useState<AlertStatus | ''>('')
  const [riskFilter, setRiskFilter] = useState<RiskLevel | ''>('')
  const [isLoading, setIsLoading] = useState(true)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const totalPages = Math.max(1, Math.ceil(total / perPage))

  const fetchAlerts = useCallback(async () => {
    setIsLoading(true)
    setErrorMsg(null)
    try {
      const params: Record<string, string | number> = { page, per_page: perPage }
      if (statusFilter) params.status = statusFilter
      if (riskFilter) params.risk_level = riskFilter

      const result: AlertListResponse = await alertApi.getAlerts(params)
      setAlerts(result.items)
      setTotal(result.total)
    } catch (err) {
      if (err instanceof AxiosError) {
        if (err.response?.status === 403) {
          setErrorMsg('You are not authorized to view alerts.')
          return
        }
        setErrorMsg('Unable to load alerts. Please try again.')
        return
      }
      setErrorMsg('Unable to connect to the server.')
    } finally {
      setIsLoading(false)
    }
  }, [page, perPage, statusFilter, riskFilter])

  useEffect(() => {
    void fetchAlerts()
  }, [fetchAlerts])

  const handleStatusChange = (value: string) => {
    setStatusFilter(value as AlertStatus | '')
    setPage(1)
  }

  const handleRiskChange = (value: string) => {
    setRiskFilter(value as RiskLevel | '')
    setPage(1)
  }

  const hasActiveFilters = Boolean(statusFilter || riskFilter)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">
          Alerts
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Fraud alerts requiring analyst review and resolution.
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <label className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Status</span>
          <select
            value={statusFilter}
            onChange={(e) => handleStatusChange(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm"
          >
            <option value="">All statuses</option>
            <option value="OPEN">Open</option>
            <option value="IN_REVIEW">In Review</option>
            <option value="RESOLVED">Resolved</option>
            <option value="DISMISSED">Dismissed</option>
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Risk Level</span>
          <select
            value={riskFilter}
            onChange={(e) => handleRiskChange(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm"
          >
            <option value="">All risk levels</option>
            <option value="LOW">Low</option>
            <option value="MEDIUM">Medium</option>
            <option value="HIGH">High</option>
          </select>
        </label>
      </div>

      {/* Error */}
      {errorMsg && (
        <div className="space-y-3">
          <div
            className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
            role="alert"
          >
            {errorMsg}
          </div>
          <Button variant="outline" size="sm" onClick={fetchAlerts}>
            Retry
          </Button>
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <p className="py-12 text-center text-sm text-muted-foreground">
          Loading alerts&hellip;
        </p>
      )}

      {/* Empty */}
      {!isLoading && !errorMsg && alerts.length === 0 && (
        <div className="flex flex-col items-center gap-2 py-12 text-center">
          <p className="text-muted-foreground">
            {hasActiveFilters
              ? 'No alerts match the selected filters.'
              : 'No alerts found.'}
          </p>
        </div>
      )}

      {/* Table (desktop) */}
      {!isLoading && alerts.length > 0 && (
        <>
          <div className="hidden overflow-x-auto rounded-lg border md:block">
            <table className="w-full text-left text-sm">
              <thead className="border-b bg-muted/50">
                <tr>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Status
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Risk
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Score
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Decision
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Merchant
                  </th>
                  <th scope="col" className="px-4 py-3 text-right font-medium text-muted-foreground">
                    Amount
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Created
                  </th>
                  <th scope="col" className="px-4 py-3">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {alerts.map((alert) => (
                  <tr
                    key={alert.id}
                    className="transition-colors hover:bg-muted/30"
                  >
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${statusStyles[alert.status] ?? ''}`}
                      >
                        {alert.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${riskStyles[alert.risk_level] ?? ''}`}
                      >
                        {alert.risk_level}
                      </span>
                    </td>
                    <td className="px-4 py-3 tabular-nums">
                      {alert.risk_score}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${decisionStyles[alert.decision] ?? ''}`}
                      >
                        {alert.decision}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-medium">
                      {getMerchantName(alert)}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums">
                      {formatAmount(getAmount(alert), getCurrency(alert))}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-muted-foreground">
                      {formatDate(alert.created_at)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        type="button"
                        onClick={() => navigate(`/analyst/alerts/${alert.id}`)}
                        className="text-sm font-medium text-primary underline-offset-2 hover:underline"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Cards (mobile / tablet) */}
          <div className="space-y-3 md:hidden">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className="space-y-3 rounded-lg border p-4"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="font-medium">
                      {getMerchantName(alert)}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatDate(alert.created_at)}
                    </p>
                  </div>
                  <p className="whitespace-nowrap text-right font-medium tabular-nums">
                    {formatAmount(getAmount(alert), getCurrency(alert))}
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${statusStyles[alert.status] ?? ''}`}
                  >
                    {alert.status}
                  </span>
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${riskStyles[alert.risk_level] ?? ''}`}
                  >
                    {alert.risk_level}
                  </span>
                  <span className="text-xs tabular-nums text-muted-foreground">
                    Score: {alert.risk_score}
                  </span>
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${decisionStyles[alert.decision] ?? ''}`}
                  >
                    {alert.decision}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => navigate(`/analyst/alerts/${alert.id}`)}
                  className="text-sm font-medium text-primary underline-offset-2 hover:underline"
                >
                  View Details
                </button>
              </div>
            ))}
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Showing {(page - 1) * perPage + 1}&ndash;
              {Math.min(page * perPage, total)} of {total} alerts
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default AlertsPage
