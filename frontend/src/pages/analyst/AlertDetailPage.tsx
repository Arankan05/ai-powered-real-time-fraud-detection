import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { AxiosError } from 'axios'
import * as alertApi from '@/services/api/alertApi'
import type { AlertDetail } from '@/types/alert'
import FraudExplanation from '@/components/customer/FraudExplanation'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
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

function formatDate(iso?: string | null): string {
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

function DetailRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs font-medium text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 text-sm text-foreground">{children}</dd>
    </div>
  )
}

function AlertDetailPage() {
  const { id } = useParams<{ id: string }>()
  const [alert, setAlert] = useState<AlertDetail | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const fetchAlert = useCallback(async () => {
    if (!id) return
    setIsLoading(true)
    setErrorMsg(null)
    try {
      const data = await alertApi.getAlert(id)
      setAlert(data)
    } catch (err) {
      if (err instanceof AxiosError) {
        if (err.response?.status === 403) {
          setErrorMsg('You are not authorized to view this alert.')
          return
        }
        if (err.response?.status === 404) {
          setErrorMsg('Alert not found.')
          return
        }
        setErrorMsg('Unable to load alert details. Please try again.')
        return
      }
      setErrorMsg('Unable to connect to the server.')
    } finally {
      setIsLoading(false)
    }
  }, [id])

  useEffect(() => {
    void fetchAlert()
  }, [fetchAlert])

  // Missing route param
  if (!id) {
    return (
      <div className="space-y-6">
        <Link
          to="/analyst/alerts"
          className="text-sm font-medium text-primary underline-offset-2 hover:underline"
        >
          &larr; Back to Alerts
        </Link>
        <p className="text-muted-foreground">Alert not found.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Link
        to="/analyst/alerts"
        className="inline-block text-sm font-medium text-primary underline-offset-2 hover:underline"
      >
        &larr; Back to Alerts
      </Link>

      <h1 className="text-2xl font-semibold text-foreground">
        Alert Detail
      </h1>

      {/* Loading */}
      {isLoading && (
        <p className="py-12 text-center text-sm text-muted-foreground">
          Loading alert details&hellip;
        </p>
      )}

      {/* Error */}
      {errorMsg && (
        <div className="space-y-3">
          <div
            className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
            role="alert"
          >
            {errorMsg}
          </div>
          <Button variant="outline" size="sm" onClick={fetchAlert}>
            Retry
          </Button>
        </div>
      )}

      {/* Content */}
      {!isLoading && !errorMsg && alert && (
        <>
          {/* Alert Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Alert Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                <DetailRow label="Alert ID">
                  <span className="font-mono text-xs">{alert.id}</span>
                </DetailRow>
                <DetailRow label="Transaction ID">
                  <span className="font-mono text-xs">{alert.transaction_id}</span>
                </DetailRow>
                <DetailRow label="Status">
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${statusStyles[alert.status] ?? ''}`}
                  >
                    {alert.status}
                  </span>
                </DetailRow>
                <DetailRow label="Risk Level">
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${riskStyles[alert.risk_level] ?? ''}`}
                  >
                    {alert.risk_level}
                  </span>
                </DetailRow>
                <DetailRow label="Risk Score">
                  <span className="tabular-nums">{alert.risk_score}</span>
                </DetailRow>
                <DetailRow label="Decision">
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${decisionStyles[alert.decision] ?? ''}`}
                  >
                    {alert.decision}
                  </span>
                </DetailRow>
                <DetailRow label="Created">
                  {formatDate(alert.created_at)}
                </DetailRow>
                <DetailRow label="Resolved">
                  {alert.resolved_at ? formatDate(alert.resolved_at) : 'Not resolved'}
                </DetailRow>
                <DetailRow label="Analyst">
                  {alert.analyst_id ? (
                    <span className="font-mono text-xs">{alert.analyst_id}</span>
                  ) : (
                    'Not assigned'
                  )}
                </DetailRow>
              </dl>
            </CardContent>
          </Card>

          {/* Notes */}
          <Card>
            <CardHeader>
              <CardTitle>Notes</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                {alert.notes || 'No notes.'}
              </p>
            </CardContent>
          </Card>

          {/* Fraud Explanation */}
          <FraudExplanation explanation={alert.explanation} />

          {/* Risk Factors */}
          <Card>
            <CardHeader>
              <CardTitle>Risk Factors</CardTitle>
            </CardHeader>
            <CardContent>
              {alert.risk_factors && alert.risk_factors.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {alert.risk_factors.map((factor) => (
                    <span
                      key={factor}
                      className="inline-flex rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-foreground"
                    >
                      {factor}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  No risk factors identified.
                </p>
              )}
            </CardContent>
          </Card>

          {/* Transaction Information */}
          <Card>
            <CardHeader>
              <CardTitle>Transaction Information</CardTitle>
            </CardHeader>
            <CardContent>
              {alert.transaction ? (
                <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  <DetailRow label="Transaction ID">
                    <span className="font-mono text-xs">{alert.transaction.id}</span>
                  </DetailRow>
                  <DetailRow label="Customer ID">
                    <span className="font-mono text-xs">{alert.transaction.customer_id}</span>
                  </DetailRow>
                  <DetailRow label="Merchant">
                    {alert.transaction.merchant_name}
                  </DetailRow>
                  <DetailRow label="Amount">
                    <span className="tabular-nums">
                      {formatAmount(alert.transaction.amount, alert.transaction.currency)}
                    </span>
                  </DetailRow>
                  <DetailRow label="Type">
                    <span className="capitalize">{alert.transaction.transaction_type}</span>
                  </DetailRow>
                  <DetailRow label="Location">
                    {alert.transaction.location_city}, {alert.transaction.location_country}
                  </DetailRow>
                  <DetailRow label="Device Type">
                    <span className="capitalize">{alert.transaction.device_type}</span>
                  </DetailRow>
                  <DetailRow label="Timestamp">
                    {formatDate(alert.transaction.timestamp)}
                  </DetailRow>
                </dl>
              ) : (
                <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  <DetailRow label="Transaction ID">
                    <span className="font-mono text-xs">{alert.transaction_id}</span>
                  </DetailRow>
                  <DetailRow label="Merchant">
                    {alert.transaction_summary?.merchant_name ?? 'N/A'}
                  </DetailRow>
                  <DetailRow label="Amount">
                    <span className="tabular-nums">
                      {formatAmount(alert.transaction_summary?.amount, alert.transaction_summary?.currency)}
                    </span>
                  </DetailRow>
                </dl>
              )}
            </CardContent>
          </Card>

          {/* Scores */}
          <Card>
            <CardHeader>
              <CardTitle>Scoring Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <DetailRow label="ML Score">
                  <span className="text-2xl font-semibold tabular-nums">
                    {alert.transaction?.ml_score ?? 'N/A'}
                  </span>
                </DetailRow>
                <DetailRow label="Behaviour Score">
                  <span className="text-2xl font-semibold tabular-nums">
                    {alert.transaction?.behaviour_score ?? 'N/A'}
                  </span>
                </DetailRow>
                <DetailRow label="Rule Score">
                  <span className="text-2xl font-semibold tabular-nums">
                    {alert.transaction?.rule_score ?? 'N/A'}
                  </span>
                </DetailRow>
              </dl>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}

export default AlertDetailPage
