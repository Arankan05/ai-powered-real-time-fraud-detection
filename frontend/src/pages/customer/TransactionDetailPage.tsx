import { useEffect, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { AxiosError } from 'axios'
import type { TransactionResponse } from '@/types/transaction'
import * as transactionApi from '@/services/api/transactionApi'
import FraudSummary from '@/components/customer/FraudSummary'
import ScoreBreakdown from '@/components/customer/ScoreBreakdown'
import FraudExplanation from '@/components/customer/FraudExplanation'
import RiskFactors from '@/components/customer/RiskFactors'
import TransactionInfo from '@/components/customer/TransactionInfo'
import { Button } from '@/components/ui/button'

function TransactionDetailPage() {
  const { id } = useParams<{ id: string }>()
  const location = useLocation()
  const stateTx = (location.state as TransactionResponse | null) ?? null

  const [transaction, setTransaction] = useState<TransactionResponse | null>(
    stateTx,
  )
  const [isLoading, setIsLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Fetch full details when router state is missing or stale
  useEffect(() => {
    if (!id) return
    if (transaction && transaction.id === id) return

    let cancelled = false
    setIsLoading(true)
    setErrorMsg(null)

    transactionApi
      .getTransaction(id)
      .then((data) => {
        if (!cancelled) setTransaction(data)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        if (err instanceof AxiosError) {
          if (err.response?.status === 403) {
            setErrorMsg(
              'You are not authorized to view this transaction.',
            )
            return
          }
          if (err.response?.status === 404) {
            setErrorMsg('Transaction not found.')
            return
          }
        }
        setErrorMsg(
          'Failed to load transaction details. Please try again.',
        )
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [id, transaction])

  // Loading state
  if (isLoading) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold text-foreground">
          Transaction Detail
        </h1>
        <p className="text-sm text-muted-foreground">
          Loading transaction details&hellip;
        </p>
      </div>
    )
  }

  // Error state
  if (errorMsg) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold text-foreground">
          Transaction Detail
        </h1>
        <div
          className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
          role="alert"
        >
          {errorMsg}
        </div>
        <div className="flex gap-3">
          <Link to="/customer/history">
            <Button variant="outline">Transaction History</Button>
          </Link>
          <Link to="/customer">
            <Button variant="outline">New Transaction</Button>
          </Link>
        </div>
      </div>
    )
  }

  // Empty state — no data available
  if (!transaction) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold text-foreground">
          Transaction Detail
        </h1>
        <p className="text-muted-foreground">
          Transaction details are not available.
        </p>
        <div className="flex gap-3">
          <Link to="/customer/history">
            <Button variant="outline">Transaction History</Button>
          </Link>
          <Link to="/customer">
            <Button variant="outline">New Transaction</Button>
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">
            Transaction Result
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Fraud analysis for transaction {id}
          </p>
        </div>
        <div className="flex gap-2">
          <Link to="/customer/history">
            <Button variant="outline" size="sm">
              History
            </Button>
          </Link>
          <Link to="/customer">
            <Button variant="outline" size="sm">
              New Transaction
            </Button>
          </Link>
        </div>
      </div>

      {/* Fraud Analysis Summary */}
      <FraudSummary transaction={transaction} />

      {/* Score Breakdown + Explanation side by side on desktop */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <ScoreBreakdown
          mlScore={transaction.ml_score}
          behaviourScore={transaction.behaviour_score}
          ruleScore={transaction.rule_score}
          riskScore={transaction.risk_score}
        />
        <FraudExplanation explanation={transaction.explanation} />
      </div>

      {/* Risk Factors */}
      <RiskFactors factors={transaction.risk_factors} />

      {/* Alert Information */}
      {transaction.alert && (
        <div
          className="rounded-lg border border-amber-500/50 bg-amber-500/10 p-4"
          role="status"
        >
          <h2 className="text-sm font-medium text-foreground">
            Alert Generated
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Alert ID: {transaction.alert.id} &middot; Status:{' '}
            {transaction.alert.status}
          </p>
        </div>
      )}

      {/* Transaction Details */}
      <TransactionInfo transaction={transaction} />
    </div>
  )
}

export default TransactionDetailPage
