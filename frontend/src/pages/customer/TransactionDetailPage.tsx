import { Link, useLocation, useParams } from 'react-router-dom'
import type { TransactionResponse } from '@/types/transaction'
import FraudSummary from '@/components/customer/FraudSummary'
import ScoreBreakdown from '@/components/customer/ScoreBreakdown'
import FraudExplanation from '@/components/customer/FraudExplanation'
import RiskFactors from '@/components/customer/RiskFactors'
import TransactionInfo from '@/components/customer/TransactionInfo'
import { Button } from '@/components/ui/button'

function TransactionDetailPage() {
  const { id } = useParams<{ id: string }>()
  const location = useLocation()
  const transaction = (location.state as TransactionResponse | null) ?? null

  // Empty state — page opened without transaction data
  if (!transaction) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold text-foreground">
          Transaction Detail
        </h1>
        <p className="text-muted-foreground">
          Transaction details are not available.
        </p>
        <Link to="/customer">
          <Button variant="outline">Back to Banking</Button>
        </Link>
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
        <Link to="/customer">
          <Button variant="outline" size="sm">
            New Transaction
          </Button>
        </Link>
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
