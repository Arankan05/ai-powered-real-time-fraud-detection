import type { TransactionResponse } from '@/types/transaction'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'

interface FraudSummaryProps {
  transaction: TransactionResponse
}

const riskStyles: Record<string, string> = {
  LOW: 'border-emerald-500/50 bg-emerald-500/10 text-emerald-700',
  MEDIUM: 'border-amber-500/50 bg-amber-500/10 text-amber-700',
  HIGH: 'border-red-500/50 bg-red-500/10 text-red-700',
}

const decisionStyles: Record<string, string> = {
  APPROVE: 'border-emerald-500/50 bg-emerald-500/10 text-emerald-700',
  VERIFY: 'border-amber-500/50 bg-amber-500/10 text-amber-700',
  HOLD: 'border-red-500/50 bg-red-500/10 text-red-700',
}

function FraudSummary({ transaction }: FraudSummaryProps) {
  const { risk_score, risk_level, decision } = transaction

  return (
    <Card>
      <CardHeader>
        <CardTitle>Fraud Analysis Summary</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
          {/* Risk Score */}
          <div className="text-center">
            <p className="text-sm font-medium text-muted-foreground">
              Risk Score
            </p>
            <p className="mt-1 text-4xl font-bold tabular-nums text-foreground">
              {risk_score}
            </p>
            <p className="text-xs text-muted-foreground">out of 100</p>
          </div>

          {/* Risk Level */}
          <div className="text-center">
            <p className="text-sm font-medium text-muted-foreground">
              Risk Level
            </p>
            <span
              className={`mt-2 inline-block rounded-full border px-3 py-1 text-sm font-semibold ${riskStyles[risk_level] ?? ''}`}
            >
              {risk_level}
            </span>
          </div>

          {/* Decision */}
          <div className="text-center">
            <p className="text-sm font-medium text-muted-foreground">
              Decision
            </p>
            <span
              className={`mt-2 inline-block rounded-full border px-3 py-1 text-sm font-semibold ${decisionStyles[decision] ?? ''}`}
            >
              {decision}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default FraudSummary
