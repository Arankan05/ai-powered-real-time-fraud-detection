import type { TransactionExplanation } from '@/types/transaction'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'

interface FraudExplanationProps {
  explanation?: TransactionExplanation | null
}

function FraudExplanation({ explanation }: FraudExplanationProps) {
  const ml_top_factors = explanation?.ml_top_factors ?? []
  const behaviour_signals = explanation?.behaviour_signals ?? []
  const rules_triggered = explanation?.rules_triggered ?? []
  const hasML = ml_top_factors.length > 0
  const hasBehaviour = behaviour_signals.length > 0
  const hasRules = rules_triggered.length > 0

  return (
    <Card>
      <CardHeader>
        <CardTitle>Explanation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* ML Top Factors */}
        <section>
          <h3 className="mb-2 text-sm font-medium text-foreground">
            ML Top Factors
          </h3>
          {hasML ? (
            <ul className="space-y-1">
              {ml_top_factors.map((factor) => (
                <li
                  key={factor.feature}
                  className="flex items-center justify-between text-sm"
                >
                  <span className="text-muted-foreground">
                    {factor.feature}
                  </span>
                  <span className="font-medium tabular-nums">
                    {(factor.importance * 100).toFixed(0)}%
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">
              No ML factors available
            </p>
          )}
        </section>

        {/* Behaviour Signals */}
        <section>
          <h3 className="mb-2 text-sm font-medium text-foreground">
            Behaviour Signals
          </h3>
          {hasBehaviour ? (
            <ul className="space-y-1">
              {behaviour_signals.map((signal) => (
                <li
                  key={signal.signal}
                  className="flex items-center justify-between text-sm"
                >
                  <span className="text-muted-foreground">
                    {signal.signal}
                  </span>
                  <span className="font-medium tabular-nums">
                    {(signal.severity * 100).toFixed(0)}%
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">
              No behaviour signals available
            </p>
          )}
        </section>

        {/* Rules Triggered */}
        <section>
          <h3 className="mb-2 text-sm font-medium text-foreground">
            Rules Triggered
          </h3>
          {hasRules ? (
            <ul className="space-y-1">
              {rules_triggered.map((rule) => (
                <li
                  key={rule.rule}
                  className="flex items-center justify-between text-sm"
                >
                  <span className="text-muted-foreground">{rule.rule}</span>
                  <span className="font-medium tabular-nums">
                    +{rule.contribution}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">
              No rules triggered
            </p>
          )}
        </section>
      </CardContent>
    </Card>
  )
}

export default FraudExplanation
