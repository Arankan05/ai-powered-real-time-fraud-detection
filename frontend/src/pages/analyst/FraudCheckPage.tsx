import { Link } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { buttonVariants } from '@/components/ui/button'

function FraudCheckPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">
          Real-Time Fraud Analysis
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          System operational overview and transaction evaluation process.
        </p>
      </div>

      <Card className="max-w-2xl">
        <CardHeader>
          <CardTitle className="text-lg">Automated Real-Time Fraud Scoring</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-muted-foreground">
          <p>
            Fraud analysis is automatically executed in real time whenever a transaction is submitted through the payment processing pipeline.
          </p>
          <p>
            Every submission is evaluated against the production XGBoost ML model, behavioural anomaly detectors, and rule-based controls. High-risk transactions assigned a <strong className="text-foreground">HOLD</strong> decision automatically generate alerts for analyst review.
          </p>
          <div className="flex flex-wrap gap-3 pt-2">
            <Link to="/analyst/alerts" className={buttonVariants({ size: 'sm' })}>
              View Active Alerts
            </Link>
            <Link to="/analyst" className={buttonVariants({ variant: 'outline', size: 'sm' })}>
              Go to Dashboard
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default FraudCheckPage
