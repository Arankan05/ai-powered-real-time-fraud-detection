import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'

interface RiskFactorsProps {
  factors: string[]
}

function RiskFactors({ factors }: RiskFactorsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Risk Factors</CardTitle>
      </CardHeader>
      <CardContent>
        {factors.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {factors.map((factor) => (
              <span
                key={factor}
                className="rounded-full border bg-muted px-2.5 py-0.5 text-xs font-medium text-foreground"
              >
                {factor}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            No risk factors identified
          </p>
        )}
      </CardContent>
    </Card>
  )
}

export default RiskFactors
