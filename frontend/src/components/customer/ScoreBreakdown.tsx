import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'

interface ScoreBreakdownProps {
  mlScore: number
  behaviourScore: number
  ruleScore: number
  riskScore: number
}

interface ScoreBarProps {
  label: string
  score: number
}

function ScoreBar({ label, score }: ScoreBarProps) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium tabular-nums">{score}</span>
      </div>
      <div
        className="h-2 w-full overflow-hidden rounded-full bg-muted"
        role="progressbar"
        aria-valuenow={score}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label}: ${score} out of 100`}
      >
        <div
          className="h-full rounded-full bg-primary transition-all"
          style={{ width: `${Math.min(Math.max(score, 0), 100)}%` }}
        />
      </div>
    </div>
  )
}

function ScoreBreakdown(props: ScoreBreakdownProps) {
  const { mlScore, behaviourScore, ruleScore, riskScore } = props

  return (
    <Card>
      <CardHeader>
        <CardTitle>Score Breakdown</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <ScoreBar label="ML Score" score={mlScore} />
        <ScoreBar label="Behaviour Score" score={behaviourScore} />
        <ScoreBar label="Rule Score" score={ruleScore} />
        <div className="border-t pt-4">
          <ScoreBar label="Overall Risk Score" score={riskScore} />
        </div>
      </CardContent>
    </Card>
  )
}

export default ScoreBreakdown
