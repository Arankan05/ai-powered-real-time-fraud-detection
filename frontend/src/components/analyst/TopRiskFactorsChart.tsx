import type { TopRiskFactor } from '@/types/dashboard'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

interface TopRiskFactorsChartProps {
  riskFactors: TopRiskFactor[]
}

function TopRiskFactorsChart({ riskFactors }: TopRiskFactorsChartProps) {
  if (riskFactors.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Top Risk Factors</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="py-8 text-center text-sm text-muted-foreground">
            No top risk factors available.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Top Risk Factors</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={riskFactors}
              layout="vertical"
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" allowDecimals={false} />
              <YAxis
                type="category"
                dataKey="factor"
                width={140}
                tick={{ fontSize: 12 }}
              />
              <Tooltip />
              <Bar
                dataKey="count"
                name="Count"
                fill="#8b5cf6"
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}

export default TopRiskFactorsChart
