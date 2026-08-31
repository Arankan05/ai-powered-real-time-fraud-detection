import type { RiskDistribution } from '@/types/dashboard'
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
  Cell,
} from 'recharts'

interface RiskDistributionChartProps {
  riskDistribution: RiskDistribution
}

const RISK_DATA = [
  { name: 'LOW', fill: '#10b981' },
  { name: 'MEDIUM', fill: '#f59e0b' },
  { name: 'HIGH', fill: '#ef4444' },
] as const

function RiskDistributionChart({
  riskDistribution,
}: RiskDistributionChartProps) {
  const data = RISK_DATA.map((item) => ({
    ...item,
    count: riskDistribution[item.name],
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle>Risk Distribution</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" name="Transactions" radius={[4, 4, 0, 0]}>
                {data.map((entry) => (
                  <Cell key={entry.name} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Text legend — accessible, not color-dependent */}
        <div className="flex flex-wrap gap-4 text-sm">
          {data.map((item) => (
            <div key={item.name} className="flex items-center gap-2">
              <span
                className="inline-block h-3 w-3 rounded-sm"
                style={{ backgroundColor: item.fill }}
                aria-hidden="true"
              />
              <span className="font-medium">{item.name}:</span>
              <span className="tabular-nums text-muted-foreground">
                {new Intl.NumberFormat().format(item.count)}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export default RiskDistributionChart
