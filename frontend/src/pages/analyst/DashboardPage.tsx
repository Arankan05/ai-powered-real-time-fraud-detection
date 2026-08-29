import { useCallback, useEffect, useState } from 'react'
import { AxiosError } from 'axios'
import {
  Activity,
  ShieldAlert,
  AlertTriangle,
  CheckCircle,
} from 'lucide-react'
import type { DashboardResponse } from '@/types/dashboard'
import * as analyticsApi from '@/services/api/analyticsApi'
import StatCard from '@/components/analyst/StatCard'
import RiskDistributionChart from '@/components/analyst/RiskDistributionChart'
import TransactionsOverTimeChart from '@/components/analyst/TransactionsOverTimeChart'
import TopRiskFactorsChart from '@/components/analyst/TopRiskFactorsChart'
import { Button } from '@/components/ui/button'

function DashboardPage() {
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const fetchDashboard = useCallback(async () => {
    setIsLoading(true)
    setErrorMsg(null)
    try {
      const data = await analyticsApi.getDashboard()
      setDashboard(data)
    } catch (err) {
      if (err instanceof AxiosError) {
        if (err.response?.status === 403) {
          setErrorMsg(
            'You are not authorized to view the dashboard.',
          )
          return
        }
        setErrorMsg(
          'Unable to load dashboard data. Please try again.',
        )
        return
      }
      setErrorMsg(
        'Unable to connect to the server. Please check your connection.',
      )
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchDashboard()
  }, [fetchDashboard])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">
          Analyst Dashboard
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Overview of fraud-detection activity and transaction metrics.
        </p>
      </div>

      {/* Loading */}
      {isLoading && (
        <p className="py-12 text-center text-sm text-muted-foreground">
          Loading dashboard&hellip;
        </p>
      )}

      {/* Error */}
      {errorMsg && (
        <div className="space-y-4">
          <div
            className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
            role="alert"
          >
            {errorMsg}
          </div>
          <Button variant="outline" size="sm" onClick={fetchDashboard}>
            Retry
          </Button>
        </div>
      )}

      {/* Statistics + Risk Distribution */}
      {dashboard && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              title="Total Transactions"
              value={dashboard.total_transactions}
              icon={Activity}
            />
            <StatCard
              title="Flagged Transactions"
              value={dashboard.flagged_transactions}
              icon={ShieldAlert}
            />
            <StatCard
              title="Open Alerts"
              value={dashboard.alerts_open}
              icon={AlertTriangle}
            />
            <StatCard
              title="Resolved Alerts"
              value={dashboard.alerts_resolved}
              icon={CheckCircle}
            />
          </div>

          {/* Risk Distribution */}
          <RiskDistributionChart
            riskDistribution={dashboard.risk_distribution}
          />

          {/* Transactions Over Time */}
          <TransactionsOverTimeChart
            transactions={dashboard.transactions_over_time}
          />

          {/* Top Risk Factors */}
          <TopRiskFactorsChart
            riskFactors={dashboard.top_risk_factors}
          />
        </>
      )}
    </div>
  )
}

export default DashboardPage
