import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AxiosError } from 'axios'
import * as transactionApi from '@/services/api/transactionApi'
import type {
  TransactionSummaryItem,
  TransactionListResponse,
  TransactionStatus,
  RiskLevel,
} from '@/types/transaction'
import { Button } from '@/components/ui/button'

const riskStyles: Record<string, string> = {
  LOW: 'bg-emerald-500/10 text-emerald-700',
  MEDIUM: 'bg-amber-500/10 text-amber-700',
  HIGH: 'bg-red-500/10 text-red-700',
}

const decisionStyles: Record<string, string> = {
  APPROVE: 'bg-emerald-500/10 text-emerald-700',
  VERIFY: 'bg-amber-500/10 text-amber-700',
  HOLD: 'bg-red-500/10 text-red-700',
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: 'short',
    timeStyle: 'short',
  })
}

function formatAmount(amount: number, currency: string): string {
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency,
  }).format(amount)
}

function TransactionHistoryPage() {
  const navigate = useNavigate()
  const [transactions, setTransactions] = useState<TransactionSummaryItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [perPage] = useState(20)
  const [statusFilter, setStatusFilter] = useState<TransactionStatus | ''>('')
  const [riskFilter, setRiskFilter] = useState<RiskLevel | ''>('')
  const [isLoading, setIsLoading] = useState(true)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const totalPages = Math.max(1, Math.ceil(total / perPage))

  const fetchTransactions = useCallback(async () => {
    setIsLoading(true)
    setErrorMsg(null)
    try {
      const params: Record<string, string | number> = { page, per_page: perPage }
      if (statusFilter) params.status = statusFilter
      if (riskFilter) params.risk_level = riskFilter

      const result: TransactionListResponse =
        await transactionApi.getTransactions(params)
      setTransactions(result.items)
      setTotal(result.total)
    } catch (err) {
      if (err instanceof AxiosError) {
        if (err.response?.status === 403) {
          setErrorMsg('You are not authorized to view transaction history.')
          return
        }
        setErrorMsg(
          'Failed to load transaction history. Please try again.',
        )
        return
      }
      setErrorMsg(
        'Unable to connect to the server. Please check your connection.',
      )
    } finally {
      setIsLoading(false)
    }
  }, [page, perPage, statusFilter, riskFilter])

  useEffect(() => {
    void fetchTransactions()
  }, [fetchTransactions])

  const handleStatusChange = (value: string) => {
    setStatusFilter(value as TransactionStatus | '')
    setPage(1)
  }

  const handleRiskChange = (value: string) => {
    setRiskFilter(value as RiskLevel | '')
    setPage(1)
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">
            Transaction History
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            View your past transactions and fraud analysis results.
          </p>
        </div>
        <Link to="/customer">
          <Button variant="outline" size="sm">
            New Transaction
          </Button>
        </Link>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <label className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Status</span>
          <select
            value={statusFilter}
            onChange={(e) => handleStatusChange(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm"
          >
            <option value="">All</option>
            <option value="PENDING">Pending</option>
            <option value="COMPLETED">Completed</option>
            <option value="FAILED">Failed</option>
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Risk Level</span>
          <select
            value={riskFilter}
            onChange={(e) => handleRiskChange(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm"
          >
            <option value="">All</option>
            <option value="LOW">Low</option>
            <option value="MEDIUM">Medium</option>
            <option value="HIGH">High</option>
          </select>
        </label>
      </div>

      {/* Error */}
      {errorMsg && (
        <div
          className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
          role="alert"
        >
          {errorMsg}
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <p className="py-12 text-center text-sm text-muted-foreground">
          Loading transactions&hellip;
        </p>
      )}

      {/* Empty */}
      {!isLoading && !errorMsg && transactions.length === 0 && (
        <div className="flex flex-col items-center gap-4 py-12 text-center">
          <p className="text-muted-foreground">No transactions yet.</p>
          <Link to="/customer">
            <Button variant="outline">Create Transaction</Button>
          </Link>
        </div>
      )}

      {/* Table (desktop) */}
      {!isLoading && transactions.length > 0 && (
        <>
          <div className="hidden overflow-x-auto rounded-lg border md:block">
            <table className="w-full text-left text-sm">
              <thead className="border-b bg-muted/50">
                <tr>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Date
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Merchant
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Type
                  </th>
                  <th scope="col" className="px-4 py-3 text-right font-medium text-muted-foreground">
                    Amount
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Status
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Risk
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                    Decision
                  </th>
                  <th scope="col" className="px-4 py-3">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {transactions.map((tx) => (
                  <tr
                    key={tx.id}
                    className="transition-colors hover:bg-muted/30"
                  >
                    <td className="whitespace-nowrap px-4 py-3 text-muted-foreground">
                      {formatDate(tx.timestamp)}
                    </td>
                    <td className="px-4 py-3 font-medium">{tx.merchant_name}</td>
                    <td className="px-4 py-3 capitalize text-muted-foreground">
                      {tx.transaction_type}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums">
                      {formatAmount(tx.amount, tx.currency)}
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-sm">{tx.status}</span>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${riskStyles[tx.risk_level] ?? ''}`}
                      >
                        {tx.risk_level}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${decisionStyles[tx.decision] ?? ''}`}
                      >
                        {tx.decision}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        type="button"
                        onClick={() =>
                          navigate(`/customer/transactions/${tx.id}`)
                        }
                        className="text-sm font-medium text-primary underline-offset-2 hover:underline"
                      >
                        Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Cards (mobile / tablet) */}
          <div className="space-y-3 md:hidden">
            {transactions.map((tx) => (
              <div
                key={tx.id}
                className="space-y-3 rounded-lg border p-4"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="font-medium">{tx.merchant_name}</p>
                    <p className="text-xs text-muted-foreground">
                      {formatDate(tx.timestamp)}
                    </p>
                  </div>
                  <p className="whitespace-nowrap text-right font-medium tabular-nums">
                    {formatAmount(tx.amount, tx.currency)}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <span className="capitalize text-xs text-muted-foreground">
                    {tx.transaction_type}
                  </span>
                  <span className="text-xs text-muted-foreground">&middot;</span>
                  <span className="text-xs">{tx.status}</span>
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${riskStyles[tx.risk_level] ?? ''}`}
                  >
                    {tx.risk_level}
                  </span>
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${decisionStyles[tx.decision] ?? ''}`}
                  >
                    {tx.decision}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() =>
                    navigate(`/customer/transactions/${tx.id}`)
                  }
                  className="text-sm font-medium text-primary underline-offset-2 hover:underline"
                >
                  View Details
                </button>
              </div>
            ))}
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Showing {(page - 1) * perPage + 1}&ndash;
              {Math.min(page * perPage, total)} of {total}
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default TransactionHistoryPage
