import type { TransactionSummaryItem } from '@/types/transaction'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'

interface RecentTransactionsProps {
  transactions: TransactionSummaryItem[]
}

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

function truncateId(id: string): string {
  return id.length > 8 ? `${id.slice(0, 8)}…` : id
}

function RecentTransactions({ transactions }: RecentTransactionsProps) {
  if (transactions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Recent Transactions</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="py-8 text-center text-sm text-muted-foreground">
            No recent transactions available.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Transactions</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Desktop table */}
        <div className="hidden overflow-x-auto md:block">
          <table className="w-full text-left text-sm">
            <thead className="border-b bg-muted/50">
              <tr>
                <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                  Merchant
                </th>
                <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                  Customer
                </th>
                <th scope="col" className="px-4 py-3 text-right font-medium text-muted-foreground">
                  Amount
                </th>
                <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                  Type
                </th>
                <th scope="col" className="px-4 py-3 font-medium text-muted-foreground">
                  Date
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
              </tr>
            </thead>
            <tbody className="divide-y">
              {transactions.map((tx) => (
                <tr key={tx.id} className="transition-colors hover:bg-muted/30">
                  <td className="px-4 py-3 font-medium">{tx.merchant_name}</td>
                  <td className="px-4 py-3 text-muted-foreground">
                    <span title={tx.customer_id}>{truncateId(tx.customer_id)}</span>
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums">
                    {formatAmount(tx.amount, tx.currency)}
                  </td>
                  <td className="px-4 py-3 capitalize text-muted-foreground">
                    {tx.transaction_type}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-muted-foreground">
                    {formatDate(tx.timestamp)}
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
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Mobile cards */}
        <div className="space-y-3 md:hidden">
          {transactions.map((tx) => (
            <div
              key={tx.id}
              className="space-y-2 rounded-lg border p-4"
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
              <div className="flex flex-wrap items-center gap-2">
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
              <p className="text-xs text-muted-foreground">
                Customer: <span title={tx.customer_id}>{truncateId(tx.customer_id)}</span>
              </p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export default RecentTransactions
