import { useState } from 'react'
import { AxiosError } from 'axios'
import TransactionForm from '@/components/customer/TransactionForm'
import type { TransactionFormValues } from '@/components/customer/TransactionForm'
import type { TransactionResponse } from '@/types/transaction'
import type { ApiError } from '@/types/auth'
import * as transactionApi from '@/services/api/transactionApi'

function BankingPage() {
  const [transaction, setTransaction] = useState<TransactionResponse | null>(
    null,
  )
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const handleCreateTransaction = async (values: TransactionFormValues) => {
    setErrorMsg(null)
    setTransaction(null)
    try {
      const result = await transactionApi.createTransaction(values)
      setTransaction(result)
    } catch (err) {
      if (err instanceof AxiosError) {
        const data = err.response?.data as ApiError | undefined
        if (err.response?.status === 403) {
          throw new Error(
            'You are not authorized to submit transactions.',
          )
        }
        throw new Error(
          data?.detail ?? 'Failed to submit transaction. Please try again.',
        )
      }
      throw new Error(
        'Unable to connect to the server. Please check your connection.',
      )
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">Banking</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Submit transaction details for fraud detection review.
        </p>
      </div>

      {/* Error from previous attempt */}
      {errorMsg && (
        <div
          className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
          role="alert"
        >
          {errorMsg}
        </div>
      )}

      {/* Minimal truthful confirmation from actual backend response */}
      {transaction && (
        <div
          className="rounded-lg border border-emerald-500/50 bg-emerald-500/10 p-4 text-sm text-emerald-700"
          role="status"
        >
          <p className="font-medium">Transaction submitted successfully</p>
          <p className="mt-1 text-xs">
            ID: {transaction.id} &middot; Status: {transaction.status}
          </p>
        </div>
      )}

      <TransactionForm
        onSubmit={handleCreateTransaction}
        onError={setErrorMsg}
        onSuccess={() => {
          setErrorMsg(null)
          setTransaction(null)
        }}
      />
    </div>
  )
}

export default BankingPage
