import TransactionForm from '@/components/customer/TransactionForm'
import type { TransactionFormValues } from '@/components/customer/TransactionForm'

function BankingPage() {
  const handleSubmit = async (_values: TransactionFormValues) => {
    // API integration will be connected in a later step.
    // The form has already been validated client-side.
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">
          Banking
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Submit transaction details for fraud detection review.
        </p>
      </div>
      <TransactionForm onSubmit={handleSubmit} />
    </div>
  )
}

export default BankingPage
