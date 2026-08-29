import { useParams } from 'react-router-dom'

function TransactionDetailPage() {
  const { id } = useParams<{ id: string }>()

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-foreground">
        Transaction Detail
      </h1>
      <p className="text-muted-foreground">
        Fraud analysis results for transaction {id}.
      </p>
    </div>
  )
}

export default TransactionDetailPage
