import { useParams } from 'react-router-dom'

function AlertDetailPage() {
  const { id } = useParams<{ id: string }>()

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-foreground">
        Alert Detail
      </h1>
      <p className="text-muted-foreground">
        Investigate and manage alert {id}.
      </p>
    </div>
  )
}

export default AlertDetailPage
