import type { TransactionResponse } from '@/types/transaction'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'

interface TransactionInfoProps {
  transaction: TransactionResponse
}

interface DetailRowProps {
  label: string
  value: string
}

function DetailRow({ label, value }: DetailRowProps) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm font-medium text-foreground">{value}</span>
    </div>
  )
}

function TransactionInfo({ transaction }: TransactionInfoProps) {
  const {
    id,
    amount,
    currency,
    transaction_type,
    merchant_name,
    merchant_category,
    location_country,
    location_city,
    device_type,
    timestamp,
    status,
  } = transaction

  const formattedDate = new Date(timestamp).toLocaleString()
  const location = location_city
    ? `${location_city}, ${location_country}`
    : location_country

  return (
    <Card>
      <CardHeader>
        <CardTitle>Transaction Details</CardTitle>
      </CardHeader>
      <CardContent className="divide-y">
        <DetailRow label="Transaction ID" value={id} />
        <DetailRow
          label="Amount"
          value={`${amount.toFixed(2)} ${currency}`}
        />
        <DetailRow label="Type" value={transaction_type} />
        <DetailRow label="Merchant" value={merchant_name} />
        <DetailRow label="Category" value={merchant_category} />
        <DetailRow label="Location" value={location} />
        <DetailRow label="Device" value={device_type} />
        <DetailRow label="Timestamp" value={formattedDate} />
        <DetailRow label="Status" value={status} />
      </CardContent>
    </Card>
  )
}

export default TransactionInfo
