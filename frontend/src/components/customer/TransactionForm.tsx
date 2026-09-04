import { useMemo, useState } from 'react'
import { useForm } from 'react-hook-form'
import { z } from 'zod'
import { zodResolver } from '@hookform/resolvers/zod'
import type { TransactionType, DeviceType } from '@/types/transaction'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'

// ── Zod schema ──────────────────────────────────────────────────────
const transactionSchema = z.object({
  amount: z
    .number('Enter a valid amount')
    .min(0.01, 'Amount must be at least 0.01')
    .max(9999999999.99, 'Amount exceeds maximum'),
  currency: z
    .string()
    .min(1, 'Currency is required')
    .length(3, 'Currency must be a 3-letter ISO code')
    .toUpperCase(),
  merchant_name: z
    .string()
    .min(1, 'Merchant name is required')
    .max(255, 'Merchant name is too long'),
  merchant_category: z
    .string()
    .min(1, 'Merchant category is required')
    .max(10, 'Category code is too long'),
  transaction_type: z.enum(
    ['purchase', 'transfer', 'withdrawal'] as const,
    { message: 'Select a valid transaction type' },
  ),
  location_country: z
    .string()
    .min(1, 'Country is required')
    .max(100, 'Country is too long'),
  location_city: z
    .string()
    .max(100, 'City name is too long'),
  device_fingerprint: z
    .string()
    .min(1, 'Device fingerprint is required')
    .max(255, 'Fingerprint is too long'),
  device_type: z.enum(
    ['mobile', 'desktop', 'pos'] as const,
    { message: 'Select a valid device type' },
  ),
  ip_address: z
    .string()
    .min(1, 'IP address is required')
    .max(45, 'IP address is too long'),
})

export type TransactionFormValues = z.infer<typeof transactionSchema>

export interface TransactionFormProps {
  onSubmit: (values: TransactionFormValues) => void | Promise<void>
  onSuccess?: () => void
  onError?: (message: string) => void
  disabled?: boolean
}

const TRANSACTION_TYPES: { value: TransactionType; label: string }[] = [
  { value: 'purchase', label: 'Purchase' },
  { value: 'transfer', label: 'Transfer' },
  { value: 'withdrawal', label: 'Withdrawal' },
]

const DEVICE_TYPES: { value: DeviceType; label: string }[] = [
  { value: 'mobile', label: 'Mobile' },
  { value: 'desktop', label: 'Desktop' },
  { value: 'pos', label: 'Point of Sale' },
]

/** Shared Tailwind classes for native <select> to match Input styling */
const selectClasses =
  'h-8 w-full rounded-lg border border-input bg-transparent px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20'

// ── Component ───────────────────────────────────────────────────────

function TransactionForm({
  onSubmit,
  onSuccess,
  onError,
  disabled,
}: TransactionFormProps) {
  const [submitted, setSubmitted] = useState(false)

  // Generate a stable fingerprint for this browser session
  const fingerprint = useMemo(
    () =>
      Array.from(crypto.getRandomValues(new Uint8Array(12)))
        .map((b) => b.toString(16).padStart(2, '0'))
        .join(''),
    [],
  )

  const form = useForm<TransactionFormValues>({
    resolver: zodResolver(transactionSchema),
    defaultValues: {
      amount: undefined,
      currency: 'USD',
      merchant_name: '',
      merchant_category: '',
      transaction_type: 'purchase',
      location_country: '',
      location_city: '',
      device_fingerprint: fingerprint,
      device_type: 'mobile',
      ip_address: '',
    },
  })

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = form

  const handleValidSubmit = handleSubmit(async (data) => {
    try {
      await onSubmit(data)
      setSubmitted(true)
      onSuccess?.()
      reset()
      setTimeout(() => setSubmitted(false), 5000)
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'An unexpected error occurred. Please try again.'
      onError?.(message)
    }
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>New Transaction</CardTitle>
        <CardDescription>
          Enter the transaction details below. The information will be reviewed
          by the fraud detection system after submission.
        </CardDescription>
      </CardHeader>

      <CardContent>
        {submitted && (
          <div
            className="mb-4 rounded-lg border border-emerald-500/50 bg-emerald-500/10 p-3 text-sm text-emerald-700"
            role="status"
          >
            Transaction details accepted and submitted for review.
          </div>
        )}

        <form onSubmit={handleValidSubmit} className="space-y-6" noValidate>
          {/* ── Amount & Currency ─────────────────────────────────── */}
          <fieldset className="space-y-3">
            <legend className="text-sm font-medium text-foreground">
              Amount
            </legend>
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-2 space-y-2">
                <Label htmlFor="amount">Amount</Label>
                <Input
                  id="amount"
                  type="number"
                  inputMode="decimal"
                  step="0.01"
                  min="0.01"
                  autoComplete="off"
                  aria-invalid={!!errors.amount}
                  aria-describedby={errors.amount ? 'amount-error' : undefined}
                  {...register('amount', { valueAsNumber: true })}
                />
                {errors.amount && (
                  <p id="amount-error" className="text-sm text-destructive">
                    {errors.amount.message}
                  </p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="currency">Currency</Label>
                <Input
                  id="currency"
                  maxLength={3}
                  autoComplete="off"
                  aria-invalid={!!errors.currency}
                  {...register('currency')}
                />
                {errors.currency && (
                  <p className="text-sm text-destructive">
                    {errors.currency.message}
                  </p>
                )}
              </div>
            </div>
          </fieldset>

          {/* ── Transaction Type ──────────────────────────────────── */}
          <fieldset className="space-y-3">
            <legend className="text-sm font-medium text-foreground">
              Transaction Type
            </legend>
            <div className="flex flex-wrap gap-4">
              {TRANSACTION_TYPES.map((type) => (
                <label key={type.value} className="flex items-center gap-2">
                  <input
                    type="radio"
                    value={type.value}
                    className="accent-primary"
                    {...register('transaction_type')}
                  />
                  <span className="text-sm">{type.label}</span>
                </label>
              ))}
            </div>
            {errors.transaction_type && (
              <p className="text-sm text-destructive">
                {errors.transaction_type.message}
              </p>
            )}
          </fieldset>

          {/* ── Merchant ──────────────────────────────────────────── */}
          <fieldset className="space-y-3">
            <legend className="text-sm font-medium text-foreground">
              Merchant Information
            </legend>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="merchant_name">Merchant name</Label>
                <Input
                  id="merchant_name"
                  autoComplete="off"
                  aria-invalid={!!errors.merchant_name}
                  {...register('merchant_name')}
                />
                {errors.merchant_name && (
                  <p className="text-sm text-destructive">
                    {errors.merchant_name.message}
                  </p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="merchant_category">Merchant category</Label>
                <Input
                  id="merchant_category"
                  maxLength={10}
                  autoComplete="off"
                  placeholder="e.g. 5732"
                  aria-invalid={!!errors.merchant_category}
                  {...register('merchant_category')}
                />
                {errors.merchant_category && (
                  <p className="text-sm text-destructive">
                    {errors.merchant_category.message}
                  </p>
                )}
              </div>
            </div>
          </fieldset>

          {/* ── Location ──────────────────────────────────────────── */}
          <fieldset className="space-y-3">
            <legend className="text-sm font-medium text-foreground">
              Location
            </legend>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="location_country">Country</Label>
                <Input
                  id="location_country"
                  autoComplete="country-name"
                  placeholder="e.g. US"
                  aria-invalid={!!errors.location_country}
                  {...register('location_country')}
                />
                {errors.location_country && (
                  <p className="text-sm text-destructive">
                    {errors.location_country.message}
                  </p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="location_city">City</Label>
                <Input
                  id="location_city"
                  autoComplete="address-level2"
                  aria-invalid={!!errors.location_city}
                  {...register('location_city')}
                />
                {errors.location_city && (
                  <p className="text-sm text-destructive">
                    {errors.location_city.message}
                  </p>
                )}
              </div>
            </div>
          </fieldset>

          {/* ── Device ────────────────────────────────────────────── */}
          <fieldset className="space-y-3">
            <legend className="text-sm font-medium text-foreground">
              Device Information
            </legend>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="device_type">Device type</Label>
                <select
                  id="device_type"
                  className={selectClasses}
                  aria-invalid={!!errors.device_type}
                  {...register('device_type')}
                >
                  {DEVICE_TYPES.map((d) => (
                    <option key={d.value} value={d.value}>
                      {d.label}
                    </option>
                  ))}
                </select>
                {errors.device_type && (
                  <p className="text-sm text-destructive">
                    {errors.device_type.message}
                  </p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="ip_address">IP address</Label>
                <Input
                  id="ip_address"
                  autoComplete="off"
                  placeholder="e.g. 192.168.1.100"
                  aria-invalid={!!errors.ip_address}
                  {...register('ip_address')}
                />
                {errors.ip_address && (
                  <p className="text-sm text-destructive">
                    {errors.ip_address.message}
                  </p>
                )}
              </div>
            </div>
            {/* Read-only fingerprint */}
            <div className="space-y-2">
              <Label htmlFor="device_fingerprint">Device fingerprint</Label>
              <Input
                id="device_fingerprint"
                readOnly
                tabIndex={-1}
                className="bg-muted/50 font-mono text-xs"
                {...register('device_fingerprint')}
              />
              <p className="text-xs text-muted-foreground">
                Auto-generated identifier for this browser session
              </p>
            </div>
          </fieldset>

          {/* ── Submit ────────────────────────────────────────────── */}
          <Button
            type="submit"
            className="w-full"
            disabled={disabled || isSubmitting}
          >
            {isSubmitting ? 'Submitting…' : 'Submit Transaction'}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

export default TransactionForm
