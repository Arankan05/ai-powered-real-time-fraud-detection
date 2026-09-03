import { useContext, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import { z } from 'zod'
import { zodResolver } from '@hookform/resolvers/zod'
import { AuthContext } from '@/context/AuthContext'
import type { ApiError } from '@/types/auth'
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

const registerSchema = z
  .object({
    first_name: z.string().min(1, 'First name is required'),
    last_name: z.string().min(1, 'Last name is required'),
    email: z
      .string()
      .min(1, 'Email is required')
      .email('Invalid email address'),
    phone: z.string().min(1, 'Phone number is required'),
    date_of_birth: z.string().min(1, 'Date of birth is required'),
    address: z.string().min(1, 'Address is required'),
    password: z
      .string()
      .min(8, 'Password must be at least 8 characters')
      .regex(/[A-Z]/, 'Must contain at least one uppercase letter')
      .regex(/[a-z]/, 'Must contain at least one lowercase letter')
      .regex(/[0-9]/, 'Must contain at least one digit'),
    confirmPassword: z.string().min(1, 'Please confirm your password'),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: 'Passwords do not match',
    path: ['confirmPassword'],
  })

type RegisterFormValues = z.input<typeof registerSchema>

function isApiError(err: unknown): err is ApiError {
  return (
    typeof err === 'object' &&
    err !== null &&
    'detail' in err &&
    'error_code' in err
  )
}

function RegisterPage() {
  const { register: registerUser, isLoading } = useContext(AuthContext)!
  const navigate = useNavigate()
  const [serverError, setServerError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  const form = useForm<RegisterFormValues>({
    resolver: zodResolver(registerSchema),
    defaultValues: {
      first_name: '',
      last_name: '',
      email: '',
      phone: '',
      date_of_birth: '',
      address: '',
      password: '',
      confirmPassword: '',
    },
  })

  const onSubmit = form.handleSubmit(async (data) => {
    setServerError(null)
    setSuccessMessage(null)

    // Strip confirmPassword — not part of the API contract
    const { confirmPassword: _confirmPassword, ...payload } = data

    try {
      await registerUser(payload)
      setSuccessMessage('Registration successful. Please sign in.')
      setTimeout(() => navigate('/login'), 2000)
    } catch (err) {
      if (isApiError(err)) {
        setServerError(err.detail)
      } else if (err instanceof Error) {
        setServerError(err.message)
      } else {
        setServerError('An unexpected error occurred. Please try again.')
      }
    }
  })

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Create an account</CardTitle>
          <CardDescription>
            Register as a new customer to get started
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-4" noValidate>
            {serverError && (
              <div
                className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
                role="alert"
              >
                {serverError}
              </div>
            )}
            {successMessage && (
              <div
                className="rounded-lg border border-emerald-500/50 bg-emerald-500/10 p-3 text-sm text-emerald-700"
                role="status"
              >
                {successMessage}
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="first_name">First name</Label>
                <Input
                  id="first_name"
                  autoComplete="given-name"
                  aria-invalid={!!form.formState.errors.first_name}
                  {...form.register('first_name')}
                />
                {form.formState.errors.first_name && (
                  <p className="text-sm text-destructive">
                    {form.formState.errors.first_name.message}
                  </p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="last_name">Last name</Label>
                <Input
                  id="last_name"
                  autoComplete="family-name"
                  aria-invalid={!!form.formState.errors.last_name}
                  {...form.register('last_name')}
                />
                {form.formState.errors.last_name && (
                  <p className="text-sm text-destructive">
                    {form.formState.errors.last_name.message}
                  </p>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
                aria-invalid={!!form.formState.errors.email}
                {...form.register('email')}
              />
              {form.formState.errors.email && (
                <p className="text-sm text-destructive">
                  {form.formState.errors.email.message}
                </p>
              )}
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="phone">Phone</Label>
                <Input
                  id="phone"
                  type="tel"
                  autoComplete="tel"
                  placeholder="+1234567890"
                  aria-invalid={!!form.formState.errors.phone}
                  {...form.register('phone')}
                />
                {form.formState.errors.phone && (
                  <p className="text-sm text-destructive">
                    {form.formState.errors.phone.message}
                  </p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="date_of_birth">Date of birth</Label>
                <Input
                  id="date_of_birth"
                  type="date"
                  autoComplete="bday"
                  aria-invalid={!!form.formState.errors.date_of_birth}
                  {...form.register('date_of_birth')}
                />
                {form.formState.errors.date_of_birth && (
                  <p className="text-sm text-destructive">
                    {form.formState.errors.date_of_birth.message}
                  </p>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="address">Address</Label>
              <Input
                id="address"
                autoComplete="street-address"
                aria-invalid={!!form.formState.errors.address}
                {...form.register('address')}
              />
              {form.formState.errors.address && (
                <p className="text-sm text-destructive">
                  {form.formState.errors.address.message}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="new-password"
                aria-invalid={!!form.formState.errors.password}
                aria-describedby="password-requirements"
                {...form.register('password')}
              />
              <p
                id="password-requirements"
                className="text-xs text-muted-foreground"
              >
                Min 8 characters, one uppercase, one lowercase, one digit
              </p>
              {form.formState.errors.password && (
                <p className="text-sm text-destructive">
                  {form.formState.errors.password.message}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="confirmPassword">Confirm password</Label>
              <Input
                id="confirmPassword"
                type="password"
                autoComplete="new-password"
                aria-invalid={!!form.formState.errors.confirmPassword}
                {...form.register('confirmPassword')}
              />
              {form.formState.errors.confirmPassword && (
                <p className="text-sm text-destructive">
                  {form.formState.errors.confirmPassword.message}
                </p>
              )}
            </div>

            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? 'Creating account…' : 'Create account'}
            </Button>
          </form>
        </CardContent>
      </Card>

      <p className="text-center text-sm text-muted-foreground">
        Already have an account?{' '}
        <Link to="/login" className="font-medium text-primary hover:underline">
          Sign in
        </Link>
      </p>
    </>
  )
}

export default RegisterPage
