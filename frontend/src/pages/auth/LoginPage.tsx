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

const loginSchema = z.object({
  email: z.string().min(1, 'Email is required').email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
})

type LoginFormValues = z.infer<typeof loginSchema>

function isApiError(err: unknown): err is ApiError {
  return (
    typeof err === 'object' &&
    err !== null &&
    'detail' in err &&
    'error_code' in err
  )
}

function LoginPage() {
  const { login, isLoading } = useContext(AuthContext)!
  const navigate = useNavigate()
  const [serverError, setServerError] = useState<string | null>(null)

  const form = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: '',
      password: '',
    },
  })

  const onSubmit = form.handleSubmit(async (data) => {
    setServerError(null)
    try {
      const user = await login(data)
      switch (user.role) {
        case 'customer':
          navigate('/customer')
          break
        case 'fraud_analyst':
        case 'admin':
          navigate('/analyst/dashboard')
          break
        default:
          setServerError('Unable to determine your dashboard destination.')
      }
    } catch (err) {
      if (isApiError(err)) {
        if (err.error_code === 'INACTIVE_ACCOUNT') {
          setServerError(
            'Your account is inactive. Please contact support.',
          )
        } else {
          setServerError(err.detail)
        }
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
          <CardTitle>Sign in</CardTitle>
          <CardDescription>
            Enter your credentials to access your account
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

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
                aria-invalid={!!form.formState.errors.email}
                aria-describedby={
                  form.formState.errors.email ? 'email-error' : undefined
                }
                {...form.register('email')}
              />
              {form.formState.errors.email && (
                <p id="email-error" className="text-sm text-destructive">
                  {form.formState.errors.email.message}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                aria-invalid={!!form.formState.errors.password}
                aria-describedby={
                  form.formState.errors.password ? 'password-error' : undefined
                }
                {...form.register('password')}
              />
              {form.formState.errors.password && (
                <p id="password-error" className="text-sm text-destructive">
                  {form.formState.errors.password.message}
                </p>
              )}
            </div>

            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? 'Signing in…' : 'Sign in'}
            </Button>
          </form>
        </CardContent>
      </Card>

      <p className="text-center text-sm text-muted-foreground">
        Don&apos;t have an account?{' '}
        <Link to="/register" className="font-medium text-primary hover:underline">
          Register
        </Link>
      </p>
    </>
  )
}

export default LoginPage
