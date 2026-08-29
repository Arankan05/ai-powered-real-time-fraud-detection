import { useContext } from 'react'
import { Navigate } from 'react-router-dom'
import type { ReactNode } from 'react'
import { AuthContext } from '@/context/AuthContext'

interface ProtectedRouteProps {
  children: ReactNode
}

/**
 * Reusable protected-route wrapper.
 * - While auth is initializing, shows a loading indicator.
 * - If the user is not authenticated, redirects to /login.
 * - Otherwise renders children.
 *
 * Role-specific guards will be added in a later step.
 */
function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { isAuthenticated, isInitializing } = useContext(AuthContext)!

  if (isInitializing) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-muted-foreground">Loading…</p>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}

export default ProtectedRoute
