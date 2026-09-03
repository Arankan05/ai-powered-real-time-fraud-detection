import { useContext } from 'react'
import { Navigate } from 'react-router-dom'
import type { ReactNode } from 'react'
import { AuthContext } from '@/context/AuthContext'
import type { UserRole } from '@/types/auth'

interface ProtectedRouteProps {
  children: ReactNode
  /** Optional list of roles permitted to access this route. */
  allowedRoles?: UserRole[]
}

/** Fallback home page per role when access is denied. */
const ROLE_HOME: Record<UserRole, string> = {
  customer: '/customer',
  fraud_analyst: '/analyst/dashboard',
  admin: '/analyst/dashboard',
}

/**
 * Reusable protected-route wrapper.
 * - While auth is initializing, shows a loading indicator.
 * - If the user is not authenticated, redirects to /login.
 * - If allowedRoles is provided and the user's role is not listed,
 *   redirects to the appropriate portal home for their role.
 * - Otherwise renders children.
 */
function ProtectedRoute({ children, allowedRoles }: ProtectedRouteProps) {
  const { user, isAuthenticated, isInitializing } = useContext(AuthContext)!

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

  if (allowedRoles && !allowedRoles.includes(user!.role)) {
    return <Navigate to={ROLE_HOME[user!.role]} replace />
  }

  return <>{children}</>
}

export default ProtectedRoute
