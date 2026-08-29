import { useContext } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { AuthContext } from '@/context/AuthContext'
import type { User } from '@/types/auth'
import { Button } from '@/components/ui/button'

function getDisplayName(user: User | null): string {
  if (!user) return ''
  return `${user.first_name} ${user.last_name}`
}

function CustomerLayout() {
  const { user, logout } = useContext(AuthContext)!
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <span className="text-lg font-semibold text-foreground">
              Fraud Detection
            </span>
            <span className="hidden rounded-full bg-muted px-2.5 py-0.5 text-xs font-medium text-muted-foreground sm:inline-block">
              Customer Portal
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden text-sm text-muted-foreground sm:inline">
              {getDisplayName(user)}
            </span>
            <Button variant="outline" size="sm" onClick={handleLogout}>
              Sign out
            </Button>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="border-b bg-card" aria-label="Customer navigation">
        <div className="mx-auto flex max-w-7xl gap-4 px-4 py-2">
          <NavLink
            to="/customer"
            end
            className={({ isActive }) =>
              `text-sm font-medium transition-colors ${
                isActive
                  ? 'text-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`
            }
          >
            Banking
          </NavLink>
          <NavLink
            to="/customer/history"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors ${
                isActive
                  ? 'text-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`
            }
          >
            Transaction History
          </NavLink>
        </div>
      </nav>

      {/* Main content */}
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}

export default CustomerLayout
