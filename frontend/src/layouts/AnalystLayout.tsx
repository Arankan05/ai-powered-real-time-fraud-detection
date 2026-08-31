import { useContext } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { AuthContext } from '@/context/AuthContext'
import type { User } from '@/types/auth'
import { Button } from '@/components/ui/button'
import {
  LayoutDashboard,
  ShieldAlert,
  SearchCheck,
  LogOut,
} from 'lucide-react'

function getDisplayName(user: User | null): string {
  if (!user) return ''
  return `${user.first_name} ${user.last_name}`
}

const navItems = [
  {
    to: '/analyst/dashboard',
    label: 'Dashboard',
    icon: LayoutDashboard,
    end: true,
  },
  {
    to: '/analyst/alerts',
    label: 'Alerts',
    icon: ShieldAlert,
    end: false,
  },
  {
    to: '/analyst/fraud-check',
    label: 'Fraud Check',
    icon: SearchCheck,
    end: true,
  },
] as const

function AnalystLayout() {
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
              Analyst Portal
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden text-sm text-muted-foreground sm:inline">
              {getDisplayName(user)}
            </span>
            <Button variant="outline" size="sm" onClick={handleLogout}>
              <LogOut className="mr-1.5 h-4 w-4" />
              Sign out
            </Button>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="border-b bg-card" aria-label="Analyst navigation">
        <div className="mx-auto flex max-w-7xl gap-1 px-4 py-2">
          {navItems.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-muted text-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                }`
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </div>
      </nav>

      {/* Main content */}
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}

export default AnalystLayout
