import { Outlet } from 'react-router-dom'

function AuthLayout() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-muted/40 px-4 py-8">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold tracking-tight text-foreground">
            Fraud Detection
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            AI-Powered Real-Time Protection
          </p>
        </div>
        <Outlet />
      </div>
    </div>
  )
}

export default AuthLayout
