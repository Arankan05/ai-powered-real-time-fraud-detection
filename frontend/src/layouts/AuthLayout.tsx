import { Outlet } from 'react-router-dom'

function AuthLayout() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="w-full max-w-md px-4">
        <Outlet />
      </div>
    </div>
  )
}

export default AuthLayout
