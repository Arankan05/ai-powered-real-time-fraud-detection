import { Outlet } from 'react-router-dom'

function CustomerLayout() {
  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-7xl px-4 py-6">
        <Outlet />
      </div>
    </div>
  )
}

export default CustomerLayout
