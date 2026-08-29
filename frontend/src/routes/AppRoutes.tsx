import { Routes, Route, Navigate } from 'react-router-dom'

import AuthLayout from '@/layouts/AuthLayout'
import CustomerLayout from '@/layouts/CustomerLayout'
import AnalystLayout from '@/layouts/AnalystLayout'
import ProtectedRoute from '@/routes/ProtectedRoute'

import LoginPage from '@/pages/auth/LoginPage'
import RegisterPage from '@/pages/auth/RegisterPage'
import BankingPage from '@/pages/customer/BankingPage'
import TransactionDetailPage from '@/pages/customer/TransactionDetailPage'
import TransactionHistoryPage from '@/pages/customer/TransactionHistoryPage'
import DashboardPage from '@/pages/analyst/DashboardPage'
import AlertDetailPage from '@/pages/analyst/AlertDetailPage'
import FraudCheckPage from '@/pages/analyst/FraudCheckPage'

function AppRoutes() {
  return (
    <Routes>
      {/* Redirect root to login */}
      <Route path="/" element={<Navigate to="/login" replace />} />

      {/* Auth routes */}
      <Route element={<AuthLayout />}>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
      </Route>

      {/* Customer routes */}
      <Route
        element={
          <ProtectedRoute>
            <CustomerLayout />
          </ProtectedRoute>
        }
      >
        <Route path="/customer" element={<BankingPage />} />
        <Route path="/customer/history" element={<TransactionHistoryPage />} />
        <Route path="/customer/transactions/:id" element={<TransactionDetailPage />} />
      </Route>

      {/* Analyst routes */}
      <Route
        element={
          <ProtectedRoute>
            <AnalystLayout />
          </ProtectedRoute>
        }
      >
        <Route path="/analyst/dashboard" element={<DashboardPage />} />
        <Route path="/analyst/alerts/:id" element={<AlertDetailPage />} />
        <Route path="/analyst/fraud-check" element={<FraudCheckPage />} />
      </Route>
    </Routes>
  )
}

export default AppRoutes
