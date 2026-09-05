import { getTransactionId } from '../types/transaction'
import type { TransactionResponse } from '../types/transaction'

// Focused test verifying transaction_id extraction when id is null
function testTransactionIdExtraction() {
  const mockApiResponse: TransactionResponse = {
    transaction_id: 'a1eff806-bcd0-43c9-a22b-e9a63329d3ea',
    id: null,
    amount: 49.99,
    currency: 'USD',
    merchant_name: 'Coffee Shop',
    merchant_category: 'cafe',
    transaction_type: 'purchase',
    location_country: 'US',
    location_city: 'New York',
    device_fingerprint: 'dev_123',
    device_type: 'mobile',
    ip_address: '192.168.1.1',
    timestamp: 1788580941,
    status: 'COMPLETED',
    ml_score: 91,
    behaviour_score: 75,
    rule_score: 0,
    risk_score: 68,
    risk_level: 'MEDIUM',
    decision: 'VERIFY',
    model_version: 'fraud-xgb-v1.0.0',
    risk_factors: [],
    explanation: {
      ml_top_factors: [],
      behaviour_signals: [],
      rules_triggered: [],
    },
  }

  const resolvedId = getTransactionId(mockApiResponse)
  if (resolvedId !== 'a1eff806-bcd0-43c9-a22b-e9a63329d3ea') {
    throw new Error(`Expected canonical transaction_id, got '${resolvedId}'`)
  }

  console.log('✓ transaction_id extraction test passed: canonical transaction_id extracted successfully when id=null')
}

testTransactionIdExtraction()
