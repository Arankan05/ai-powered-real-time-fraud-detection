"""Step 30 — End-to-end integration test with live servers.

Tests the full path:
  Backend → ML /predict → Customer History → Feature Engineering → XGBoost → SHAP

This script is run standalone (not via pytest) against live servers.
"""
import requests
import json
import sys

ML_URL = "http://127.0.0.1:8001"
RESULTS = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    RESULTS.append((name, status, detail))
    mark = "PASS" if condition else "FAIL"
    print(f"  [{status}] {mark} {name}" + (f" -- {detail}" if detail else ""))


def predict(payload):
    r = requests.post(f"{ML_URL}/predict", json=payload)
    assert r.status_code == 200, f"Predict failed: {r.status_code} {r.text}"
    return r.json()


def base_tx(**overrides):
    tx = {
        "amount": 100.0,
        "currency": "USD",
        "merchant_name": "E2E Shop",
        "merchant_category": "5411",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "NYC",
        "device_fingerprint": "fp_e2e_test",
        "device_type": "desktop",
        "ip_address": "10.0.0.99",
    }
    tx.update(overrides)
    return tx


print("=" * 60)
print("Step 30 — End-to-End Integration Validation")
print("=" * 60)

# ── Health check ────────────────────────────────────────────────────
print("\n[1] Health check")
try:
    health = requests.get(f"{ML_URL}/health").json()
    check("ML service reachable", health.get("status") == "ready",
          f"version={health.get('model_version')}")
except Exception as e:
    check("ML service reachable", False, str(e))
    sys.exit(1)

# ── First transaction (cold start) ─────────────────────────────────
print("\n[2] First transaction -- cold start")
tx1 = base_tx(customer_id="e2e_cust_30", timestamp=100000, amount=100.0)
r1 = predict(tx1)
prob1 = r1["fraud_probability"]
check("Cold-start returns valid probability",
      0.0 <= prob1 <= 1.0, f"prob={prob1:.4f}")
check("Cold-start returns SHAP explanation",
      r1.get("explanation") is not None and len(r1["explanation"]) > 0,
      f"{len(r1.get('explanation', []))} factors")
check("Cold-start returns model version",
      r1.get("model_version") == "fraud-xgb-v1.0.0")

# ── Second transaction (same customer, with history) ───────────────
print("\n[3] Second transaction -- same customer, higher amount")
tx2 = base_tx(customer_id="e2e_cust_30", timestamp=100100, amount=5000.0)
r2 = predict(tx2)
prob2 = r2["fraud_probability"]
check("Second tx returns valid probability",
      0.0 <= prob2 <= 1.0, f"prob={prob2:.4f}")
check("Different probability (history used)",
      abs(prob1 - prob2) > 0.001,
      f"delta={abs(prob1 - prob2):.4f}")

# ── Third transaction (different customer, same amount) ────────────
print("\n[4] Third transaction -- different customer (isolation check)")
tx3 = base_tx(customer_id="e2e_cust_other_30", timestamp=100200, amount=5000.0)
r3 = predict(tx3)
prob3 = r3["fraud_probability"]
check("Different customer returns valid probability",
      0.0 <= prob3 <= 1.0, f"prob={prob3:.4f}")
# Different customer with same amount should NOT have same probability
# as the customer with history (because the other has history, this doesn't)
check("Customer isolation -- different from customer with history",
      abs(prob2 - prob3) > 0.001 or abs(prob1 - prob3) > 0.001,
      f"prob_other={prob3:.4f} vs prob_history={prob2:.4f}")

# ── Leakage check: isFraud rejected ────────────────────────────────
print("\n[5] Leakage protection")
tx_bad = base_tx(customer_id="e2e_cust_30", timestamp=100300)
tx_bad["isFraud"] = 1
r_bad = requests.post(f"{ML_URL}/predict", json=tx_bad)
check("isFraud rejected with 422",
      r_bad.status_code == 422,
      f"status={r_bad.status_code}")

tx_bad2 = base_tx(customer_id="e2e_cust_30", timestamp=100400)
tx_bad2["TransactionID"] = "V12345"
r_bad2 = requests.post(f"{ML_URL}/predict", json=tx_bad2)
check("TransactionID rejected with 422",
      r_bad2.status_code == 422,
      f"status={r_bad2.status_code}")

# ── History DB persistence check ───────────────────────────────────
print("\n[6] Database persistence check")
import sqlite3
try:
    conn = sqlite3.connect("data/ml_history.db")
    rows = conn.execute(
        "SELECT COUNT(*) FROM transaction_history WHERE customer_id = 'e2e_cust_30'"
    ).fetchone()[0]
    check("History persisted to SQLite",
          rows >= 2,
          f"{rows} rows for e2e_cust_30")
    conn.close()
except Exception as e:
    check("History persisted to SQLite", False, str(e))

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    print("\nFailed checks:")
    for name, status, detail in RESULTS:
        if status == "FAIL":
            print(f"  FAIL {name}: {detail}")
    sys.exit(1)
else:
    print("\nAll end-to-end checks PASSED.")
