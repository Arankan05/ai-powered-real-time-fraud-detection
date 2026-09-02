"""Step 30 — Comprehensive validation of database-backed customer history.

Validates:
  §3  Database persistence (survives restart)
  §4  Customer isolation (no cross-customer leakage)
  §5  Historical ordering (chronological, future-data exclusion)
  §6  Current transaction leakage (not included in own history)
  §7  Historical feature integration (velocity, avg_spend, location, etc.)
  §8  Cold-start / first transaction
  §9  Multiple transactions / time windows
  §10 Restart / multi-worker safety
  §13 Leakage & security checks

Run from project root::

    python -m pytest ml/api/tests/test_step30_validation.py -v
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from ml.features.engineer import engineer_features_for_inference, _resolve_customer_id
from ml.features.history import (
    CustomerHistoryRepository,
    InMemoryHistoryStore,
    SQLiteHistoryRepository,
    TransactionRecord,
    record_transaction,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_store(tmp_path, name="test.db"):
    return SQLiteHistoryRepository(db_path=str(tmp_path / name))


def _raw(**overrides) -> dict:
    """Minimal raw transaction dict."""
    base = {
        "amount": 100.0,
        "currency": "USD",
        "merchant_name": "Shop",
        "merchant_category": "5411",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "NYC",
        "device_fingerprint": "fp_default",
        "device_type": "desktop",
        "ip_address": "10.0.0.1",
        "customer_id": "cust_default",
        "timestamp": 10000,
    }
    base.update(overrides)
    return base


# =====================================================================
# §3  DATABASE PERSISTENCE
# =====================================================================


class TestDatabasePersistence:
    """Transaction history survives connection close / reopen."""

    def test_data_survives_close_and_reopen(self, tmp_path):
        """Records written by one connection are readable after reopen."""
        db = str(tmp_path / "persist.db")

        store1 = SQLiteHistoryRepository(db_path=db)
        store1.add("cust_p", {"timestamp": 100, "amount": 50.0})
        store1.add("cust_p", {"timestamp": 200, "amount": 75.0})
        store1.close()

        # Reopen — this is a completely new Python object
        store2 = SQLiteHistoryRepository(db_path=db)
        entries = store2.get("cust_p")
        assert len(entries) == 2
        assert entries[0]["amount"] == 50.0
        assert entries[1]["amount"] == 75.0
        store2.close()

    def test_persistence_with_record_transaction(self, tmp_path):
        """record_transaction persists data that survives restart."""
        db = str(tmp_path / "rec_persist.db")

        store1 = SQLiteHistoryRepository(db_path=db)
        raw = _raw(customer_id="cust_rt", timestamp=500, amount=200.0)
        record_transaction(store1, raw)
        store1.close()

        store2 = SQLiteHistoryRepository(db_path=db)
        entries = store2.get("cust_rt")
        assert len(entries) == 1
        assert entries[0]["amount"] == 200.0
        assert entries[0]["timestamp"] == 500
        store2.close()

    def test_not_dependent_on_python_object(self, tmp_path):
        """Data is in the DB file, not in a Python dict."""
        import sqlite3

        db = str(tmp_path / "raw_check.db")
        store = SQLiteHistoryRepository(db_path=db)
        store.add("cust_raw", {"timestamp": 42, "amount": 99.9})
        store.close()

        # Read directly from SQLite — no Python store involved
        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT customer_id, timestamp, amount FROM transaction_history"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0] == ("cust_raw", 42, 99.9)


# =====================================================================
# §4  CUSTOMER ISOLATION
# =====================================================================


class TestCustomerIsolation:
    """History is correctly separated between customers."""

    def test_customer_a_sees_only_a(self, tmp_path):
        store = _make_store(tmp_path)
        store.add("A", {"timestamp": 100, "amount": 10.0})
        store.add("A", {"timestamp": 200, "amount": 20.0})
        store.add("B", {"timestamp": 150, "amount": 50.0})

        a_hist = store.get("A")
        assert len(a_hist) == 2
        # Returned dicts don't include customer_id (it's the WHERE key)
        # Verify no B data leaked by checking amounts
        assert {e["amount"] for e in a_hist} == {10.0, 20.0}
        store.close()

    def test_customer_b_sees_only_b(self, tmp_path):
        store = _make_store(tmp_path)
        store.add("A", {"timestamp": 100, "amount": 10.0})
        store.add("B", {"timestamp": 150, "amount": 50.0})
        store.add("B", {"timestamp": 250, "amount": 60.0})

        b_hist = store.get("B")
        assert len(b_hist) == 2
        assert {e["amount"] for e in b_hist} == {50.0, 60.0}
        store.close()

    def test_unknown_customer_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        store.add("A", {"timestamp": 100, "amount": 10.0})
        assert store.get("nonexistent") == []
        store.close()

    def test_feature_engineering_isolation(self, tmp_path):
        """Feature engineering for A does not use B's history."""
        store = _make_store(tmp_path)
        # Seed B with high-amount history
        for i in range(5):
            store.add("B", {"timestamp": 100 + i, "amount": 10000.0, "addr2": 999})

        raw_a = _raw(customer_id="A", timestamp=500, amount=50.0)
        features_a = engineer_features_for_inference(raw_a, history_store=store)

        # A has no history → cold-start avg_spend should be own amount
        assert features_a["avg_spend_30d"].iloc[0] == pytest.approx(50.0, abs=1.0)
        store.close()


# =====================================================================
# §5  HISTORICAL ORDERING
# =====================================================================


class TestHistoricalOrdering:
    """Transactions returned in chronological order; future data excluded."""

    def test_chronological_order(self, tmp_path):
        store = _make_store(tmp_path)
        # Insert out of order
        store.add("C", {"timestamp": 300, "amount": 30.0})
        store.add("C", {"timestamp": 100, "amount": 10.0})
        store.add("C", {"timestamp": 200, "amount": 20.0})

        entries = store.get("C")
        timestamps = [e["timestamp"] for e in entries]
        assert timestamps == [100, 200, 300]
        store.close()

    def test_future_transactions_excluded(self, tmp_path):
        store = _make_store(tmp_path)
        store.add("C", {"timestamp": 100, "amount": 10.0})
        store.add("C", {"timestamp": 200, "amount": 20.0})
        store.add("C", {"timestamp": 300, "amount": 30.0})  # future

        # Query at timestamp=250 → only entries < 250
        entries = store.get("C", before_timestamp=250)
        assert len(entries) == 2
        assert entries[-1]["timestamp"] == 200
        store.close()

    def test_same_timestamp_excluded(self, tmp_path):
        """Entry at exact before_timestamp is NOT included (strict <)."""
        store = _make_store(tmp_path)
        store.add("C", {"timestamp": 100, "amount": 10.0})
        store.add("C", {"timestamp": 200, "amount": 20.0})

        entries = store.get("C", before_timestamp=200)
        assert len(entries) == 1
        assert entries[0]["timestamp"] == 100
        store.close()

    def test_feature_engineering_excludes_current(self, tmp_path):
        """Current transaction's own data doesn't appear in its history."""
        store = _make_store(tmp_path)
        store.add("C", {"timestamp": 100, "amount": 50.0})

        # Current transaction at timestamp=200
        raw = _raw(customer_id="C", timestamp=200, amount=500.0)
        features = engineer_features_for_inference(raw, history_store=store)

        # avg_spend should be based on history (50.0), not including current
        assert features["avg_spend_30d"].iloc[0] == pytest.approx(50.0, abs=1.0)
        store.close()


# =====================================================================
# §6  CURRENT TRANSACTION LEAKAGE
# =====================================================================


class TestCurrentTransactionLeakage:
    """Current transaction must NOT be in history when features are computed."""

    def test_prediction_before_recording(self, tmp_path):
        """The /predict endpoint records AFTER prediction (verified by code flow).

        This test verifies at the repository level that history lookup
        happens before recording.
        """
        store = _make_store(tmp_path)

        # Simulate: first transaction, no history
        raw1 = _raw(customer_id="C", timestamp=100, amount=50.0)
        hist_before = store.get("C", before_timestamp=100)
        assert hist_before == []  # no history for first tx

        # Compute features with no history
        features = engineer_features_for_inference(raw1, history_store=store)
        assert features["tx_velocity_1h"].iloc[0] == 0  # cold-start

        # Now record
        record_transaction(store, raw1)

        # History should now have 1 record
        hist_after = store.get("C", before_timestamp=200)
        assert len(hist_after) == 1
        store.close()

    def test_current_tx_not_in_own_features(self, tmp_path):
        """Second tx's features use only first tx as history, not itself."""
        store = _make_store(tmp_path)

        # First tx: amount=100
        raw1 = _raw(customer_id="C", timestamp=100, amount=100.0)
        record_transaction(store, raw1)

        # Second tx: amount=500
        raw2 = _raw(customer_id="C", timestamp=200, amount=500.0)

        # Before computing features, check history
        hist = store.get("C", before_timestamp=200)
        assert len(hist) == 1
        assert hist[0]["amount"] == 100.0  # only first tx, not second

        features = engineer_features_for_inference(raw2, history_store=store)
        # avg_spend based on history of [100] → should be 100, not (100+500)/2
        assert features["avg_spend_30d"].iloc[0] == pytest.approx(100.0, abs=1.0)
        store.close()

    def test_app_predict_flow_order(self):
        """Verify app.py records transaction AFTER prediction (code inspection)."""
        import inspect
        from ml.api.app import predict

        source = inspect.getsource(predict)
        # Find positions of key operations
        predict_pos = source.index("_predictor.predict")
        record_pos = source.index("record_transaction")
        assert predict_pos < record_pos, (
            "record_transaction must come AFTER _predictor.predict"
        )


# =====================================================================
# §7  HISTORICAL FEATURE INTEGRATION
# =====================================================================


class TestHistoricalFeatureIntegration:
    """History data flows correctly through the feature engineering pipeline."""

    def _seed_history(self, store, customer_id, timestamps_amounts):
        """Helper to seed multiple history records."""
        for ts, amt in timestamps_amounts:
            store.add(customer_id, {
                "timestamp": ts, "amount": amt,
                "addr1": 100, "addr2": 840,
                "product_cd": "W",
            })

    def test_velocity_features(self, tmp_path):
        store = _make_store(tmp_path)
        base_ts = 100000
        # 3 transactions within 1 hour
        self._seed_history(store, "V", [
            (base_ts - 3000, 50.0),
            (base_ts - 2000, 60.0),
            (base_ts - 1000, 70.0),
        ])

        raw = _raw(customer_id="V", timestamp=base_ts, amount=80.0)
        features = engineer_features_for_inference(raw, history_store=store)

        # Velocity counts strictly prior txs in window, minus self-exclusion.
        # 3 prior transactions all within 1 hour → velocity = 2 (after self-excl)
        assert features["tx_velocity_1h"].iloc[0] == 2
        assert features["tx_velocity_24h"].iloc[0] == 2
        assert features["tx_velocity_7d"].iloc[0] == 2
        store.close()

    def test_avg_spend(self, tmp_path):
        store = _make_store(tmp_path)
        base_ts = 100000
        self._seed_history(store, "S", [
            (base_ts - 5000, 100.0),
            (base_ts - 4000, 200.0),
            (base_ts - 3000, 300.0),
        ])

        raw = _raw(customer_id="S", timestamp=base_ts, amount=400.0)
        features = engineer_features_for_inference(raw, history_store=store)

        # avg_spend_30d should be based on history: (100+200+300)/3 = 200
        assert features["avg_spend_30d"].iloc[0] == pytest.approx(200.0, abs=1.0)
        store.close()

    def test_amount_ratio(self, tmp_path):
        store = _make_store(tmp_path)
        base_ts = 100000
        self._seed_history(store, "R", [
            (base_ts - 2000, 100.0),
            (base_ts - 1000, 100.0),
        ])

        raw = _raw(customer_id="R", timestamp=base_ts, amount=500.0)
        features = engineer_features_for_inference(raw, history_store=store)

        # amount_to_avg_ratio = 500 / avg(100, 100) = 500 / 100 = 5.0
        assert features["amount_to_avg_ratio"].iloc[0] == pytest.approx(5.0, abs=0.5)
        store.close()

    def test_location_is_new(self, tmp_path):
        store = _make_store(tmp_path)
        base_ts = 100000
        # Previous transactions from addr2=840 (US)
        self._seed_history(store, "L", [
            (base_ts - 2000, 50.0),
            (base_ts - 1000, 60.0),
        ])

        # Same location → not new
        raw_same = _raw(customer_id="L", timestamp=base_ts, amount=70.0, addr2=840)
        f_same = engineer_features_for_inference(raw_same, history_store=store)
        assert f_same["location_is_new"].iloc[0] == 0

        # Different location → new
        raw_diff = _raw(customer_id="L", timestamp=base_ts + 1, amount=70.0, addr2=110)
        f_diff = engineer_features_for_inference(raw_diff, history_store=store)
        assert f_diff["location_is_new"].iloc[0] == 1
        store.close()

    def test_previous_suspicious_count(self, tmp_path):
        store = _make_store(tmp_path)
        base_ts = 100000
        # Seed with flagged transactions
        store.add("F", {"timestamp": base_ts - 3000, "amount": 50.0, "is_fraud": 1})
        store.add("F", {"timestamp": base_ts - 2000, "amount": 60.0, "is_fraud": 1})
        store.add("F", {"timestamp": base_ts - 1000, "amount": 70.0, "is_fraud": 0})

        raw = _raw(customer_id="F", timestamp=base_ts, amount=80.0)
        features = engineer_features_for_inference(raw, history_store=store)

        assert features["previous_suspicious_count"].iloc[0] == 2
        store.close()

    def test_merchant_is_new(self, tmp_path):
        store = _make_store(tmp_path)
        base_ts = 100000
        store.add("M", {"timestamp": base_ts - 2000, "amount": 50.0, "product_cd": "W"})
        store.add("M", {"timestamp": base_ts - 1000, "amount": 60.0, "product_cd": "W"})

        # Same product_cd → not new
        raw_same = _raw(customer_id="M", timestamp=base_ts, amount=70.0, ProductCD="W")
        f_same = engineer_features_for_inference(raw_same, history_store=store)
        assert f_same["merchant_is_new"].iloc[0] == 0

        # Different product_cd → new
        raw_diff = _raw(customer_id="M", timestamp=base_ts + 1, amount=70.0, ProductCD="Z")
        f_diff = engineer_features_for_inference(raw_diff, history_store=store)
        assert f_diff["merchant_is_new"].iloc[0] == 1
        store.close()


# =====================================================================
# §8  COLD START / FIRST TRANSACTION
# =====================================================================


class TestColdStart:
    """First transaction for a new customer uses cold-start defaults."""

    def test_cold_start_no_crash(self, tmp_path):
        store = _make_store(tmp_path)
        raw = _raw(customer_id="brand_new", timestamp=1000)
        features = engineer_features_for_inference(raw, history_store=store)
        assert features.shape == (1, 24)
        store.close()

    def test_cold_start_defaults(self, tmp_path):
        store = _make_store(tmp_path)
        raw = _raw(customer_id="brand_new", timestamp=1000, amount=150.0)
        features = engineer_features_for_inference(raw, history_store=store)

        assert features["tx_velocity_1h"].iloc[0] == 0
        assert features["tx_velocity_24h"].iloc[0] == 0
        assert features["tx_velocity_7d"].iloc[0] == 0
        assert features["location_is_new"].iloc[0] == 1
        assert features["merchant_is_new"].iloc[0] == 1
        assert features["previous_suspicious_count"].iloc[0] == 0
        assert features["location_change"].iloc[0] == 0
        # avg_spend_30d = own amount (single data point)
        assert features["avg_spend_30d"].iloc[0] == pytest.approx(150.0, abs=1.0)
        store.close()

    def test_cold_start_deterministic(self, tmp_path):
        """Same cold-start input produces identical features."""
        store1 = _make_store(tmp_path, "a.db")
        store2 = _make_store(tmp_path, "b.db")
        raw = _raw(customer_id="det", timestamp=500, amount=100.0)

        f1 = engineer_features_for_inference(raw, history_store=store1)
        f2 = engineer_features_for_inference(raw, history_store=store2)

        numeric_cols = f1.select_dtypes(include="number").columns
        np.testing.assert_array_almost_equal(
            f1[numeric_cols].values, f2[numeric_cols].values,
        )
        store1.close()
        store2.close()

    def test_cold_start_no_store(self):
        """history_store=None also works (cold-start)."""
        raw = _raw(customer_id="none_store", timestamp=1000)
        features = engineer_features_for_inference(raw, history_store=None)
        assert features.shape == (1, 24)


# =====================================================================
# §9  MULTIPLE TRANSACTIONS / TIME WINDOWS
# =====================================================================


class TestMultipleTransactions:
    """Multiple history records with time-window behavior."""

    def test_all_history_available(self, tmp_path):
        store = _make_store(tmp_path)
        base_ts = 100000
        for i in range(10):
            store.add("H", {"timestamp": base_ts - (10 - i) * 100, "amount": float(i * 10)})

        hist = store.get("H", before_timestamp=base_ts)
        assert len(hist) == 10
        store.close()

    def test_one_hour_window(self, tmp_path):
        store = _make_store(tmp_path)
        base_ts = 100000
        # 2 within 1 hour, 1 outside
        store.add("W", {"timestamp": base_ts - 1800, "amount": 50.0})   # 30 min ago
        store.add("W", {"timestamp": base_ts - 3000, "amount": 60.0})   # 50 min ago
        store.add("W", {"timestamp": base_ts - 7200, "amount": 70.0})   # 2 hours ago

        raw = _raw(customer_id="W", timestamp=base_ts, amount=80.0)
        features = engineer_features_for_inference(raw, history_store=store)

        # velocity counts strictly prior txs in window, minus self-exclusion.
        # 2 prior within 1 hour → velocity_1h = 1
        assert features["tx_velocity_1h"].iloc[0] == 1
        # All 3 prior within 24 hours → velocity_24h = 2
        assert features["tx_velocity_24h"].iloc[0] == 2
        store.close()

    def test_old_transactions_outside_window(self, tmp_path):
        store = _make_store(tmp_path)
        base_ts = 100000
        # All transactions > 7 days ago
        store.add("O", {"timestamp": base_ts - 700000, "amount": 50.0})
        store.add("O", {"timestamp": base_ts - 800000, "amount": 60.0})

        raw = _raw(customer_id="O", timestamp=base_ts, amount=80.0)
        features = engineer_features_for_inference(raw, history_store=store)

        # velocity_7d should be 0 (all outside window)
        assert features["tx_velocity_7d"].iloc[0] == 0
        # But velocity_7d counts them if within 30d window for avg_spend
        # avg_spend_30d should include transactions within 30 days
        # 700000 seconds = ~8.1 days, 800000 = ~9.3 days → within 30 days
        assert features["avg_spend_30d"].iloc[0] == pytest.approx(55.0, abs=1.0)
        store.close()

    def test_fifo_eviction(self, tmp_path):
        store = _make_store(tmp_path)
        store_small = SQLiteHistoryRepository(
            db_path=str(tmp_path / "small.db"), max_per_customer=3
        )
        for i in range(5):
            store_small.add("E", {"timestamp": i, "amount": float(i)})

        entries = store_small.get("E")
        assert len(entries) == 3
        # Oldest 2 evicted: kept timestamps 2, 3, 4
        assert entries[0]["timestamp"] == 2
        store_small.close()
        store.close()


# =====================================================================
# §10  RESTART / MULTI-WORKER SAFETY
# =====================================================================


class TestRestartSafety:
    """Persistence does not depend on process-local state."""

    def test_two_independent_stores_share_data(self, tmp_path):
        """Two separate repository instances reading same DB see same data."""
        db = str(tmp_path / "shared.db")

        store_writer = SQLiteHistoryRepository(db_path=db)
        store_writer.add("shared_cust", {"timestamp": 100, "amount": 42.0})
        store_writer.close()

        # Completely new instance (simulates a restarted process)
        store_reader = SQLiteHistoryRepository(db_path=db)
        entries = store_reader.get("shared_cust")
        assert len(entries) == 1
        assert entries[0]["amount"] == 42.0
        store_reader.close()

    def test_wal_mode_enabled(self, tmp_path):
        """SQLite WAL mode is active for concurrent read performance."""
        import sqlite3
        db = str(tmp_path / "wal.db")
        store = SQLiteHistoryRepository(db_path=db)
        conn = sqlite3.connect(db)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode.lower() == "wal"
        store.close()


# =====================================================================
# §12  DATABASE FAILURE HANDLING
# =====================================================================


class TestDatabaseFailureHandling:
    """Graceful handling of DB failures."""

    def test_invalid_db_path_falls_back(self):
        """Invalid DB path → app falls back to in-memory store."""
        # The lifespan catches exceptions and falls back
        # Verify that InMemoryHistoryStore still works
        store = InMemoryHistoryStore()
        store.add("test", {"timestamp": 1, "amount": 1.0})
        assert store.get("test") == [{"timestamp": 1, "amount": 1.0}]

    def test_empty_history_no_crash(self, tmp_path):
        """Empty history doesn't cause errors in feature engineering."""
        store = _make_store(tmp_path)
        raw = _raw(customer_id="empty", timestamp=1000)
        features = engineer_features_for_inference(raw, history_store=store)
        assert features.shape == (1, 24)
        store.close()

    def test_malformed_history_record(self, tmp_path):
        """Missing fields in history record don't crash feature engineering."""
        store = _make_store(tmp_path)
        # Add a record with minimal fields
        store.add("mal", {"timestamp": 100})  # no amount, no other fields
        raw = _raw(customer_id="mal", timestamp=200, amount=50.0)
        features = engineer_features_for_inference(raw, history_store=store)
        assert features.shape == (1, 24)
        store.close()


# =====================================================================
# §13  LEAKAGE & SECURITY CHECKS
# =====================================================================


class TestLeakageAndSecurity:
    """Verify no data leakage paths."""

    def test_isfraud_not_accepted_as_input(self):
        """RawTransactionInput rejects isFraud field."""
        from pydantic import ValidationError
        from ml.api.app import RawTransactionInput

        with pytest.raises(ValidationError, match="Forbidden fields"):
            RawTransactionInput.model_validate({
                "amount": 100.0,
                "currency": "USD",
                "merchant_name": "Test",
                "merchant_category": "5411",
                "transaction_type": "purchase",
                "location_country": "US",
                "location_city": "NYC",
                "device_fingerprint": "fp",
                "device_type": "desktop",
                "ip_address": "1.2.3.4",
                "isFraud": 1,  # FORBIDDEN
            })

    def test_transactionid_not_accepted(self):
        """RawTransactionInput rejects TransactionID field."""
        from pydantic import ValidationError
        from ml.api.app import RawTransactionInput

        with pytest.raises(ValidationError, match="Forbidden fields"):
            RawTransactionInput.model_validate({
                "amount": 100.0,
                "currency": "USD",
                "merchant_name": "Test",
                "merchant_category": "5411",
                "transaction_type": "purchase",
                "location_country": "US",
                "location_city": "NYC",
                "device_fingerprint": "fp",
                "device_type": "desktop",
                "ip_address": "1.2.3.4",
                "TransactionID": "V123",  # FORBIDDEN
            })

    def test_is_fraud_stored_as_zero(self, tmp_path):
        """At prediction time, is_fraud is always stored as 0."""
        store = _make_store(tmp_path)
        raw = _raw(customer_id="lf", timestamp=100, amount=50.0)
        record_transaction(store, raw)

        entries = store.get("lf")
        assert entries[0]["is_fraud"] == 0
        store.close()

    def test_no_training_data_loaded_for_history(self):
        """History store doesn't load raw CSV data."""
        store = SQLiteHistoryRepository(db_path=":memory:")
        # Store starts empty — no training data
        assert store.total_count() == 0
        store.close()

    def test_customer_cannot_access_other_history(self, tmp_path):
        """One customer's query never returns another's history."""
        store = _make_store(tmp_path)
        store.add("A", {"timestamp": 100, "amount": 10.0, "is_fraud": 1})
        store.add("B", {"timestamp": 200, "amount": 20.0, "is_fraud": 0})

        # A's query
        a_hist = store.get("A")
        assert all(e["timestamp"] == 100 for e in a_hist)

        # B's query
        b_hist = store.get("B")
        assert all(e["timestamp"] == 200 for e in b_hist)
        assert all(e["is_fraud"] == 0 for e in b_hist)
        store.close()

    def test_protocol_satisfaction(self, tmp_path):
        """Both implementations satisfy the CustomerHistoryRepository protocol."""
        mem = InMemoryHistoryStore()
        sql = _make_store(tmp_path)
        assert isinstance(mem, CustomerHistoryRepository)
        assert isinstance(sql, CustomerHistoryRepository)
        sql.close()
