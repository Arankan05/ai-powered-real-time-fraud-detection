"""Thread-safe prediction metrics collector and drift detector.

Provides :class:`PredictionMetrics` — a lightweight, process-local
metrics aggregator that tracks request counts, latency, error
categories, prediction distributions, and basic drift signals.

Design constraints (Step 43):
  * **Thread-safe** — all counters protected by ``threading.Lock``.
  * **Bounded memory** — latency samples use a fixed-size ``deque``.
  * **No raw data storage** — no transaction payloads, customer IDs,
    or PII retained in metrics.
  * **No prediction impact** — monitoring is purely observational.
  * **Cardinality-safe labels** — only bounded categorical labels
    (decision, risk_level, error_category); never per-customer or
    per-transaction identifiers.

Usage::

    from ml.monitoring.metrics import metrics

    metrics.record_success(
        latency_ms=12.3,
        fraud_prediction=0,
        fraud_probability=0.15,
        decision="APPROVE",
        risk_level="LOW",
        amount=150.0,
        model_version="fraud-xgb-v1.0.0",
    )

    snapshot = metrics.snapshot()
"""

from __future__ import annotations

import logging
import math
import os
import statistics
import threading
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

_MAX_LATENCY_SAMPLES = 5_000
_MAX_DRIFT_SAMPLES = 2_000

_VALID_DECISIONS = frozenset({"APPROVE", "VERIFY", "HOLD"})
_VALID_RISK_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH"})
_VALID_ERROR_CATEGORIES = frozenset({
    "validation",
    "model_unavailable",
    "prediction_failure",
    "feature_engineering",
    "history_failure",
    "shap_failure",
    "timeout",
    "unknown",
})


# ── Configuration ─────────────────────────────────────────────────────


class MonitoringConfig:
    """Environment-driven monitoring thresholds.

    All values are loaded from environment variables at construction
    time.  Defaults are production-reasonable; override via ``.env``
    or deployment configuration.
    """

    def __init__(self) -> None:
        self.latency_warn_seconds: float = float(
            os.environ.get("ML_LATENCY_WARN_SECONDS", "5.0")
        )
        self.error_rate_warn_threshold: float = float(
            os.environ.get("ML_ERROR_RATE_WARN_THRESHOLD", "0.10")
        )
        self.drift_std_multiplier: float = float(
            os.environ.get("ML_DRIFT_STD_MULTIPLIER", "3.0")
        )
        self.fraud_rate_warn_threshold: float = float(
            os.environ.get("ML_FRAUD_RATE_WARN_THRESHOLD", "0.50")
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "latency_warn_seconds": self.latency_warn_seconds,
            "error_rate_warn_threshold": self.error_rate_warn_threshold,
            "drift_std_multiplier": self.drift_std_multiplier,
            "fraud_rate_warn_threshold": self.fraud_rate_warn_threshold,
        }


# ── Metrics collector ─────────────────────────────────────────────────


class PredictionMetrics:
    """Thread-safe prediction metrics aggregator.

    Counters, latency samples, and drift statistics are protected by a
    single ``threading.Lock``.  All public methods are safe to call
    from any thread.
    """

    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self._config = config or MonitoringConfig()
        self._lock = threading.Lock()

        # ── Counters ──────────────────────────────────────────────
        self._total_requests: int = 0
        self._successful_predictions: int = 0
        self._failed_predictions: int = 0
        self._fraud_count: int = 0
        self._non_fraud_count: int = 0
        self._slow_predictions: int = 0

        # ── Error categories ──────────────────────────────────────
        self._errors: dict[str, int] = {c: 0 for c in _VALID_ERROR_CATEGORIES}

        # ── Distributions ─────────────────────────────────────────
        self._decisions: dict[str, int] = {d: 0 for d in _VALID_DECISIONS}
        self._risk_levels: dict[str, int] = {r: 0 for r in _VALID_RISK_LEVELS}

        # ── Model version ─────────────────────────────────────────
        self._model_version: str | None = None

        # ── Latency samples (bounded) ─────────────────────────────
        self._latency_samples: deque[float] = deque(maxlen=_MAX_LATENCY_SAMPLES)

        # ── Drift tracking (bounded numeric distributions) ────────
        self._amount_samples: deque[float] = deque(maxlen=_MAX_DRIFT_SAMPLES)
        self._probability_samples: deque[float] = deque(maxlen=_MAX_DRIFT_SAMPLES)
        self._risk_score_samples: deque[int] = deque(maxlen=_MAX_DRIFT_SAMPLES)

        # ── Baseline (optional, loaded from env/config) ───────────
        self._baseline: dict[str, dict[str, float]] | None = None
        self._load_baseline()

        # ── Drift warnings ────────────────────────────────────────
        self._drift_warnings: list[dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────

    def record_success(
        self,
        *,
        latency_ms: float,
        fraud_prediction: int,
        fraud_probability: float,
        decision: str,
        risk_level: str,
        risk_score: int,
        amount: float,
        model_version: str,
    ) -> None:
        """Record a successful prediction with all associated metrics."""
        latency_sec = latency_ms / 1000.0

        with self._lock:
            self._total_requests += 1
            self._successful_predictions += 1

            if fraud_prediction == 1:
                self._fraud_count += 1
            else:
                self._non_fraud_count += 1

            if decision in _VALID_DECISIONS:
                self._decisions[decision] += 1
            if risk_level in _VALID_RISK_LEVELS:
                self._risk_levels[risk_level] += 1

            self._model_version = model_version
            self._latency_samples.append(latency_sec)

            if latency_sec > self._config.latency_warn_seconds:
                self._slow_predictions += 1

            # Drift tracking — bounded numeric samples
            self._amount_samples.append(float(amount))
            self._probability_samples.append(float(fraud_probability))
            self._risk_score_samples.append(int(risk_score))

        # Check thresholds (outside lock to minimize hold time)
        self._check_thresholds()

    def record_error(self, *, category: str) -> None:
        """Record a failed prediction with a bounded error category."""
        safe_cat = category if category in _VALID_ERROR_CATEGORIES else "unknown"
        with self._lock:
            self._total_requests += 1
            self._failed_predictions += 1
            self._errors[safe_cat] += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time snapshot of all metrics.

        The snapshot contains only aggregate data — no raw transactions,
        customer IDs, or sensitive information.
        """
        with self._lock:
            total = self._total_requests
            successful = self._successful_predictions
            failed = self._failed_predictions

            # Error rate
            error_rate = (failed / total) if total > 0 else 0.0

            # Latency statistics
            latency_stats = self._compute_latency_stats()

            # Drift status
            drift_status = self._compute_drift_status()

            return {
                "total_requests": total,
                "successful_predictions": successful,
                "failed_predictions": failed,
                "error_rate": round(error_rate, 4),
                "fraud_count": self._fraud_count,
                "non_fraud_count": self._non_fraud_count,
                "slow_predictions": self._slow_predictions,
                "decisions": dict(self._decisions),
                "risk_levels": dict(self._risk_levels),
                "errors": dict(self._errors),
                "model_version": self._model_version,
                "latency": latency_stats,
                "drift": drift_status,
                "config": self._config.to_dict(),
            }

    def reset(self) -> None:
        """Reset all counters and samples (for testing only)."""
        with self._lock:
            self._total_requests = 0
            self._successful_predictions = 0
            self._failed_predictions = 0
            self._fraud_count = 0
            self._non_fraud_count = 0
            self._slow_predictions = 0
            for k in self._errors:
                self._errors[k] = 0
            for k in self._decisions:
                self._decisions[k] = 0
            for k in self._risk_levels:
                self._risk_levels[k] = 0
            self._model_version = None
            self._latency_samples.clear()
            self._amount_samples.clear()
            self._probability_samples.clear()
            self._risk_score_samples.clear()
            self._drift_warnings.clear()

    # ── Internal helpers ──────────────────────────────────────────

    def _compute_latency_stats(self) -> dict[str, Any]:
        """Compute latency statistics from bounded samples.

        Must be called while holding ``self._lock``.
        """
        samples = list(self._latency_samples)
        if not samples:
            return {
                "count": 0,
                "mean_seconds": None,
                "p50_seconds": None,
                "p95_seconds": None,
                "p99_seconds": None,
                "min_seconds": None,
                "max_seconds": None,
            }
        return {
            "count": len(samples),
            "mean_seconds": round(statistics.mean(samples), 4),
            "p50_seconds": round(self._percentile(samples, 50), 4),
            "p95_seconds": round(self._percentile(samples, 95), 4),
            "p99_seconds": round(self._percentile(samples, 99), 4),
            "min_seconds": round(min(samples), 4),
            "max_seconds": round(max(samples), 4),
        }

    @staticmethod
    def _percentile(sorted_data: list[float], pct: float) -> float:
        """Compute a percentile from a list of values."""
        if not sorted_data:
            return 0.0
        data = sorted(sorted_data)
        k = (len(data) - 1) * (pct / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        return data[f] * (c - k) + data[c] * (k - f)

    def _compute_drift_status(self) -> dict[str, Any]:
        """Compute drift status against baseline (if configured).

        Must be called while holding ``self._lock``.
        """
        if self._baseline is None:
            return {
                "baseline_configured": False,
                "message": "No baseline configured. Drift monitoring unavailable.",
                "warnings": [],
            }

        warnings: list[dict[str, Any]] = []

        # Amount drift
        amount_drift = self._check_drift(
            list(self._amount_samples),
            self._baseline.get("amount"),
            "amount",
        )
        if amount_drift:
            warnings.append(amount_drift)

        # Fraud probability drift
        prob_drift = self._check_drift(
            list(self._probability_samples),
            self._baseline.get("fraud_probability"),
            "fraud_probability",
        )
        if prob_drift:
            warnings.append(prob_drift)

        # Risk score drift
        risk_drift = self._check_drift(
            [float(x) for x in self._risk_score_samples],
            self._baseline.get("risk_score"),
            "risk_score",
        )
        if risk_drift:
            warnings.append(risk_drift)

        return {
            "baseline_configured": True,
            "baseline": {
                k: {"mean": v["mean"], "std": v["std"]}
                for k, v in self._baseline.items()
            },
            "warnings": warnings,
            "drift_detected": len(warnings) > 0,
        }

    def _check_drift(
        self,
        samples: list[float],
        baseline: dict[str, float] | None,
        feature_name: str,
    ) -> dict[str, Any] | None:
        """Check if recent samples drift from baseline.

        Returns a drift warning dict if drift is detected, else None.
        """
        if baseline is None or len(samples) < 10:
            return None

        baseline_mean = baseline["mean"]
        baseline_std = baseline["std"]
        if baseline_std <= 0:
            return None  # Cannot detect drift with zero variance

        recent_mean = statistics.mean(samples)
        threshold = self._config.drift_std_multiplier * baseline_std
        deviation = abs(recent_mean - baseline_mean)

        if deviation > threshold:
            return {
                "feature": feature_name,
                "baseline_mean": round(baseline_mean, 4),
                "baseline_std": round(baseline_std, 4),
                "recent_mean": round(recent_mean, 4),
                "deviation": round(deviation, 4),
                "threshold": round(threshold, 4),
            }
        return None

    def _check_thresholds(self) -> None:
        """Log warnings if operational thresholds are exceeded.

        Called after each record_success (outside the lock).
        """
        with self._lock:
            total = self._total_requests
            failed = self._failed_predictions
            fraud = self._fraud_count
            successful = self._successful_predictions

        if total < 10:
            return  # Need a minimum sample size

        # Error rate warning
        error_rate = failed / total
        if error_rate > self._config.error_rate_warn_threshold:
            logger.warning(
                "Monitoring: error rate %.1f%% exceeds threshold %.1f%%",
                error_rate * 100,
                self._config.error_rate_warn_threshold * 100,
            )

        # Fraud rate warning
        if successful > 0:
            fraud_rate = fraud / successful
            if fraud_rate > self._config.fraud_rate_warn_threshold:
                logger.warning(
                    "Monitoring: fraud rate %.1f%% exceeds threshold %.1f%%",
                    fraud_rate * 100,
                    self._config.fraud_rate_warn_threshold * 100,
                )

    def _load_baseline(self) -> None:
        """Load baseline from environment variables if configured.

        Baseline format (env vars):
          ``ML_BASELINE_AMOUNT_MEAN``, ``ML_BASELINE_AMOUNT_STD``
          ``ML_BASELINE_PROB_MEAN``, ``ML_BASELINE_PROB_STD``
          ``ML_BASELINE_RISK_MEAN``, ``ML_BASELINE_RISK_STD``

        All must be present for the corresponding feature to be
        monitored.  If none are set, drift monitoring is unavailable.
        """
        baseline: dict[str, dict[str, float]] = {}

        amount_mean = os.environ.get("ML_BASELINE_AMOUNT_MEAN")
        amount_std = os.environ.get("ML_BASELINE_AMOUNT_STD")
        if amount_mean is not None and amount_std is not None:
            try:
                baseline["amount"] = {
                    "mean": float(amount_mean),
                    "std": float(amount_std),
                }
            except ValueError:
                logger.warning("Invalid ML_BASELINE_AMOUNT_MEAN/STD")

        prob_mean = os.environ.get("ML_BASELINE_PROB_MEAN")
        prob_std = os.environ.get("ML_BASELINE_PROB_STD")
        if prob_mean is not None and prob_std is not None:
            try:
                baseline["fraud_probability"] = {
                    "mean": float(prob_mean),
                    "std": float(prob_std),
                }
            except ValueError:
                logger.warning("Invalid ML_BASELINE_PROB_MEAN/STD")

        risk_mean = os.environ.get("ML_BASELINE_RISK_MEAN")
        risk_std = os.environ.get("ML_BASELINE_RISK_STD")
        if risk_mean is not None and risk_std is not None:
            try:
                baseline["risk_score"] = {
                    "mean": float(risk_mean),
                    "std": float(risk_std),
                }
            except ValueError:
                logger.warning("Invalid ML_BASELINE_RISK_MEAN/STD")

        if baseline:
            self._baseline = baseline
            logger.info(
                "Monitoring baseline loaded: %s",
                sorted(baseline.keys()),
            )
        else:
            self._baseline = None


# ── Module-level singleton ─────────────────────────────────────────────

metrics = PredictionMetrics()
"""Global prediction metrics instance.

Shared across all prediction requests.  Thread-safe.
"""
