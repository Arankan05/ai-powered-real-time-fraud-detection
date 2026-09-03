"""Step 47 — End-to-end offline evaluation validation.

Runs the complete offline evaluation of the **real, verified production
model** on the **real held-out temporal test split** and verifies the 15
required end-to-end checks (spec §22):

 1. load verified model through existing governance
 2. identify exact model version/checksum
 3. load authorized evaluation data
 4. run evaluation
 5. calculate classification metrics
 6. calculate PR-AUC/ROC-AUC
 7. calculate threshold sweep
 8. perform recommendation
 9. calculate calibration metrics
10. verify recommendation is clearly marked evaluation-only
11. verify production threshold did not change
12. verify live `/predict` behavior remains unchanged
13. verify `/health` and `/ready` still expose the same active model
14. verify monitoring remains correct
15. verify no transaction audit records were incorrectly created

This script is run standalone (not via pytest), from the project root::

    python ml/tests/e2e_step47_evaluation.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

# Make the repository importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Isolate the live-service history store BEFORE the app is imported so
# this validation never touches data/ml_history.db.
_TMP_DIR = Path(tempfile.mkdtemp(prefix="step47_e2e_"))
HISTORY_DB = _TMP_DIR / "history.db"
os.environ["ML_HISTORY_DB_PATH"] = str(HISTORY_DB)

from ml.evaluation.config import EvaluationConfig  # noqa: E402
from ml.evaluation.runner import (  # noqa: E402
    DATASET_IDENTIFIER,
    REPORT_SCOPE,
    build_report,
    load_holdout_test_set,
)
from ml.predict.integrity import (  # noqa: E402
    compute_checksum,
    default_model_directory,
    load_manifest,
)
from ml.predict.registry import ModelRegistry  # noqa: E402

RESULTS = []

PAYLOAD = {
    "amount": 250.0,
    "currency": "USD",
    "merchant_name": "Step47 E2E Merchant",
    "merchant_category": "5732",
    "transaction_type": "purchase",
    "location_country": "US",
    "location_city": "Testville",
    "device_fingerprint": "fp_step47_e2e",
    "device_type": "desktop",
    "ip_address": "192.168.1.51",
    "timestamp": 1_000_000,
    "card1": 47002,
}

DETERMINISTIC_FIELDS = (
    "fraud_probability",
    "fraud_prediction",
    "threshold",
    "model_version",
    "ml_score",
    "behaviour_score",
    "rule_score",
    "risk_score",
    "risk_level",
    "decision",
)


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    RESULTS.append((name, status))
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


def history_count() -> int:
    conn = sqlite3.connect(HISTORY_DB)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM transaction_history"
        ).fetchone()[0]
    finally:
        conn.close()


def main() -> int:
    print("=" * 70)
    print("Step 47 — End-to-End Offline Evaluation Validation")
    print("=" * 70)

    # ── [1] Load the verified model through existing governance ──────
    print("\n[1] Load verified model through governance")
    registry = ModelRegistry()
    identity = registry.activate_from_manifest()
    bundle = registry.bundle
    check("Model activated through registry governance", registry.is_ready,
          f"version={identity.model_version}")

    # ── [2] Identify the exact model version / checksum ───────────────
    print("\n[2] Identify exact model version and checksum")
    directory = default_model_directory()
    manifest = load_manifest(directory)
    manifest_path = directory / "model_manifest.json"
    manifest_bytes_before = manifest_path.read_bytes()
    check("Identity version matches manifest",
          identity.model_version == manifest.model_version,
          manifest.model_version)
    check("Identity checksum matches manifest",
          identity.artifact_checksum == manifest.artifact_checksum,
          f"{identity.artifact_checksum[:16]}...")
    check("Checksum is a 64-hex SHA-256 digest",
          len(identity.artifact_checksum) == 64
          and all(c in "0123456789abcdef" for c in identity.artifact_checksum))

    # ── Live service (kept running across the whole evaluation) ──────
    from fastapi.testclient import TestClient

    from ml.api.app import app

    with TestClient(app) as client:
        probe_a = client.post("/predict", json=PAYLOAD)
        assert probe_a.status_code == 200, probe_a.text
        before = probe_a.json()
        metrics_a = client.get("/metrics").json()
        health_a = client.get("/health").json()
        ready_a = client.get("/ready").json()
        count_a = history_count()

        # ── [3] Load authorized evaluation data ───────────────────────
        print("\n[3] Load authorized evaluation data (holdout test split)")
        X_test, y_test, split_metadata = load_holdout_test_set()
        check("Holdout test split loaded from approved source",
              len(X_test) > 0 and len(y_test) == len(X_test),
              f"n_test={len(X_test):,}")
        check("Dataset identifier recorded",
              split_metadata["dataset_identifier"] == DATASET_IDENTIFIER,
              split_metadata["dataset_identifier"])
        check("No leakage columns in evaluation features",
              "isFraud" not in X_test.columns
              and "TransactionID" not in X_test.columns)

        # ── [4] Run evaluation ────────────────────────────────────────
        print("\n[4] Run offline evaluation")
        config = EvaluationConfig(
            false_negative_cost=150.0,
            false_positive_cost=2.0,
            min_recall=0.80,
        )
        config.validate()
        report = build_report(
            bundle=bundle,
            identity=identity,
            X=X_test,
            y=y_test,
            config=config,
            dataset_metadata=split_metadata,
        )
        check("Evaluation report produced",
              report.report_scope == REPORT_SCOPE,
              f"scope={report.report_scope}")

        # ── [5] Classification metrics ────────────────────────────────
        print("\n[5] Classification metrics")
        cm = report.classification_metrics
        check("Confusion counts computed",
              all(isinstance(cm[k], int) and cm[k] >= 0
                  for k in ("tp", "tn", "fp", "fn")),
              f"TP={cm['tp']:,} TN={cm['tn']:,} "
              f"FP={cm['fp']:,} FN={cm['fn']:,}")
        check("Precision/recall/F1 within [0, 1]",
              all(0.0 <= cm[k] <= 1.0
                  for k in ("precision", "recall", "f1")),
              f"P={cm['precision']:.4f} R={cm['recall']:.4f} "
              f"F1={cm['f1']:.4f}")
        check("Classification uses the observed production threshold",
              cm["threshold"] == manifest.threshold
              and "NOT modified" in cm["threshold_source"],
              f"threshold={cm['threshold']}")

        # ── [6] Ranking metrics ───────────────────────────────────────
        print("\n[6] Ranking metrics (ROC-AUC, PR-AUC)")
        rm = report.ranking_metrics
        check("ROC-AUC and PR-AUC computed",
              rm is not None
              and 0.0 <= rm["roc_auc"] <= 1.0
              and 0.0 <= rm["pr_auc"] <= 1.0,
              f"ROC-AUC={rm['roc_auc']:.4f} PR-AUC={rm['pr_auc']:.4f}"
              if rm else "unavailable")

        # ── [7] Threshold sweep ────────────────────────────────────────
        print("\n[7] Threshold sweep")
        sweep = report.threshold_analysis
        thresholds = [p["threshold"] for p in sweep]
        check("Sweep covers a deterministic ascending grid",
              len(sweep) > 0 and thresholds == sorted(thresholds),
              f"{len(sweep)} points, "
              f"{thresholds[0]:.2f}..{thresholds[-1]:.2f}")
        check("Every sweep point has P/R/F1",
              all({"precision", "recall", "f1"} <= set(p) for p in sweep))

        # ── [8] Recommendations ────────────────────────────────────────
        print("\n[8] Threshold recommendations")
        recs = {r["strategy"]: r for r in report.recommendations}
        check("All four recommendation strategies present",
              set(recs) == {"max_f1", "min_cost", "min_recall",
                            "min_precision"},
              f"{len(recs)} strategies")
        check("max_f1 recommendation available",
              recs["max_f1"]["available"] is True,
              f"threshold={recs['max_f1']['threshold']:.2f} "
              f"F1={recs['max_f1']['f1']:.4f}")
        check("min_cost recommendation available (costs configured)",
              recs["min_cost"]["available"] is True,
              f"threshold={recs['min_cost']['threshold']:.2f}")
        min_recall_rec = recs["min_recall"]
        check("min_recall recommendation satisfies the constraint",
              (min_recall_rec["available"] is True
               and min_recall_rec["recall"] >= 0.80),
              f"recall={min_recall_rec['recall']:.4f}")
        check("min_precision honestly reports 'not configured'",
              recs["min_precision"]["available"] is False
              and "not configured" in recs["min_precision"]["reason"])

        # ── [9] Calibration metrics ────────────────────────────────────
        print("\n[9] Calibration metrics")
        cal = report.calibration_metrics
        bins = cal["reliability_bins"]
        check("Brier score and reliability bins computed",
              0.0 <= cal["brier_score"] <= 1.0 and len(bins) > 0,
              f"brier={cal['brier_score']:.4f}, {len(bins)} non-empty bins")
        check("Reliability bins cover every evaluation sample",
              sum(b["count"] for b in bins) == len(y_test))
        check("Raw probabilities only (no silent recalibration)",
              cal["recalibration_applied"] is False
              and cal["probability_type"] == "raw_model_probability")

        # ── [10] Evaluation-only labelling ─────────────────────────────
        print("\n[10] Recommendations clearly marked evaluation-only")
        check("Report-level disclaimer present",
              "EVALUATION / RECOMMENDATION ONLY" in report.disclaimer
              and "DO NOT automatically change production"
              in report.disclaimer)
        check("Every recommendation carries the disclaimer",
              all("EVALUATION / RECOMMENDATION ONLY"
                  in r["classification"] for r in report.recommendations))
        report_text = json.dumps(report.to_dict())
        check("Report output contains no raw/customer/secret fields",
              not any(field in report_text for field in (
                  "card1", "TransactionID", "ip_address",
                  "device_fingerprint", "password", "secret", "token",
              )))

        # ── [11] Production threshold unchanged ───────────────────────
        print("\n[11] Production threshold unchanged")
        check("Report observes the production threshold",
              report.production_threshold == manifest.threshold,
              f"{report.production_threshold}")
        check("Registry bundle threshold untouched",
              bundle.threshold == manifest.threshold)
        check("Manifest file byte-identical after evaluation",
              manifest_path.read_bytes() == manifest_bytes_before)
        artifact_checksum = compute_checksum(
            directory / manifest.artifact_filename
        )
        check("Model artifact checksum unchanged",
              artifact_checksum == manifest.artifact_checksum,
              f"{artifact_checksum[:16]}...")

        # ── [12] Live /predict behavior unchanged ─────────────────────
        print("\n[12] Live /predict behavior unchanged by evaluation")
        # Snapshot the history count after the evaluation but before
        # the second probe so check [15] can isolate evaluation writes.
        count_mid = history_count()
        probe_b = client.post("/predict", json=PAYLOAD)
        assert probe_b.status_code == 200, probe_b.text
        after = probe_b.json()
        differing = [
            f for f in DETERMINISTIC_FIELDS
            if after[f] != before[f]
        ]
        check("Identical probe returns identical decisions",
              not differing,
              f"prob={after['fraud_probability']:.4f} "
              f"decision={after['decision']}" if not differing
              else f"differing={differing}")

        # ── [13] /health and /ready expose the same active model ───────
        print("\n[13] /health and /ready expose the same active model")
        health_b = client.get("/health").json()
        ready_b = client.get("/ready").json()
        for label, payload in (("health", health_b), ("ready", ready_b)):
            check(f"/{label} reports the manifest model",
                  payload["status"] == "ready"
                  and payload["model_version"] == manifest.model_version
                  and payload["model_identity"]["artifact_checksum"]
                  == manifest.artifact_checksum,
                  f"version={payload.get('model_version')}")
        check("Active model identical before and after evaluation",
              health_a["model_version"] == health_b["model_version"]
              and ready_a["model_identity"] == ready_b["model_identity"])

        # ── [14] Monitoring remains correct ────────────────────────────
        print("\n[14] Monitoring remains correct")
        metrics_b = client.get("/metrics").json()
        check("Counters incremented only by the two probes",
              metrics_b["total_requests"] == 2
              and metrics_b["successful_predictions"] == 2
              and metrics_b["failed_predictions"] == 0,
              f"total={metrics_b['total_requests']} "
              f"success={metrics_b['successful_predictions']}")
        check("Evaluation added no monitoring increments",
              metrics_b["total_requests"]
              == metrics_a["total_requests"] + 1,
              f"{metrics_a['total_requests']} -> "
              f"{metrics_b['total_requests']} (probe B only)")
        check("Monitoring reports the active model identity",
              metrics_b["model_identity"] is not None
              and metrics_b["model_identity"]["artifact_checksum"]
              == manifest.artifact_checksum
              and "drift" in metrics_b)

        # ── [15] No transaction audit records created ─────────────────
        print("\n[15] No transaction audit records created by evaluation")
        check("Evaluation wrote no history records",
              count_mid == count_a,
              f"{count_a} -> {count_mid} during evaluation")
        count_final = history_count()
        check("History contains only the two probe transactions",
              count_final == count_a + 1,
              f"total={count_final} (2 probes expected)")

    # ── Persist the evaluation report for operator inspection ────────
    report_path = _TMP_DIR / "step47_e2e_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nEvaluation report written to: {report_path}")

    # ── Summary ────────────────────────────────────────────────────────
    passed = sum(1 for _, s in RESULTS if s == "PASS")
    failed = len(RESULTS) - passed
    print("\n" + "=" * 70)
    print(f"RESULT: {passed} passed, {failed} failed "
          f"({len(RESULTS)} checks)")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
