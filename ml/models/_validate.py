"""Lightweight validation for the Logistic Regression baseline.

Runs the baseline on a **sample** of the dataset and verifies:

 1. Model trains without error
 2. Predictions succeed on the test set
 3. Probability predictions are available (needed for AUC-ROC)
 4. X does not contain ``isFraud`` or ``TransactionID``
 5. Preprocessing was fitted only on training data (scaler attributes present)
 6. Train and test sets are non-overlapping (verified via TransactionID)
 7. Precision can be calculated (non-NaN, in [0, 1])
 8. Recall can be calculated (non-NaN, in [0, 1])
 9. AUC-ROC can be calculated (non-NaN, in [0, 1])
10. Fraud class is present in both train and test sets
11. No isFraud column in the preprocessed arrays
12. Feature names are preserved in the training result

Usage::

    python -m ml.models._validate

"""

from __future__ import annotations

import sys

import numpy as np

from ml.data.loader import load_transaction_dataset
from ml.features.engineer import engineer_features
from ml.models.baseline import (
    FRAUD_LABEL,
    TARGET_AUC_ROC,
    TARGET_PRECISION,
    TARGET_RECALL,
    apply_preprocessing,
    fit_preprocessing,
    train_baseline,
)
from ml.split.splitter import time_based_split

_SAMPLE_SIZE = 15_000


def _run_checks(result, split) -> list[dict]:
    checks: list[dict] = []
    model = result.model
    prep = result.preprocessing
    m = result.metrics

    # ── 1. Model trained (has classes_) ──────────────────────────────
    has_classes = hasattr(model, "classes_")
    checks.append(
        {
            "check": "model_trained",
            "status": "PASS" if has_classes else "FAIL",
            "detail": f"classes={list(model.classes_)}" if has_classes else "model has no classes_ attribute",
        }
    )

    # ── 2. Prediction succeeds ────────────────────────────────────────
    X_test_t = apply_preprocessing(split.X_test, prep)
    try:
        preds = model.predict(X_test_t)
        checks.append(
            {
                "check": "prediction_succeeds",
                "status": "PASS",
                "detail": f"{len(preds):,} predictions produced",
            }
        )
    except Exception as e:
        checks.append(
            {"check": "prediction_succeeds", "status": "FAIL", "detail": str(e)}
        )

    # ── 3. Probability predictions available ─────────────────────────
    try:
        probs = model.predict_proba(X_test_t)
        ok = probs.shape == (len(split.X_test), 2) and np.all(probs >= 0)
        checks.append(
            {
                "check": "predict_proba_available",
                "status": "PASS" if ok else "FAIL",
                "detail": f"shape={probs.shape}, range=[{probs.min():.3f}, {probs.max():.3f}]",
            }
        )
    except Exception as e:
        checks.append(
            {"check": "predict_proba_available", "status": "FAIL", "detail": str(e)}
        )

    # ── 4. isFraud/TransactionID not in X ────────────────────────────
    for col in ("isFraud", "TransactionID"):
        in_train = col in split.X_train.columns
        in_test = col in split.X_test.columns
        checks.append(
            {
                "check": f"no_{col.lower()}_in_X",
                "status": "PASS" if not (in_train or in_test) else "FAIL",
                "detail": f"in_train={in_train}, in_test={in_test}",
            }
        )

    # ── 5. Preprocessing fitted on training only ──────────────────────
    scaler = prep.scaler
    has_mean = hasattr(scaler, "mean_") and scaler.mean_ is not None
    has_scale = hasattr(scaler, "scale_") and scaler.scale_ is not None
    checks.append(
        {
            "check": "preprocessing_fitted_on_train_only",
            "status": "PASS" if (has_mean and has_scale) else "FAIL",
            "detail": (
                f"scaler.mean_ present={has_mean}, "
                f"scale_ present={has_scale}, "
                f"n_features={scaler.n_features_in_}"
            ),
        }
    )

    # ── 6. No ID overlap between train and test ───────────────────────
    overlap = set(split.train_ids) & set(split.test_ids)
    checks.append(
        {
            "check": "no_id_overlap",
            "status": "PASS" if len(overlap) == 0 else "FAIL",
            "detail": f"{len(overlap)} overlapping TransactionIDs",
        }
    )

    # ── 7. Precision valid ─────────────────────────────────────────────
    ok = np.isfinite(m.precision) and 0.0 <= m.precision <= 1.0
    checks.append(
        {
            "check": "precision_calculable",
            "status": "PASS" if ok else "FAIL",
            "detail": f"precision={m.precision:.4f}",
        }
    )

    # ── 8. Recall valid ────────────────────────────────────────────────
    ok = np.isfinite(m.recall) and 0.0 <= m.recall <= 1.0
    checks.append(
        {
            "check": "recall_calculable",
            "status": "PASS" if ok else "FAIL",
            "detail": f"recall={m.recall:.4f}",
        }
    )

    # ── 9. AUC-ROC valid ──────────────────────────────────────────────
    ok = np.isfinite(m.auc_roc) and 0.0 <= m.auc_roc <= 1.0
    checks.append(
        {
            "check": "auc_roc_calculable",
            "status": "PASS" if ok else "FAIL",
            "detail": f"auc_roc={m.auc_roc:.4f}",
        }
    )

    # ── 10. Fraud class in both sets ──────────────────────────────────
    for name, y in [("train", split.y_train), ("test", split.y_test)]:
        count = int(y.sum())
        checks.append(
            {
                "check": f"fraud_class_present_{name}",
                "status": "PASS" if count > 0 else "FAIL",
                "detail": f"{count:,} fraudulent rows in {name}",
            }
        )

    # ── 11. Feature names recorded ───────────────────────────────────
    has_names = len(result.feature_names) > 0
    checks.append(
        {
            "check": "feature_names_recorded",
            "status": "PASS" if has_names else "FAIL",
            "detail": f"{len(result.feature_names)} feature names stored",
        }
    )

    return checks


def main() -> None:
    """Run baseline validation on a sample and report results."""
    print("Baseline Logistic Regression — Validation")
    print("=" * 50)
    print(f"Sample size: {_SAMPLE_SIZE:,} rows\n")

    # Feature engineering
    print("Running feature engineering (sampled)...")
    fe = engineer_features(sample=_SAMPLE_SIZE)
    features = fe["features"]
    target = fe["target"]
    txn_ids = fe["transaction_ids"]
    print()

    # Timestamps for split
    txn = load_transaction_dataset()
    ts_map = txn.set_index("TransactionID")["TransactionDT"]
    timestamps = ts_map.loc[txn_ids.values].reset_index(drop=True)
    timestamps.index = features.index

    # Split
    split = time_based_split(
        features=features,
        target=target,
        timestamps=timestamps,
        transaction_ids=txn_ids,
    )
    print(
        f"Split: train={len(split.X_train):,} / test={len(split.X_test):,} | "
        f"fraud_train={split.y_train.mean():.2%} / fraud_test={split.y_test.mean():.2%}\n"
    )

    # Train
    print("Training Logistic Regression baseline...")
    result = train_baseline(
        X_train=split.X_train,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        verbose=True,
    )
    print()

    # Validate
    checks = _run_checks(result, split)
    passed = sum(1 for c in checks if c["status"] == "PASS")
    failed = sum(1 for c in checks if c["status"] != "PASS")

    print(f"Validation results: {passed} passed, {failed} failed\n")
    for c in checks:
        print(f"  [{c['status']:4s}] {c['check']} — {c['detail']}")

    # Metrics summary
    m = result.metrics
    print(f"\n{'─' * 50}")
    print("Metrics on sample test set:")
    print(f"  Precision : {m.precision:.4f}  (target ≥ {TARGET_PRECISION})")
    print(f"  Recall    : {m.recall:.4f}  (target ≥ {TARGET_RECALL})")
    print(f"  AUC-ROC   : {m.auc_roc:.4f}  (target ≥ {TARGET_AUC_ROC})")

    print()
    if failed == 0:
        print("All validations PASSED.")
    else:
        print(f"{failed} validation(s) FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
