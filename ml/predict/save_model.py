"""CLI: train the tuned XGBoost model and save artifacts to disk.

Reuses the exact tuning configuration from ``ml.models.tune_xgboost``
(best config: max_depth=4, lr=0.05, n_estimators=500, threshold=0.50)
and serialises the complete model bundle via :func:`save_bundle`.

Step 46: After saving the bundle, a model manifest (``model_manifest.json``)
is generated with SHA-256 checksum, feature schema, and version metadata.

Usage::

    python -m ml.predict.save_model

The saved ``.joblib`` file is excluded from version control
(``.gitignore``: ``ml/models/*.joblib``).
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from ml.models.baseline import RANDOM_SEED, fit_preprocessing
from ml.models.tune_xgboost import (
    _FIXED_PARAMS,
    _INNER_VAL_FRACTION,
    _analyze_thresholds,
)
from ml.predict.bundle import ModelBundle, save_bundle


def main() -> None:
    """Train the tuned XGBoost configuration and save the bundle."""
    print("Save Tuned Model — Training & Serialisation")
    print("=" * 60)
    print()

    # ── Load data ─────────────────────────────────────────────────────
    print("Step 1/4 — Loading data...")
    from ml.features.engineer import engineer_features
    from ml.data.loader import load_transaction_dataset
    from ml.split.splitter import time_based_split

    fe = engineer_features()
    features = fe["features"]
    target = fe["target"]
    txn_ids = fe["transaction_ids"]
    feature_names = list(features.columns)

    txn = load_transaction_dataset()
    ts_map = txn.set_index("TransactionID")["TransactionDT"]
    timestamps = ts_map.loc[txn_ids.values].reset_index(drop=True)
    timestamps.index = features.index

    split = time_based_split(
        features=features, target=target,
        timestamps=timestamps, transaction_ids=txn_ids,
    )
    X_train = split.X_train
    y_train = split.y_train
    print(f"  Training: {len(X_train):,} rows")
    print(f"  Features: {len(feature_names)} ({feature_names[:3]}...)")
    print()

    # ── Compute threshold on inner validation split ───────────────────
    print("Step 2/4 — Computing threshold on validation split...")
    n_train = len(X_train)
    val_cut = int(n_train * (1.0 - _INNER_VAL_FRACTION))

    from ml.models.baseline import apply_preprocessing, _guard_against_leakage

    X_inner = X_train.iloc[:val_cut].reset_index(drop=True)
    y_inner = y_train.iloc[:val_cut].reset_index(drop=True)
    X_val = X_train.iloc[val_cut:].reset_index(drop=True)
    y_val = y_train.iloc[val_cut:].reset_index(drop=True)

    _guard_against_leakage(X_inner, X_val)

    prep_inner = fit_preprocessing(X_inner)
    X_inner_t = apply_preprocessing(X_inner, prep_inner)
    X_val_t = apply_preprocessing(X_val, prep_inner)

    n_pos = int(y_inner.sum())
    n_neg = len(y_inner) - n_pos
    spw_inner = n_neg / max(n_pos, 1)

    # Train the best config on inner split to get val probabilities
    best_params = {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 500}
    full_params = {**_FIXED_PARAMS, **best_params}

    temp_model = XGBClassifier(
        n_estimators=full_params["n_estimators"],
        max_depth=full_params["max_depth"],
        learning_rate=full_params["learning_rate"],
        subsample=full_params["subsample"],
        colsample_bytree=full_params["colsample_bytree"],
        reg_alpha=full_params["reg_alpha"],
        reg_lambda=full_params["reg_lambda"],
        min_child_weight=full_params["min_child_weight"],
        gamma=full_params["gamma"],
        scale_pos_weight=spw_inner,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    temp_model.fit(X_inner_t, y_inner.values)
    val_prob = temp_model.predict_proba(X_val_t)[:, 1]

    selected_threshold, _ = _analyze_thresholds(y_val.values, val_prob)
    print(f"  Selected threshold: {selected_threshold:.2f}")
    print()

    # ── Retrain on full training set ──────────────────────────────────
    print("Step 3/4 — Training final model on full training set...")
    prep_full = fit_preprocessing(X_train)
    X_train_t = apply_preprocessing(X_train, prep_full)

    n_pos_full = int(y_train.sum())
    n_neg_full = len(y_train) - n_pos_full
    spw_full = n_neg_full / max(n_pos_full, 1)

    final_model = XGBClassifier(
        n_estimators=full_params["n_estimators"],
        max_depth=full_params["max_depth"],
        learning_rate=full_params["learning_rate"],
        subsample=full_params["subsample"],
        colsample_bytree=full_params["colsample_bytree"],
        reg_alpha=full_params["reg_alpha"],
        reg_lambda=full_params["reg_lambda"],
        min_child_weight=full_params["min_child_weight"],
        gamma=full_params["gamma"],
        scale_pos_weight=spw_full,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    final_model.fit(X_train_t, y_train.values)
    print(f"  Trained on {len(X_train):,} rows, scale_pos_weight={spw_full:.2f}")
    print()

    # ── Save ──────────────────────────────────────────────────────────
    print("Step 4/4 — Saving model bundle...")
    bundle = ModelBundle(
        model=final_model,
        preprocessing=prep_full,
        threshold=selected_threshold,
        feature_names=feature_names,
        model_version="fraud-xgb-v1.0.0",
    )
    saved_path = save_bundle(bundle)
    size_mb = saved_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {saved_path}")
    print(f"  Size:  {size_mb:.1f} MB")
    print(f"  Version: {bundle.model_version}")
    print(f"  Threshold: {bundle.threshold:.2f}")
    print(f"  Features: {bundle.n_features}")
    print()

    # ── Generate manifest (Step 46) ─────────────────────────────────
    print("Step 4b — Generating model manifest...")
    from ml.predict.integrity import build_manifest, save_manifest

    manifest = build_manifest(
        model_name="fraud-xgb",
        model_version=bundle.model_version,
        artifact_path=saved_path,
        n_features=bundle.n_features,
        threshold=bundle.threshold,
    )
    manifest_path = save_manifest(manifest)
    print(f"  Manifest: {manifest_path}")
    print(f"  Checksum: {manifest.artifact_checksum[:12]}...")
    print(f"  Schema:   {manifest.feature_schema_version}")
    print()

    # ── Verify round-trip ────────────────────────────────────────────
    from ml.predict.bundle import load_bundle
    loaded = load_bundle(saved_path)
    assert loaded.model_version == bundle.model_version
    assert loaded.threshold == bundle.threshold
    assert loaded.feature_names == bundle.feature_names
    print("  Round-trip verification: PASSED")

    # Verify manifest integrity
    from ml.predict.integrity import verify_artifact, load_manifest
    loaded_manifest = load_manifest()
    verify_artifact(loaded_manifest)
    print(f"  Manifest integrity: VERIFIED")
    print(f"  Active version: {loaded_manifest.model_version}")
    print(f"  Checksum: {loaded_manifest.artifact_checksum[:12]}...")

    print("\nDone.")


if __name__ == "__main__":
    main()
