"""XGBoost fraud detection model.

Implements the second ML model in the training pipeline defined in
``docs/ml-architecture.md``:

    Feature Engineering → Train/Test Split → Model Training → Evaluation

Algorithm
---------
XGBoost (gradient-boosted decision trees) — the model specified in the
architecture documentation (L159) for improved performance after the
Logistic Regression baseline.

XGBoost advantages over Logistic Regression for this task:
  - Captures non-linear feature interactions automatically.
  - Handles severe class imbalance via ``scale_pos_weight``.
  - Robust to feature scale differences (tree-based).
  - Better suited to the obfuscated V-features and count features.

Class-imbalance strategy: ``scale_pos_weight``
  - Set to the ratio of negative to positive samples in the training set.
  - For ~3.51 % fraud: scale_pos_weight ≈ 27.5×.
  - Equivalent to up-weighting the positive class during gradient
    computation — avoids resampling and keeps the approach reproducible.

Preprocessing
-------------
Reuses ``fit_preprocessing`` / ``apply_preprocessing`` from
``ml.models.baseline`` — the same StandardScaler + LabelEncoder
pipeline fitted on training data only.

XGBoost is scale-invariant, so the StandardScaler is technically
unnecessary for prediction quality.  It is retained for consistency
with the existing pipeline and because the same preprocessed features
will later be shared with SHAP explainability.

Initial configuration (no tuning in this step):
  - n_estimators: 500
  - max_depth: 6
  - learning_rate: 0.05
  - subsample: 0.8
  - colsample_bytree: 0.8
  - reg_alpha: 0.1
  - reg_lambda: 1.0
  - random_state: 42

Metrics reported (spec § 2):
  - Precision (fraud class, label=1)
  - Recall (fraud class, label=1)
  - AUC-ROC

Performance targets (``docs/ml-architecture.md``):
  - Precision ≥ 0.70
  - Recall   ≥ 0.80
  - AUC-ROC  ≥ 0.85
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from ml.models.baseline import (
    FRAUD_LABEL,
    RANDOM_SEED,
    TARGET_AUC_ROC,
    TARGET_PRECISION,
    TARGET_RECALL,
    BaselineMetrics,
    PreprocessingArtifacts,
    _guard_against_leakage,
    apply_preprocessing,
    fit_preprocessing,
)

# ── XGBoost default hyperparameters (initial, untuned) ───────────────

XGB_DEFAULT_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
    "gamma": 0.0,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "tree_method": "hist",
}


# ── Training ─────────────────────────────────────────────────────────


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    scale_pos_weight: float | None = None,
    xgb_params: dict | None = None,
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict:
    """Fit XGBoost and evaluate on the test set.

    Reuses the baseline preprocessing (fitted on training data only).
    ``scale_pos_weight`` defaults to the ratio of negative to positive
    samples in the training set.

    Args:
        X_train: Training feature matrix (must not contain isFraud).
        y_train: Training target series.
        X_test: Test feature matrix.
        y_test: Test target series.
        scale_pos_weight: Override for the class-weight ratio.
                          Defaults to (n_neg / n_pos) from training set.
        xgb_params: Override for XGBoost hyperparameters.
                    Defaults to :data:`XGB_DEFAULT_PARAMS`.
        random_state: Random seed for reproducibility.
        verbose: Print progress messages.

    Returns:
        Dict with keys ``model``, ``preprocessing``, ``metrics``,
        ``feature_names``, ``scale_pos_weight``, ``training_rows``.

    Raises:
        ValueError: If isFraud or TransactionID appear in feature
                    matrices.
    """
    _guard_against_leakage(X_train, X_test)

    # ── Class imbalance ───────────────────────────────────────────────
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if scale_pos_weight is None:
        scale_pos_weight = n_neg / max(n_pos, 1)

    if verbose:
        print(f"[xgboost] Training set: {len(X_train):,} rows, {X_train.shape[1]} features")
        print(f"[xgboost] Test set:     {len(X_test):,} rows")
        print(f"[xgboost] Fraud in train: {n_pos:,} ({y_train.mean():.2%})")
        print(f"[xgboost] scale_pos_weight: {scale_pos_weight:.2f}")

    # ── Preprocessing (reuse from baseline) ───────────────────────────
    if verbose:
        print("[xgboost] Fitting preprocessing on training data...")
    preprocessing = fit_preprocessing(X_train)
    X_train_t = apply_preprocessing(X_train, preprocessing)
    X_test_t = apply_preprocessing(X_test, preprocessing)

    if verbose:
        print(f"[xgboost] Preprocessed shape: {X_train_t.shape}")

    # ── Model ─────────────────────────────────────────────────────────
    params = {**XGB_DEFAULT_PARAMS}
    if xgb_params:
        params.update(xgb_params)

    if verbose:
        print(f"[xgboost] Training XGBoost ({params['n_estimators']} trees)...")

    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],
        scale_pos_weight=scale_pos_weight,
        eval_metric=params["eval_metric"],
        tree_method=params["tree_method"],
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X_train_t, y_train.values)

    # ── Evaluation ────────────────────────────────────────────────────
    if verbose:
        print("[xgboost] Evaluating on test set...")
    y_pred = model.predict(X_test_t)
    y_prob = model.predict_proba(X_test_t)[:, 1]

    metrics = BaselineMetrics(
        precision=float(precision_score(y_test, y_pred, pos_label=FRAUD_LABEL, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, pos_label=FRAUD_LABEL, zero_division=0)),
        auc_roc=float(roc_auc_score(y_test, y_prob)),
        avg_precision=float(average_precision_score(y_test, y_prob, pos_label=FRAUD_LABEL)),
        n_test=len(y_test),
        n_fraud_test=int(y_test.sum()),
    )

    return {
        "model": model,
        "preprocessing": preprocessing,
        "metrics": metrics,
        "feature_names": list(X_train.columns),
        "scale_pos_weight": scale_pos_weight,
        "training_rows": len(X_train),
        "xgb_params": params,
    }


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    """CLI: run the full XGBoost training pipeline and compare with baseline."""
    print("XGBoost Fraud Model — Training Pipeline")
    print("=" * 60)
    print()

    # ── Feature engineering ──────────────────────────────────────────
    print("Step 1/3 — Feature engineering...")
    from ml.features.engineer import engineer_features
    fe_result = engineer_features()
    features = fe_result["features"]
    target = fe_result["target"]
    txn_ids = fe_result["transaction_ids"]
    print(f"  Features: {features.shape[0]:,} rows × {features.shape[1]} cols")
    print()

    # ── Time-based split ─────────────────────────────────────────────
    print("Step 2/3 — Time-based split...")
    from ml.data.loader import load_transaction_dataset
    from ml.split.splitter import time_based_split

    txn = load_transaction_dataset()
    ts_map = txn.set_index("TransactionID")["TransactionDT"]
    timestamps = ts_map.loc[txn_ids.values].reset_index(drop=True)
    timestamps.index = features.index

    split = time_based_split(
        features=features,
        target=target,
        timestamps=timestamps,
        transaction_ids=txn_ids,
    )
    print(f"  Train: {len(split.X_train):,} rows ({split.train_fraction:.0%})")
    print(f"  Test:  {len(split.X_test):,} rows ({split.test_fraction:.0%})")
    print()

    # ── XGBoost training ──────────────────────────────────────────────
    print("Step 3/3 — Training XGBoost...")
    result = train_xgboost(
        X_train=split.X_train,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
    )
    print()

    # ── Results ───────────────────────────────────────────────────────
    m = result["metrics"]
    print("=" * 60)
    print("XGBoost Results")
    print("=" * 60)

    # Baseline comparison
    baseline_precision = 0.0762
    baseline_recall = 0.6681
    baseline_auc_roc = 0.7472

    def _delta(val, base):
        diff = val - base
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.4f}"

    print(f"\n  Precision  : {m.precision:.4f}  (baseline {baseline_precision:.4f}, Δ {_delta(m.precision, baseline_precision)})")
    print(f"  Recall     : {m.recall:.4f}  (baseline {baseline_recall:.4f}, Δ {_delta(m.recall, baseline_recall)})")
    print(f"  AUC-ROC    : {m.auc_roc:.4f}  (baseline {baseline_auc_roc:.4f}, Δ {_delta(m.auc_roc, baseline_auc_roc)})")
    print(f"  Avg Prec   : {m.avg_precision:.4f}")
    print(f"\n  Test rows      : {m.n_test:,}")
    print(f"  Fraud in test  : {m.n_fraud_test:,} ({m.n_fraud_test / m.n_test:.2%})")
    print(f"  Training rows  : {result['training_rows']:,}")
    print(f"  scale_pos_wt   : {result['scale_pos_weight']:.2f}")
    print(f"  Features       : {len(result['feature_names'])}")
    print(f"  Trees          : {result['xgb_params']['n_estimators']}")

    # Target comparison
    print(f"\n{'─' * 60}")
    print("Target Comparison")
    print(f"{'─' * 60}")

    def _status(value, target):
        return "MEETS TARGET" if value >= target else "below target"

    print(f"  Precision  : {m.precision:.4f}  (target ≥ {TARGET_PRECISION})  [{_status(m.precision, TARGET_PRECISION)}]")
    print(f"  Recall     : {m.recall:.4f}  (target ≥ {TARGET_RECALL})  [{_status(m.recall, TARGET_RECALL)}]")
    print(f"  AUC-ROC    : {m.auc_roc:.4f}  (target ≥ {TARGET_AUC_ROC})  [{_status(m.auc_roc, TARGET_AUC_ROC)}]")

    meets_all = m.meets_precision_target and m.meets_recall_target and m.meets_auc_target
    if meets_all:
        print("\nAll three performance targets met.")
    else:
        missed = []
        if not m.meets_precision_target:
            missed.append(f"Precision={m.precision:.4f} < {TARGET_PRECISION}")
        if not m.meets_recall_target:
            missed.append(f"Recall={m.recall:.4f} < {TARGET_RECALL}")
        if not m.meets_auc_target:
            missed.append(f"AUC-ROC={m.auc_roc:.4f} < {TARGET_AUC_ROC}")
        print(f"\nTargets not met: {', '.join(missed)}")
        print("Hyperparameter tuning or feature engineering iteration recommended.")

    print("\nDone.")


if __name__ == "__main__":
    main()
