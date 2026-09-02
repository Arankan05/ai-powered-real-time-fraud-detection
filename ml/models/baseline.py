"""Logistic Regression baseline for fraud detection.

Implements the first ML model in the training pipeline defined in
``docs/ml-architecture.md``:

    Feature Engineering → Train/Test Split → Model Training → Evaluation

Algorithm
---------
Logistic Regression with ``class_weight='balanced'`` to handle the
~1:27.6 class imbalance without resampling.  This is the recommended
approach for a deterministic, reproducible baseline.

Class-imbalance strategy: ``class_weight='balanced'``
  - Automatically adjusts sample weights inversely proportional to
    class frequencies: weight = n_samples / (n_classes * n_class_count).
  - For the training set (~3.51 % fraud): fraud class weight ≈ 13.9×.
  - Reproducible — no random resampling, no synthetic data, no
    dependency on sampling order.
  - Conservative starting point before attempting SMOTE or XGBoost.

Preprocessing
-------------
- Numeric features: ``StandardScaler`` fitted on training data only.
- Categorical: ``device_fingerprint`` (object dtype) is label-encoded
  with ``LabelEncoder``.  ``merchant_category`` is already int8.
- Fitted transformers are stored alongside the model coefficients so the
  same pipeline can be applied at inference without re-fitting.

Metrics reported (spec § 2):
  - Precision (fraud class, label=1)
  - Recall (fraud class, label=1)
  - AUC-ROC

Performance targets (``docs/ml-architecture.md``):
  - Precision ≥ 0.70
  - Recall   ≥ 0.80
  - AUC-ROC  ≥ 0.85

Random seed: 42 (used for LogisticRegression solver reproducibility).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Constants ─────────────────────────────────────────────────────────

RANDOM_SEED = 42
FRAUD_LABEL = 1
TARGET_PRECISION = 0.70
TARGET_RECALL = 0.80
TARGET_AUC_ROC = 0.85

# Categorical columns that need label encoding
_CATEGORICAL_COLS = ("device_fingerprint",)


# ── Result containers ────────────────────────────────────────────────


@dataclass
class PreprocessingArtifacts:
    """Transformers fitted on training data.

    Attributes:
        scaler: StandardScaler fitted on numeric training columns.
        label_encoders: Dict mapping column name → fitted LabelEncoder.
        numeric_cols: List of numeric column names scaled by *scaler*.
        categorical_cols: List of categorical columns that were encoded.
    """

    scaler: StandardScaler
    label_encoders: dict[str, LabelEncoder]
    numeric_cols: list[str]
    categorical_cols: list[str]


@dataclass
class BaselineMetrics:
    """Evaluation metrics for the baseline model.

    Attributes:
        precision: Precision on the fraud class (label=1).
        recall: Recall on the fraud class (label=1).
        auc_roc: Area under the ROC curve.
        avg_precision: Average precision (area under PR curve).
        n_test: Number of test samples.
        n_fraud_test: Number of fraudulent samples in test set.
        meets_precision_target: Whether precision ≥ 0.70.
        meets_recall_target: Whether recall ≥ 0.80.
        meets_auc_target: Whether AUC-ROC ≥ 0.85.
    """

    precision: float
    recall: float
    auc_roc: float
    avg_precision: float
    n_test: int
    n_fraud_test: int

    @property
    def meets_precision_target(self) -> bool:
        return self.precision >= TARGET_PRECISION

    @property
    def meets_recall_target(self) -> bool:
        return self.recall >= TARGET_RECALL

    @property
    def meets_auc_target(self) -> bool:
        return self.auc_roc >= TARGET_AUC_ROC


@dataclass
class TrainingResult:
    """Complete output of a baseline training run.

    Attributes:
        model: Fitted LogisticRegression.
        preprocessing: Fitted preprocessing artifacts.
        metrics: Evaluation metrics on the test set.
        feature_names: Ordered list of input feature names.
        class_weight: Class-weight strategy used.
        training_rows: Number of training samples.
    """

    model: LogisticRegression
    preprocessing: PreprocessingArtifacts
    metrics: BaselineMetrics
    feature_names: list[str]
    class_weight: str
    training_rows: int


# ── Preprocessing ─────────────────────────────────────────────────────


def fit_preprocessing(X_train: pd.DataFrame) -> PreprocessingArtifacts:
    """Fit all preprocessing transformers on the training data.

    Fits:
    - ``StandardScaler`` on all numeric (non-object) columns.
    - ``LabelEncoder`` on each categorical (object-dtype) column listed
      in ``_CATEGORICAL_COLS`` that is present in *X_train*.

    Args:
        X_train: Training feature matrix.

    Returns:
        :class:`PreprocessingArtifacts` with fitted transformers.
    """
    numeric_cols = [
        c for c in X_train.columns if X_train[c].dtype != object
    ]
    categorical_cols = [
        c for c in _CATEGORICAL_COLS if c in X_train.columns
    ]

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols].values)

    label_encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X_train[col].astype(str).values)
        label_encoders[col] = le

    return PreprocessingArtifacts(
        scaler=scaler,
        label_encoders=label_encoders,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def apply_preprocessing(
    X: pd.DataFrame,
    artifacts: PreprocessingArtifacts,
    *,
    unseen_label: int = 0,
) -> np.ndarray:
    """Apply fitted preprocessing to a feature DataFrame.

    Args:
        X: Feature DataFrame to transform.
        artifacts: Fitted preprocessing artifacts from
                   :func:`fit_preprocessing`.
        unseen_label: Encoded integer for unseen categorical labels.
                      Defaults to 0.

    Returns:
        2-D float64 NumPy array ready for model input.
    """
    X_out = X.copy()

    # Label-encode categorical columns
    for col, le in artifacts.label_encoders.items():
        if col in X_out.columns:
            known_map = {v: int(i) for i, v in enumerate(le.classes_)}
            X_out[col] = (
                X_out[col].astype(str).map(known_map).fillna(unseen_label).astype(int)
            )

    # Scale numeric columns with the fitted scaler
    X_out[artifacts.numeric_cols] = artifacts.scaler.transform(
        X_out[artifacts.numeric_cols].values
    )

    return X_out.values.astype(np.float64)


# ── Training ─────────────────────────────────────────────────────────


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    class_weight: str = "balanced",
    max_iter: int = 5_000,
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
) -> TrainingResult:
    """Fit the Logistic Regression baseline and evaluate on the test set.

    Preprocessing is fitted **only on training data** and then applied
    to both training and test data — no data leakage from test to fit.

    Args:
        X_train: Training feature matrix (must not contain ``isFraud``).
        y_train: Training target series.
        X_test: Test feature matrix.
        y_test: Test target series.
        class_weight: Weight strategy passed to LogisticRegression.
                      Default: ``'balanced'`` — handles 1:27.6 imbalance.
        max_iter: Maximum solver iterations.
        random_state: Random seed for reproducibility.
        verbose: Print progress messages.

    Returns:
        :class:`TrainingResult` with model, preprocessing, and metrics.

    Raises:
        ValueError: If ``isFraud`` or ``TransactionID`` appear in
                    *X_train* or *X_test*.
    """
    _guard_against_leakage(X_train, X_test)

    if verbose:
        print(f"[baseline] Training set: {len(X_train):,} rows, {X_train.shape[1]} features")
        print(f"[baseline] Test set:     {len(X_test):,} rows")
        print(f"[baseline] Fraud in train: {int(y_train.sum()):,} ({y_train.mean():.2%})")
        print(f"[baseline] Class weight:   {class_weight}")

    # ── Preprocessing ────────────────────────────────────────────────
    if verbose:
        print("[baseline] Fitting preprocessing on training data...")
    preprocessing = fit_preprocessing(X_train)
    X_train_t = apply_preprocessing(X_train, preprocessing)
    X_test_t = apply_preprocessing(X_test, preprocessing)

    if verbose:
        print(f"[baseline] Preprocessed shape: {X_train_t.shape}")

    # ── Model ─────────────────────────────────────────────────────────
    if verbose:
        print("[baseline] Training Logistic Regression...")
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_train_t, y_train.values)

    # ── Evaluation ────────────────────────────────────────────────────
    if verbose:
        print("[baseline] Evaluating on test set...")
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

    return TrainingResult(
        model=model,
        preprocessing=preprocessing,
        metrics=metrics,
        feature_names=list(X_train.columns),
        class_weight=class_weight,
        training_rows=len(X_train),
    )


# ── Leakage guard ─────────────────────────────────────────────────────

_FORBIDDEN = frozenset({"isFraud", "TransactionID"})


def _guard_against_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    for name, df in [("X_train", X_train), ("X_test", X_test)]:
        found = _FORBIDDEN & set(df.columns)
        if found:
            raise ValueError(
                f"Forbidden columns found in {name}: {sorted(found)}. "
                f"Remove isFraud and TransactionID before training."
            )


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    """CLI: run the full baseline training pipeline."""
    print("Baseline Logistic Regression — Training Pipeline")
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

    # ── Model training ────────────────────────────────────────────────
    print("Step 3/3 — Training Logistic Regression...")
    result = train_baseline(
        X_train=split.X_train,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
    )
    print()

    # ── Results ───────────────────────────────────────────────────────
    print("=" * 60)
    print("Results")
    print("=" * 60)
    m = result.metrics

    def _status(value: float, target: float) -> str:
        return "MEETS TARGET" if value >= target else "below target"

    print(f"\n  Precision  : {m.precision:.4f}  (target ≥ {TARGET_PRECISION})  [{_status(m.precision, TARGET_PRECISION)}]")
    print(f"  Recall     : {m.recall:.4f}  (target ≥ {TARGET_RECALL})  [{_status(m.recall, TARGET_RECALL)}]")
    print(f"  AUC-ROC    : {m.auc_roc:.4f}  (target ≥ {TARGET_AUC_ROC})  [{_status(m.auc_roc, TARGET_AUC_ROC)}]")
    print(f"  Avg Prec   : {m.avg_precision:.4f}")
    print(f"\n  Test rows      : {m.n_test:,}")
    print(f"  Fraud in test  : {m.n_fraud_test:,} ({m.n_fraud_test / m.n_test:.2%})")
    print(f"  Training rows  : {result.training_rows:,}")
    print(f"  Features       : {len(result.feature_names)}")
    print()

    meets_all = (
        m.meets_precision_target
        and m.meets_recall_target
        and m.meets_auc_target
    )
    if meets_all:
        print("All three performance targets met.")
    else:
        missed = []
        if not m.meets_precision_target:
            missed.append(f"Precision={m.precision:.4f} < {TARGET_PRECISION}")
        if not m.meets_recall_target:
            missed.append(f"Recall={m.recall:.4f} < {TARGET_RECALL}")
        if not m.meets_auc_target:
            missed.append(f"AUC-ROC={m.auc_roc:.4f} < {TARGET_AUC_ROC}")
        print(f"Targets not met: {', '.join(missed)}")
        print("Baseline established. XGBoost step follows.")

    print("\nDone.")


if __name__ == "__main__":
    main()
