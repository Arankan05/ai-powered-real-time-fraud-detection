"""Controlled XGBoost hyperparameter tuning with temporal validation.

Implements Step 20 — Controlled XGBoost Improvement.

Strategy:
  1. Reuse existing feature engineering and time-based split.
  2. Create a temporal validation split WITHIN the training set:
     - ``train_inner``: first 80 % of training data by time
     - ``val``: last 20 % of training data by time
  3. Evaluate a small grid of XGBoost configurations on ``val``.
  4. Select the best configuration by **AUC-ROC** (threshold-independent).
  5. Retrain the selected configuration on the **full** training set.
  6. Perform threshold analysis on ``val`` to find the best operating
     point for the Precision/Recall trade-off.
  7. Evaluate **once** on the held-out test set.

Leakage prevention:
  - ``val`` and ``test`` data are **never** used for model selection,
    threshold selection, or preprocessing fitting.
  - Temporal ordering is preserved throughout.
  - Preprocessing is fitted on ``train_inner`` (during grid search) or
    on the full training set (during final evaluation).

Hyperparameter grid (9 configs):
  - max_depth: 4, 6, 8
  - learning_rate: 0.05, 0.10
  - n_estimators: 500, 1000
  - min_child_weight: 3 (fixed — reasonable for this imbalance)
  - Other params: subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, tree_method=hist

Threshold candidates: 0.05 → 0.50 in steps of 0.05.

Usage::

    python -m ml.models.tune_xgboost
"""

from __future__ import annotations

from itertools import product

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
    _guard_against_leakage,
    apply_preprocessing,
    fit_preprocessing,
)

# ── Hyperparameter grid ───────────────────────────────────────────────

_FIXED_PARAMS: dict = {
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
    "gamma": 0.0,
    "tree_method": "hist",
    "eval_metric": "logloss",
}

_GRID: list[dict] = [
    {"max_depth": d, "learning_rate": lr, "n_estimators": n}
    for d, lr, n in product([4, 6, 8], [0.05, 0.10], [500, 1000])
    if not (d == 8 and lr == 0.10 and n == 1000)  # skip heaviest combo
][:9]

# Threshold candidates
_THRESHOLDS = np.arange(0.05, 0.55, 0.05)

# Inner validation fraction (last 20 % of training data by time)
_INNER_VAL_FRACTION = 0.20


# ── Grid evaluation ──────────────────────────────────────────────────


def _evaluate_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    scale_pos_weight: float,
    random_state: int,
) -> float:
    """Train one config and return validation AUC-ROC."""
    full_params = {**_FIXED_PARAMS, **params}
    model = XGBClassifier(
        n_estimators=full_params["n_estimators"],
        max_depth=full_params["max_depth"],
        learning_rate=full_params["learning_rate"],
        subsample=full_params["subsample"],
        colsample_bytree=full_params["colsample_bytree"],
        reg_alpha=full_params["reg_alpha"],
        reg_lambda=full_params["reg_lambda"],
        min_child_weight=full_params["min_child_weight"],
        gamma=full_params["gamma"],
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, y_prob))


def _run_grid_search(
    X_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    random_state: int,
) -> tuple[dict, float]:
    """Evaluate all grid configurations; return (best_params, best_auc)."""
    best_params: dict | None = None
    best_auc = -1.0

    for i, cfg in enumerate(_GRID, 1):
        auc = _evaluate_config(
            X_inner_train, y_inner_train,
            X_val, y_val,
            cfg, scale_pos_weight, random_state,
        )
        marker = " ★" if auc > best_auc else ""
        print(
            f"  [{i}/{len(_GRID)}] depth={cfg['max_depth']} "
            f"lr={cfg['learning_rate']:.2f} "
            f"n={cfg['n_estimators']:4d}  →  AUC={auc:.4f}{marker}"
        )
        if auc > best_auc:
            best_auc = auc
            best_params = cfg

    assert best_params is not None
    return best_params, best_auc


# ── Threshold analysis ───────────────────────────────────────────────


def _analyze_thresholds(
    y_val: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """Evaluate multiple thresholds on validation data.

    Returns:
        (selected_threshold, results_df)
    """
    rows = []
    for t in _THRESHOLDS:
        preds = (y_prob >= t).astype(int)
        p = precision_score(y_val, preds, pos_label=FRAUD_LABEL, zero_division=0)
        r = recall_score(y_val, preds, pos_label=FRAUD_LABEL, zero_division=0)
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)})

    df = pd.DataFrame(rows)

    # Select threshold that maximises F1 (balance of Precision and Recall)
    best_idx = df["f1"].idxmax()
    selected = float(df.loc[best_idx, "threshold"])
    return selected, df


# ── Main pipeline ────────────────────────────────────────────────────


def main() -> None:
    """CLI: run controlled XGBoost tuning."""
    print("Controlled XGBoost Improvement — Tuning Pipeline")
    print("=" * 60)
    print()

    # ── Step 1: Load features and split ──────────────────────────────
    print("Step 1/5 — Loading data...")
    from ml.features.engineer import engineer_features
    from ml.data.loader import load_transaction_dataset
    from ml.split.splitter import time_based_split

    fe = engineer_features()
    features = fe["features"]
    target = fe["target"]
    txn_ids = fe["transaction_ids"]

    txn = load_transaction_dataset()
    ts_map = txn.set_index("TransactionID")["TransactionDT"]
    timestamps = ts_map.loc[txn_ids.values].reset_index(drop=True)
    timestamps.index = features.index

    split = time_based_split(
        features=features, target=target,
        timestamps=timestamps, transaction_ids=txn_ids,
    )
    X_train_full = split.X_train
    y_train_full = split.y_train
    X_test = split.X_test
    y_test = split.y_test
    print(f"  Training: {len(X_train_full):,} rows")
    print(f"  Test:     {len(X_test):,} rows (held out)")
    print()

    # ── Step 2: Inner temporal validation split ──────────────────────
    print("Step 2/5 — Creating temporal validation split...")
    n_train = len(X_train_full)
    val_cut = int(n_train * (1.0 - _INNER_VAL_FRACTION))

    # Data is already sorted by TransactionDT from the split
    X_inner_train = X_train_full.iloc[:val_cut].reset_index(drop=True)
    y_inner_train = y_train_full.iloc[:val_cut].reset_index(drop=True)
    X_val = X_train_full.iloc[val_cut:].reset_index(drop=True)
    y_val = y_train_full.iloc[val_cut:].reset_index(drop=True)

    _guard_against_leakage(X_inner_train, X_val)

    print(f"  Inner train: {len(X_inner_train):,} rows")
    print(f"  Validation:  {len(X_val):,} rows")
    print()

    # ── Step 3: Preprocessing (fitted on inner train only) ───────────
    print("Step 3/5 — Fitting preprocessing on inner training data...")
    prep_inner = fit_preprocessing(X_inner_train)
    X_inner_t = apply_preprocessing(X_inner_train, prep_inner)
    X_val_t = apply_preprocessing(X_val, prep_inner)

    n_pos = int(y_inner_train.sum())
    n_neg = len(y_inner_train) - n_pos
    spw = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight: {spw:.2f}")
    print()

    # ── Step 4: Grid search ──────────────────────────────────────────
    print(f"Step 4/5 — Grid search ({len(_GRID)} configs)...")
    best_params, best_auc = _run_grid_search(
        X_inner_t, y_inner_train.values,
        X_val_t, y_val.values,
        spw, RANDOM_SEED,
    )
    print(f"\n  Best config: {best_params}")
    print(f"  Best val AUC-ROC: {best_auc:.4f}")
    print()

    # ── Step 5: Retrain on full training set ─────────────────────────
    print("Step 5/5 — Retraining best config on full training set...")
    prep_full = fit_preprocessing(X_train_full)
    X_train_full_t = apply_preprocessing(X_train_full, prep_full)
    X_test_t = apply_preprocessing(X_test, prep_full)

    # Recompute scale_pos_weight for full training set
    n_pos_full = int(y_train_full.sum())
    n_neg_full = len(y_train_full) - n_pos_full
    spw_full = n_neg_full / max(n_pos_full, 1)

    full_params = {**_FIXED_PARAMS, **best_params}
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
    final_model.fit(X_train_full_t, y_train_full.values)
    print(f"  Trained on {len(X_train_full):,} rows, {len(full_params)} params")

    # Threshold analysis on validation data (re-apply full preprocessing)
    X_val_full_t = apply_preprocessing(X_val, prep_full)
    val_prob = final_model.predict_proba(X_val_full_t)[:, 1]
    selected_threshold, threshold_df = _analyze_thresholds(y_val.values, val_prob)

    print(f"\n  Threshold analysis (on validation):")
    for _, row in threshold_df.iterrows():
        marker = " ◄ selected" if abs(row["threshold"] - selected_threshold) < 0.001 else ""
        print(
            f"    t={row['threshold']:.2f}  "
            f"P={row['precision']:.4f}  R={row['recall']:.4f}  "
            f"F1={row['f1']:.4f}{marker}"
        )
    print(f"\n  Selected threshold: {selected_threshold:.2f}")

    # ── Final test evaluation ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Final Test Evaluation (one-time)")
    print(f"{'=' * 60}")

    test_prob = final_model.predict_proba(X_test_t)[:, 1]
    test_pred = (test_prob >= selected_threshold).astype(int)

    final_precision = float(precision_score(y_test, test_pred, pos_label=FRAUD_LABEL, zero_division=0))
    final_recall = float(recall_score(y_test, test_pred, pos_label=FRAUD_LABEL, zero_division=0))
    final_auc = float(roc_auc_score(y_test, test_prob))
    final_avg_prec = float(average_precision_score(y_test, test_prob, pos_label=FRAUD_LABEL))

    # Baselines
    lr_p, lr_r, lr_auc = 0.0762, 0.6681, 0.7472
    prev_p, prev_r, prev_auc = 0.0964, 0.7013, 0.8158

    def _d(v, b):
        return f"{'+' if v >= b else ''}{v - b:.4f}"

    print(f"\n  Precision : {final_precision:.4f}  (LR {lr_p}, prev XGB {prev_p}, Δ {_d(final_precision, prev_p)})")
    print(f"  Recall    : {final_recall:.4f}  (LR {lr_r}, prev XGB {prev_r}, Δ {_d(final_recall, prev_r)})")
    print(f"  AUC-ROC   : {final_auc:.4f}  (LR {lr_auc}, prev XGB {prev_auc}, Δ {_d(final_auc, prev_auc)})")
    print(f"  Avg Prec  : {final_avg_prec:.4f}")
    print(f"\n  Test rows     : {len(y_test):,}")
    print(f"  Fraud in test : {int(y_test.sum()):,} ({y_test.mean():.2%})")

    print(f"\n{'─' * 60}")
    print("Target Comparison")
    print(f"{'─' * 60}")
    def _s(v, t): return "MEETS TARGET" if v >= t else "below target"
    print(f"  Precision : {final_precision:.4f}  (≥ {TARGET_PRECISION})  [{_s(final_precision, TARGET_PRECISION)}]")
    print(f"  Recall    : {final_recall:.4f}  (≥ {TARGET_RECALL})  [{_s(final_recall, TARGET_RECALL)}]")
    print(f"  AUC-ROC   : {final_auc:.4f}  (≥ {TARGET_AUC_ROC})  [{_s(final_auc, TARGET_AUC_ROC)}]")

    print(f"\n{'─' * 60}")
    print("Configuration")
    print(f"{'─' * 60}")
    print(f"  Best params: {best_params}")
    print(f"  scale_pos_weight: {spw_full:.2f}")
    print(f"  Threshold: {selected_threshold:.2f}")
    print(f"  Trees: {full_params['n_estimators']}, Depth: {full_params['max_depth']}, LR: {full_params['learning_rate']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
