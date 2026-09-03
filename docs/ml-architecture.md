# ML Architecture

## Overview

The ML/Fraud Intelligence Service is a **separate internal HTTP service** that provides three independent fraud intelligence signals combined by a risk aggregator. It runs on `ML_SERVICE_HOST:ML_SERVICE_PORT` and is called by the backend via HTTP for every transaction.

The ML/Fraud developer (Developer C) owns all components of this service: feature engineering, ML model, behaviour engine, rule engine, risk aggregation, explainability, and decision engine.

```
Transaction + Customer Profile + History
                  │
                  ▼
          ┌───────────────┐
          │   Feature     │
          │  Engineering  │
          └───────┬───────┘
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌─────────────┐
   │   ML    │ │Behaviour│ │ Rule-Based  │
   │ Model   │ │ Anomaly │ │ Risk Rules  │
   └────┬────┘ └────┬────┘ └──────┬──────┘
        │           │             │
        └───────────┼─────────────┘
                    ▼
           ┌───────────────┐
           │    Risk       │
           │  Aggregator   │
           └───────┬───────┘
                   ▼
           ┌───────────────┐
           │ Explainability│
           └───────┬───────┘
                   ▼
           ┌───────────────┐
           │   Decision    │
           │    Engine     │
           └───────────────┘
```

## ML Service HTTP Interface

The backend calls the ML/Fraud Intelligence Service for every transaction and fraud check request.

### Request Schema

```
POST /predict
Content-Type: application/json

{
  "customer_id": "uuid",
  "customer_history": {
    "transaction_count_30d": 15,
    "avg_amount_30d": 500.00,
    "std_amount_30d": 200.00,
    "last_transaction_country": "US",
    "last_transaction_timestamp": "2026-08-29T10:00:00Z",
    "known_device_fingerprints": ["abc123", "def456"],
    "known_merchant_ids": ["uuid1", "uuid2"],
    "previous_flagged_count": 0
  },
  "transaction": {
    "amount": 1500.00,
    "currency": "USD",
    "merchant_id": "uuid",
    "merchant_name": "Acme Electronics",
    "merchant_category": "5732",
    "transaction_type": "purchase",
    "location_country": "US",
    "location_city": "New York",
    "device_fingerprint": "abc123def456",
    "device_type": "mobile",
    "ip_address": "192.168.1.100",
    "timestamp": "2026-08-29T14:30:00Z"
  }
}
```

### Response Schema

```
{
  "ml_score": 35,
  "behaviour_score": 52,
  "rule_score": 15,
  "risk_score": 45,
  "risk_level": "MEDIUM",
  "decision": "VERIFY",
  "explanation": {
    "ml_top_factors": [
      { "feature": "amount_deviation", "importance": 0.35 },
      { "feature": "is_new_device", "importance": 0.22 }
    ],
    "behaviour_signals": [
      { "signal": "spending_amount_anomaly", "severity": 0.6 },
      { "signal": "device_anomaly", "severity": 0.5 }
    ],
    "rules_triggered": [
      { "rule": "new_device_high_amount", "contribution": 15 }
    ]
  },
  "risk_factors": ["amount_deviation", "is_new_device", "new_device_high_amount"],
  "model_version": "fraud-xgb-v1.2.0"
}
```

### Failure Handling

| Scenario | Behaviour |
|---|---|
| **Timeout** (configurable, default 5s) | Backend returns 503 to client; transaction remains `PENDING` or is marked `FAILED`; fraud scores remain NULL |
| **Connection refused** | Backend returns 503 to client; logs alert; transaction not persisted |
| **Model not available** | ML service returns 503 with `{ "error": "MODEL_NOT_AVAILABLE" }`; backend propagates 503 |
| **ML service unhealthy** | Backend `/api/v1/health` reports `ml_service.status = "unavailable"` |
| **Invalid response from ML** | Backend logs error, returns 500 to client |

The backend **never** calculates ML predictions itself. If the ML service is unavailable, the transaction cannot be processed.

## 1. Feature Engineering

Raw transaction and customer data is transformed into numerical features consumed by the ML model and behaviour analysis.

### Planned Features

| Feature | Source | Description |
|---|---|---|
| `amount` | Transaction | Raw transaction amount |
| `amount_deviation` | Transaction + history | Z-score of amount relative to customer's historical mean |
| `amount_to_avg_ratio` | Transaction + history | Ratio of current amount to customer average |
| `location_country` | Transaction | Country of transaction (encoded) |
| `location_is_new` | Transaction + history | Whether customer has transacted from this country before |
| `location_change` | Transaction + history | Flag if country differs from last N transactions |
| `device_fingerprint` | Transaction | Device identifier (encoded) |
| `is_new_device` | Transaction + history | Whether this device has been seen before |
| `hour_of_day` | Transaction timestamp | Hour extracted (cyclical encoding) |
| `day_of_week` | Transaction timestamp | Day extracted (cyclical encoding) |
| `is_unusual_hour` | Transaction + history | Whether the hour falls outside customer's typical activity window |
| `tx_velocity_1h` | Transaction history | Number of transactions in the last 1 hour |
| `tx_velocity_24h` | Transaction history | Number of transactions in the last 24 hours |
| `tx_velocity_7d` | Transaction history | Number of transactions in the last 7 days |
| `merchant_category` | Transaction | Merchant category code (encoded) |
| `merchant_is_new` | Transaction + history | Whether customer has transacted with this merchant before |
| `transaction_type` | Transaction | Type label (encoded) |
| `avg_spend_30d` | Transaction history | Rolling 30-day average spend |
| `previous_suspicious_count` | Transaction history | Count of previously flagged transactions |

### Feature Pipeline Principles

- All features are computed server-side; the frontend sends only raw transaction fields.
- Feature computation must be deterministic and reproducible.
- Feature transformations (scaling, encoding) are fitted on training data and persisted with the model.

## 2. ML Prediction Model

### Approach

- **Algorithm:** Start with a baseline logistic regression; evaluate XGBoost for improved performance.
- **Target:** Binary classification (fraud = 1, legitimate = 0).
- **Training data:** Synthetic data and/or legitimate public fraud datasets (e.g., IEEE-CIS Fraud Detection, PaySim).

### Training Pipeline

```
Raw Data
  ↓
Data Validation (schema + quality checks)
  ↓
Feature Engineering
  ↓
Train/Test Split (stratified, time-aware if possible)
  ↓
Model Training (cross-validation)
  ↓
Evaluation (precision, recall, F1, AUC-ROC)
  ↓
Serialisation (joblib)
  ↓
Model Registry (model_metadata table)
```

### Output

- A fraud probability score in the range [0.0, 1.0], normalised to a 0–100 scale for the risk aggregator.

### Model Selection Criteria

| Metric | Minimum Threshold |
|---|---|
| Precision (fraud class) | ≥ 0.70 |
| Recall (fraud class) | ≥ 0.80 |
| AUC-ROC | ≥ 0.85 |

These are targets, not hard gates. The team will iterate to improve performance.

## 3. Behavioural Anomaly Analysis

A non-ML statistical layer that detects deviations from the customer's established baseline.

### Signals

| Signal | Method |
|---|---|
| Spending amount anomaly | Z-score against customer's rolling mean/std |
| Location anomaly | First occurrence of a country/city in customer history |
| Device anomaly | First occurrence of a device fingerprint |
| Time anomaly | Transaction outside customer's typical activity hours |
| Velocity anomaly | Transaction count exceeds customer's rolling average + threshold |

### Output

- A behaviour anomaly score in the range [0, 100], where higher values indicate greater deviation from baseline.

## 4. Rule-Based Risk Signals

Configurable business rules that flag known fraud patterns.

### Planned Rules

| Rule | Trigger | Score Contribution |
|---|---|---|
| High amount | Amount exceeds configurable threshold | +15 |
| Impossible travel | Two transactions from distant locations within an impossibly short time | +25 |
| Velocity limit | More than N transactions within T minutes | +20 |
| New device + high amount | First-seen device with amount above threshold | +15 |
| High-risk merchant category | Transaction in a category flagged as high-risk | +10 |
| Previous suspicious activity | Customer has prior flagged transactions | +10 |

Rules and their score contributions are configurable and can be adjusted without code changes (via configuration file or environment).

### Output

- A cumulative rule-based score in the range [0, 100], capped at 100.

## 5. Risk Aggregation

The ML/Fraud Intelligence Service combines the three scores internally:

```
risk_score = (w_ml × ml_score) + (w_behaviour × behaviour_score) + (w_rule × rule_score)
```

- Weights are configurable via environment variables (`ML_WEIGHT_ML`, `ML_WEIGHT_BEHAVIOUR`, `ML_WEIGHT_RULE`).
- Default weights: `w_ml = 0.50`, `w_behaviour = 0.30`, `w_rule = 0.20`.
- Final score is clamped to [0, 100].
- The backend receives the aggregated score and persists it; the backend does not recalculate or modify it.

The decision engine then applies configurable thresholds:

| Score | Level | Decision |
|---|---|---|
| 0–30 | LOW | APPROVE |
| 31–70 | MEDIUM | VERIFY |
| 71–100 | HIGH | HOLD + ALERT |

## 6. Explainability

For every transaction, the system produces an explanation of why a particular risk score was assigned.

### Approach

- **SHAP values** for the ML model component (identifies which features pushed the prediction towards fraud).
- **Rule trigger list** for the rule-based component (which rules fired).
- **Top behavioural deviations** for the behaviour component.

The top N contributing factors are returned in the API response and stored in `transactions.explanation_json`.

## 7. Model Lifecycle

| Stage | Description |
|---|---|
| Training | Offline batch process; outputs a serialised model artefact (.joblib) plus a model manifest. |
| Validation | Held-out temporal test-split evaluation through the offline evaluation framework (Step 47); produces a structured JSON report with metrics, threshold analysis, calibration, and labelled recommendations. |
| Deployment | Model artefact + manifest stored in `ml/models/`. Service activates via integrity verification at startup (Step 46). |
| Monitoring | Prediction distribution drift tracking plus model identity observability (Step 43/46). |
| Retraining | Triggered manually or on a schedule when performance degrades (future). |

### Model Loading (Step 46 — Model Governance)

- **Artifact location:** `ml/models/` by default, override via the `ML_MODEL_DIR` environment variable. The registry reads `ml/models/model_manifest.json` to locate the active artifact.
- **Activation sequence:** manifest loaded → artifact checksum (SHA-256) verified → bundle loaded → model interface validated (`predict_proba`) → feature compatibility validated (feature count + schema version) → model marked ACTIVE.
- **Startup behaviour:** The ML service loads the model through the `ModelRegistry` at startup. If the manifest is missing, the artifact is missing/corrupt, the checksum mismatches, or interface/feature validation fails, the service starts in a `model_unavailable` state — it never silently falls back to an unverified model.
- **Model identity:** The authoritative identity (name, version, artifact checksum, feature schema version, feature count) is exposed consistently via `/health`, `/ready`, `/metrics` (`model_identity` field), prediction responses (`model_version`), and fraud-decision audit records.
- **Rollback:** Configuration-based. An operator supplies a previously verified `ModelManifest` to `ModelRegistry.rollback()`; the target must still pass full checksum/interface validation. Invalid targets are rejected and the current active model is untouched. Failed activation of a candidate never destroys a working model.
- **Trust boundary:** Model artifacts use joblib serialisation (pickle-based, can execute code on load). Artifacts must originate from a trusted build/training pipeline. The checksum proves the artifact is unmodified since manifest creation — it does **not** sandbox an untrusted artifact.
- **Client isolation:** Clients can never supply or override the model version, artifact path, or active model selection. No model-management endpoint is exposed.
- **Health endpoint:** The ML service exposes a `/health` endpoint that reports `{ "status": "ready", "model_version": "...", "model_identity": {...} }` or `{ "status": "model_unavailable" }`.
- **Cold start (no trained model):** The service starts and reports `model_unavailable`. All `/predict` requests return 503. The backend propagates this as 503 to clients.
- **Model version in response:** Every prediction response includes `model_version` identifying which model was used.
- **Model version persistence:** The backend persists `model_version` in the `transactions` table for every successfully analysed transaction. This enables per-transaction audit traceability: *"Which model produced this fraud score?"*

#### Manifest format (`ml/models/model_manifest.json`)

```json
{
  "artifact_checksum": "<sha-256 hex of artifact>",
  "artifact_filename": "fraud_xgb_tuned.joblib",
  "created_at": "<ISO 8601 UTC>",
  "feature_schema_version": "1.0.0",
  "model_name": "fraud-xgb",
  "model_version": "fraud-xgb-v1.0.0",
  "n_features": 24,
  "serialization_format": "joblib",
  "status": "active",
  "threshold": 0.5
}
```

#### Operator workflow

1. Train/save: `python -m ml.predict.save_model` (writes artifact + manifest).
2. Verify: the service validates the manifest checksum on every startup.
3. Rollback: point the manifest (or `ML_MODEL_DIR`) at a previously verified version; the service re-validates the target before activating it.
4. Investigate: `/health`, `/ready`, and `/metrics` report the active identity including checksum.

### Offline Evaluation (Step 47 — Evaluation & Threshold Governance)

**Evaluation recommendations DO NOT automatically change production decisions.**
Everything in this section is observational: the production threshold, the
active model, risk aggregation, monitoring counters, and audit records are
never modified by an evaluation.

#### Evaluation architecture

- `ml/evaluation/` package: `config.py` (`EVAL_*` configuration),
  `metrics.py` (classification / ranking / calibration metrics),
  `thresholds.py` (threshold sweep + recommendation strategies), and
  `runner.py` (report assembly + offline CLI).
- Evaluation activates the model through the same Step 46 governance
  pipeline (`ModelRegistry.activate_from_manifest`); it never trusts a
  caller-claimed model version and never loads unverified artifacts.
- The evaluation dataset is the held-out temporal test split produced by the
  existing feature-engineering + `time_based_split` pipeline (approved
  source only — no caller-supplied dataset or artifact paths).
- No evaluation API endpoint exists. The operator entry point is the
  offline CLI: `python -m ml.evaluation.runner [--output report.json]`.

#### Metrics definitions

| Metric | Definition |
|---|---|
| Confusion matrix | TP / TN / FP / FN at a given threshold (`[[tn, fp], [fn, tp]]`). |
| Precision (among flagged) | TP / (TP + FP); 0.0 when nothing is flagged (documented convention). |
| Recall (fraud detection rate) | TP / (TP + FN); 0.0 when there is no fraud. |
| F1 | Harmonic mean of precision and recall. |
| FPR / FNR | FP / (FP + TN) and FN / (FN + TP); 0.0 when the denominator class is absent. |
| Fraud prevalence | Fraud / total samples in the evaluation split. |
| ROC-AUC / PR-AUC | Ranking quality; PR-AUC uses average precision (appropriate under imbalance). Single-class labels raise a clear `RankingError` instead of silently computing meaningless values. |
| Brier score | Mean squared error of raw fraud probabilities. |
| Reliability bins | Uniform reliability-diagram bins (sklearn-compatible); empty bins omitted. |

#### Threshold analysis

- Deterministic sweep over the configured grid (default 0.05–0.95, step
  0.05, stop inclusive; hard-bounded to ≤ 201 points).
- Every point records threshold, TP/TN/FP/FN, precision, recall, F1,
  FPR, FNR, and flagged count/rate.
- Tie-break rule: when several thresholds are equally optimal, the
  **highest** threshold wins (fewer flagged transactions for the same
  score).

#### Cost-sensitive analysis

- Optional business-cost model:
  `total_cost = FN × EVAL_FN_COST + FP × EVAL_FP_COST`.
- Costs are never hard-coded. When they are not configured, the report
  honestly marks cost analysis (and the `min_cost` recommendation)
  unavailable with an explicit reason instead of inventing defaults.

#### Calibration

- Brier score plus reliability-diagram bins computed on **raw** model
  probabilities.
- No recalibration (Platt / isotonic) is fitted, applied, or returned; a
  calibrated model would be a separate, explicitly approved offline
  artifact and must never silently replace the live probability.

#### Imbalance considerations

- Fraud prevalence on the holdout split is ~3.4%, so accuracy is not a
  meaningful headline; PR-AUC (average precision) is the primary ranking
  metric alongside ROC-AUC, and precision/recall/F1 are reported per
  threshold.
- Class imbalance is handled at training time (`scale_pos_weight`);
  evaluation does not resample — resampling would break the
  leakage-safe temporal split.

#### Leakage protections

- `isFraud` and `TransactionID` can never be model features: the scorer
  uses only the bundle's declared feature columns; the split and the
  live predictor enforce the same forbidden-column rules.
- The test split is strictly after every training transaction in time;
  historical features are computed strictly prior to each transaction.
- Preprocessing is reused **transform-only** — never re-fitted on
  evaluation data.

#### Model version traceability

- The report's model identity (name, version, artifact checksum, schema
  version, feature count) is taken from the verified governance identity;
  a bundle/identity mismatch aborts the evaluation.
- The report records the observed production threshold and its source
  ("observed — NOT modified by evaluation").

#### Reproducibility

- The report embeds the evaluation configuration, split strategy,
  dataset source, tie-break rule, and dataset metadata (sample/fraud
  counts, split timestamp, test fraction).
- Evaluation is fully deterministic — repeated runs differ only in the
  timestamp.
- The dataset is not committed to version control; reproduce by placing
  the IEEE-CIS CSVs in `ml/datasets/raw/` and running the CLI above.

#### Threshold governance

- Four recommendation strategies:
  `max_f1`, `min_cost` (requires configured costs), `min_recall`
  (maximise precision subject to recall ≥ `EVAL_MIN_RECALL`), and
  `min_precision` (maximise recall subject to precision ≥
  `EVAL_MIN_PRECISION`).
- Unconfigured or unsatisfiable strategies are reported unavailable with
  an explicit reason — never silently dropped or defaulted.
- Every recommendation and the report itself are labelled
  **EVALUATION / RECOMMENDATION ONLY**.
- Applying a different threshold requires producing a new model artifact
  + manifest through the Step 46 governance pipeline (retrain or re-save
  with an approved threshold); evaluation never writes production state.

#### Offline vs production separation

- Evaluation runs in its own registry instance and performs no writes
  except the optional operator-requested `--output` JSON file.
- `/predict`, `/health`, `/ready`, monitoring counters, and audit
  records are unchanged by an evaluation (verified end-to-end in
  `ml/tests/e2e_step47_evaluation.py`).

#### Operator workflow

1. Configure `EVAL_*` variables (see `.env.example`).
2. Run: `python -m ml.evaluation.runner --output report.json`.
3. Review the printed summary and the JSON report.
4. If a threshold change is warranted, follow the model lifecycle above
   (retrain/re-save with the approved threshold) — evaluation never
   modifies production configuration.

#### Known limitations

- The tuned model's precision on the holdout split is low (0.0808 at the
  0.5 production threshold); evaluation surfaces this honestly rather
  than tuning it away.
- Calibration is assessed, not corrected; probabilities remain raw.
- Recommendations are point-in-time observations on the holdout split
  without confidence intervals or drift-aware re-evaluation scheduling.

## Data Policy

- **No real banking or customer data** is used at any stage.
- Synthetic data generators and legitimate public datasets (e.g., IEEE-CIS, PaySim) are used for training and evaluation.
- Training data is stored in `ml/datasets/` and excluded from version control via `.gitignore`.

## Status

Implemented. Feature engineering, model training, behaviour/rules engines, risk aggregation, explainability, monitoring, hardening, model lifecycle governance (Step 46: manifest-based integrity verification, registry activation, rollback safety), and offline model evaluation with threshold governance (Step 47: metrics, threshold sweep, cost analysis, calibration, labelled recommendations — strictly evaluation-only) are complete.
