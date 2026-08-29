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
| Training | Offline batch process; outputs a serialised model artefact (.joblib). |
| Validation | Held-out test set evaluation; metrics logged to `model_metadata` table. |
| Deployment | Model artefact stored at `ML_MODEL_PATH`. Service loads on startup. |
| Monitoring | Track prediction distribution drift over time (future). |
| Retraining | Triggered manually or on a schedule when performance degrades (future). |

### Model Loading

- **Artifact location:** Path specified by `ML_MODEL_PATH` environment variable (e.g., `ml/models/fraud_xgb_v1.joblib`).
- **Version selection:** The latest model from `model_metadata` table, or the path specified by `ML_MODEL_VERSION` env var.
- **Startup behaviour:** The ML service loads the model at startup. If the model file is missing or corrupt, the service starts in a `model_unavailable` state.
- **Health endpoint:** The ML service exposes a `/health` endpoint that reports `{ "status": "ready", "model_version": "..." }` or `{ "status": "model_unavailable" }`.
- **Cold start (no trained model):** The service starts and reports `model_unavailable`. All `/predict` requests return 503. The backend propagates this as 503 to clients.
- **Model version in response:** Every prediction response includes `model_version` identifying which model was used.
- **Model version persistence:** The backend persists `model_version` in the `transactions` table (FK → `model_metadata.model_version`) for every successfully analysed transaction. This enables per-transaction audit traceability: *"Which model produced this fraud score?"*

## Data Policy

- **No real banking or customer data** is used at any stage.
- Synthetic data generators and legitimate public datasets (e.g., IEEE-CIS, PaySim) are used for training and evaluation.
- Training data is stored in `ml/datasets/` and excluded from version control via `.gitignore`.

## Status

This ML architecture is agreed upon but **not yet implemented**. Feature engineering, model training, and the behaviour/rules engines will be developed during the implementation phase.
