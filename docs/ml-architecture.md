# ML Architecture

## Overview

The ML subsystem provides three independent fraud intelligence signals that are combined by the backend's risk aggregator.

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
           │   Backend:    │
           │   Risk        │
           │   Aggregator  │
           └───────────────┘
```

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

## 5. Risk Aggregation (Backend Responsibility)

The backend combines the three scores:

```
risk_score = (w_ml × ml_score) + (w_behaviour × behaviour_score) + (w_rule × rule_score)
```

- Weights are configurable via environment variables.
- Default weights: `w_ml = 0.50`, `w_behaviour = 0.30`, `w_rule = 0.20`.
- Final score is clamped to [0, 100].

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
| Training | Offline batch process; outputs a serialised model artefact. |
| Validation | Held-out test set evaluation; metrics logged to `model_metadata`. |
| Deployment | Model artefact loaded into the inference service at startup. |
| Monitoring | Track prediction distribution drift over time (future). |
| Retraining | Triggered manually or on a schedule when performance degrades (future). |

## Data Policy

- **No real banking or customer data** is used at any stage.
- Synthetic data generators and legitimate public datasets (e.g., IEEE-CIS, PaySim) are used for training and evaluation.
- Training data is stored in `ml/datasets/` and excluded from version control via `.gitignore`.

## Status

This ML architecture is agreed upon but **not yet implemented**. Feature engineering, model training, and the behaviour/rules engines will be developed during the implementation phase.
