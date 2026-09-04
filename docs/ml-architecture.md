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

### Promotion Gate (Step 48 — Automated Model Validation & Promotion Gate)

**The promotion gate NEVER activates a model automatically, NEVER
modifies the production manifest, and NEVER changes the production
decision threshold.** Promotion remains an explicit operator action
through the Step 46 governance workflow.

#### Gate architecture

- `ml/evaluation/promotion_policy.py`: configurable promotion policy
  (`PROMO_*` environment variables) with 6 absolute minimum
  requirements and 6 relative regression limits vs production.
- `ml/evaluation/promotion_gate.py`: the gate itself — validates the
  candidate through the Step 46 governance sequence (scratch registry),
  evaluates both models on the same held-out dataset through the Step 47
  framework, applies every configured gate, and answers `APPROVED` or
  `REJECTED`.
- The gate reuses the Step 46 `ModelRegistry` and Step 47
  `build_report` — no code duplication of integrity verification or
  evaluation logic.
- Fail-closed: any validation gap (missing artifact, invalid checksum,
  malformed manifest, failed evaluation, unavailable metrics, invalid
  policy) yields `REJECTED` — never `APPROVED` with incomplete
  validation.

#### Policy configuration

Absolute minimum requirements (fractions in [0, 1]):

- `PROMO_MIN_PR_AUC` (default 0.10), `PROMO_MIN_ROC_AUC` (0.75),
  `PROMO_MIN_RECALL` (0.70), `PROMO_MIN_PRECISION` (0.05),
  `PROMO_MIN_F1` (0.10), `PROMO_MAX_BRIER` (0.25).

Relative regression limits vs production (fractions in [0, 1]; 0.10
means the candidate may degrade at most 10% relative to production):

- `PROMO_MAX_PR_AUC_DEGRADATION` (0.05),
  `PROMO_MAX_ROC_AUC_DEGRADATION` (0.02),
  `PROMO_MAX_RECALL_DEGRADATION` (0.10),
  `PROMO_MAX_PRECISION_DEGRADATION` (0.10),
  `PROMO_MAX_F1_DEGRADATION` (0.10),
  `PROMO_MAX_BRIER_INCREASE` (0.10).

Unset/empty variables fall back to defaults; explicit `none` or `off`
disables that gate; out-of-range values fail closed.

#### Gate semantics

- Gates are inclusive at the boundary: a candidate exactly at a
  required minimum, degradation limit, or Brier ceiling passes (tolerance
  1e-9 for float safety).
- Relative floor = production × (1 − degradation limit); relative
  Brier ceiling = production × (1 + increase limit).
- Unavailable metrics fail closed (the gate never says "approved" when
  validation is incomplete).

#### Result structure

The gate returns a structured `PromotionDecision` containing:

- Candidate and production model identities (from verified governance).
- Evaluation metadata (dataset identifier, sample counts, thresholds
  observed).
- Bounded metric summaries for both models (precision, recall, F1,
  ROC-AUC, PR-AUC, Brier score — aggregate only, no raw data).
- Every configured gate with actual value, required value, PASS/FAIL
  status, and human-readable detail.
- Overall decision (`APPROVED` or `REJECTED`) and rejection reasons.
- Promotion instructions (only when approved) for the Step 46 workflow.
- Reproducibility metadata (policy source, evaluation config, dataset
  identifier, report schema version, timestamp).

The report contains **no** raw transactions, customer IDs, raw labels,
prediction arrays, filesystem paths, or secrets.

#### CLI

```bash
python -m ml.evaluation.promotion_gate --candidate-model-dir <dir> [--output PATH]
```

Exit codes: 0 approved, 1 rejected (including fail-closed validation
failures), 2 unexpected internal error. No stack traces or internal
paths are printed in normal output.

#### Security

- The candidate is validated through the full Step 46 governance
  sequence (manifest → SHA-256 checksum → bundle load → interface →
  feature-schema/count compatibility) using a scratch registry — the
  candidate is never activated as, or swapped into, production.
- The production identity is always verified from the manifest — never
  claimed by the caller.
- No arbitrary path traversal, no unsafe candidate metadata trust, no
  pickle loading outside the existing trusted model-artifact boundary,
  no arbitrary code execution introduced by the gate.
- Reports are bounded (JSON-safe, < 32 KB), contain no secrets, and
  are safe to persist or share.
- The gate never writes to the production model directory, manifest,
  or runtime state.

#### Operator workflow

1. Place the verified candidate artifact and `model_manifest.json` in
   a trusted directory (produced by the trusted training pipeline).
2. Run: `python -m ml.evaluation.promotion_gate --candidate-model-dir <dir> --output report.json`.
3. Review the printed summary and the JSON report.
4. If `APPROVED` and promotion is warranted, follow the Step 46
   governance workflow (explicit operator action) to activate the
   candidate — the gate does not perform this step.

#### Known limitations

- The gate evaluates on the held-out test split only; it does not
  perform cross-validation or drift-aware re-evaluation scheduling.
- Promotion decisions are point-in-time observations; they do not
  account for future data distribution changes.
- The gate does not perform statistical significance testing; it
  applies configured policy thresholds deterministically.

### Promotion History (Step 49 — Audit Trail & Traceability)

**The promotion history NEVER modifies production state, the active
manifest, the production threshold, or runtime model. It is an
append-only, read-only audit trail.**

#### History architecture

- `ml/evaluation/promotion_history.py`: persistence layer for
  promotion gate decisions. After each gate run, the decision is
  automatically saved to a structured history directory.
- Storage: JSON files in `PROMO_HISTORY_DIR` (default:
  `ml/promotion_history/`). Each file is named by the gate timestamp
  (ISO 8601 UTC, filesystem-safe).
- Bounded: max 1000 files, each < 32 KB. When the limit is reached,
  the oldest files are automatically removed.
- Fail-safe: history write failures are logged but do not affect the
  promotion gate decision. The gate still returns APPROVED/REJECTED.

#### History contents

Each history file contains the full `PromotionDecision` dict (from
`PromotionDecision.to_dict()`), which includes:

- Candidate and production model identities (verified governance).
- Evaluation metadata (dataset identifier, sample counts).
- Bounded metric summaries for both models.
- Every configured gate with actual/required values and PASS/FAIL.
- Overall decision and rejection reasons.
- Reproducibility metadata (policy source, config, timestamp).

History files contain **no** raw transactions, customer IDs, raw
labels, prediction arrays, filesystem paths, or secrets.

#### CLI

```bash
# List recent decisions
python -m ml.evaluation.promotion_history

# Filter by decision
python -m ml.evaluation.promotion_history --decision APPROVED
python -m ml.evaluation.promotion_history --decision REJECTED

# Limit output
python -m ml.evaluation.promotion_history --limit 10

# Show summary statistics
python -m ml.evaluation.promotion_history --summary

# Override history directory
python -m ml.evaluation.promotion_history --history-dir /path/to/history
```

Exit codes: 0 success, 1 error.

#### Configuration

- `PROMO_HISTORY_DIR`: history directory path (default:
  `ml/promotion_history/`). Set to `none` or `off` to disable
  history persistence.
- `MAX_HISTORY_FILES`: max number of history files (default: 1000).
- `MAX_HISTORY_FILE_SIZE`: max file size in bytes (default: 32768).

#### Security

- History is append-only — existing records are never modified.
- Queries are read-only — they never mutate production state.
- Write failures are fail-safe — they don't affect the gate decision.
- No raw data, secrets, or paths appear in history files.
- Storage is bounded to prevent unbounded growth.

#### Operator workflow

1. Run the promotion gate (Step 48): the decision is automatically
   saved to history.
2. Query history: `python -m ml.evaluation.promotion_history`.
3. Review past decisions for audit, traceability, or governance.

#### Known limitations

- History files are local to the machine running the gate; there is
  no centralised history store or replication.
- History does not include the full evaluation report — only the
  promotion decision summary. For full reports, use `--output` with
  the promotion gate CLI.
- History retention is bounded; old files are automatically removed
  when the limit is reached.

### Promotion Governance (Step 50 — Centralized Approval Workflow)

**Step 50 NEVER activates models automatically. Approval and
activation are separate concepts. The production model, manifest, and
threshold are never modified by governance operations.**

#### Governance architecture

- `backend/db/promotion_governance.py`: governance record model,
  state machine, and repository (in-memory + PostgreSQL).
- `backend/routers/promotions.py`: authenticated API endpoints for
  creating, listing, approving, rejecting, and marking promotions.
- `backend/schemas.py`: Pydantic request/response schemas.
- Governance records are persisted in the `promotion_governance`
  PostgreSQL table (created idempotently at startup).

#### Governance state machine

```
PENDING → APPROVED → PROMOTED
PENDING → REJECTED
```

- `PENDING` — gate decision recorded, awaiting human review.
- `APPROVED` — reviewer approved; ready for Step 46 activation.
- `REJECTED` — reviewer rejected the promotion.
- `PROMOTED` — operator confirmed Step 46 activation completed.

Invalid transitions (e.g. REJECTED → APPROVED, PROMOTED → PENDING)
are rejected with HTTP 409.

#### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/promotions` | POST | Create governance record from gate decision |
| `/api/v1/promotions` | GET | List records (paginated, filterable) |
| `/api/v1/promotions/{id}` | GET | Get single record |
| `/api/v1/promotions/{id}/approve` | POST | Approve (PENDING → APPROVED) |
| `/api/v1/promotions/{id}/reject` | POST | Reject (PENDING → REJECTED) |
| `/api/v1/promotions/{id}/mark-promoted` | POST | Mark promoted (APPROVED → PROMOTED) |

All endpoints require authentication. Only `fraud_analyst` and
`admin` roles are authorised. Customers are denied (HTTP 403).

#### Security controls

- **Actor identity from JWT**: reviewer/approver identity always
  comes from the authenticated JWT — never from the request payload.
- **No impersonation**: clients cannot supply a reviewer ID.
- **Role enforcement**: `require_roles("fraud_analyst", "admin")` on
  all endpoints.
- **Concurrency safety**: PostgreSQL uses `SELECT ... FOR UPDATE`
  to prevent race conditions on concurrent approve/reject.
- **Idempotency**: unique constraint on (candidate_version,
  candidate_checksum, gate_decision) prevents duplicate records.
- **Bounded inputs**: comments/reasons limited to 500 chars; model
  versions to 100 chars; checksums to 128 chars.
- **No automatic activation**: approval does not modify the manifest,
  threshold, or runtime model.

#### Audit trail integration

Every governance event is recorded in the Step 45 audit trail:

- `PROMOTION_CREATED` — governance record created
- `PROMOTION_APPROVED` — reviewer approved
- `PROMOTION_REJECTED` — reviewer rejected
- `PROMOTION_MARKED_PROMOTED` — operator confirmed activation

Audit records contain actor identity, promotion ID, and relevant
model/promotion metadata. No raw data, secrets, or paths are stored.

#### Operator workflow

1. Run the promotion gate (Step 48): `python -m ml.evaluation.promotion_gate --candidate-model-dir <dir>`.
2. If APPROVED, create a governance record: `POST /api/v1/promotions`.
3. Review the record: `GET /api/v1/promotions/{id}`.
4. Approve or reject: `POST .../approve` or `POST .../reject`.
5. If approved and activation is warranted, perform Step 46 activation
   (explicit operator action).
6. Confirm activation: `POST .../mark-promoted`.

#### Known limitations

- Governance records are stored in the backend database; there is no
  cross-service replication.
- The governance workflow does not perform the Step 46 activation
  itself — it only records that it was performed.
- No notification mechanism (email/webhook) for pending approvals.

## Data Policy

- **No real banking or customer data** is used at any stage.
- Synthetic data generators and legitimate public datasets (e.g., IEEE-CIS, PaySim) are used for training and evaluation.
- Training data is stored in `ml/datasets/` and excluded from version control via `.gitignore`.

## Status

Implemented. Feature engineering, model training, behaviour/rules engines, risk aggregation, explainability, monitoring, hardening, model lifecycle governance (Step 46: manifest-based integrity verification, registry activation, rollback safety), offline model evaluation with threshold governance (Step 47: metrics, threshold sweep, cost analysis, calibration, labelled recommendations — strictly evaluation-only), automated model validation & promotion gate (Step 48: offline candidate validation, configurable policy gates, fail-closed behaviour, bounded safe reports — strictly evaluation-only, no automatic activation), promotion history & audit trail (Step 49: append-only decision persistence, bounded storage, fail-safe writes, read-only queries — strictly audit-only, no production mutation), and centralized promotion governance & approval workflow (Step 50: authenticated governance records, state machine, PostgreSQL persistence, concurrency safety, audit trail integration — no automatic activation, approval ≠ activation) are complete.
