# ML Feature Engineering Specification

This document defines the mapping between the planned ML features in
[`docs/ml-architecture.md`](ml-architecture.md) and the IEEE-CIS Fraud Detection
dataset. It serves as the authoritative reference for the feature-engineering
implementation.

> **Status:** Specification finalized. Not yet implemented.

---

## 1. Dataset

| Property | Value |
|---|---|
| **Dataset** | IEEE-CIS Fraud Detection (Kaggle, 2019) |
| **Transaction file** | `ml/datasets/raw/train_transaction.csv` |
| **Transaction rows** | 590,540 |
| **Transaction columns** | 394 |
| **Identity file** | `ml/datasets/raw/train_identity.csv` |
| **Identity rows** | 144,233 |
| **Identity columns** | 41 |
| **Target column** | `isFraud` (binary: 0 = legitimate, 1 = fraud) |
| **Fraud rate** | 3.50 % (20,663 fraudulent of 590,540) |
| **Time span** | ~182 days (~6 months) via `TransactionDT` |

### Transaction Column Groups

| Group | Columns | Count |
|---|---|---|
| Identifiers / target | `TransactionID`, `isFraud` | 2 |
| Time / amount | `TransactionDT`, `TransactionAmt` | 2 |
| Product | `ProductCD` | 1 |
| Card | `card1`–`card6` | 6 |
| Address | `addr1`, `addr2` | 2 |
| Distance | `dist1`, `dist2` | 2 |
| Email domain | `P_emaildomain`, `R_emaildomain` | 2 |
| C-count features | `C1`–`C14` | 14 |
| D-delta features | `D1`–`D15` | 15 |
| M-match features | `M1`–`M9` | 9 |
| V-features | `V1`–`V339` | 339 |

### Identity Column Groups

| Group | Columns | Count |
|---|---|---|
| Identifier | `TransactionID` | 1 |
| Numeric id features | `id_01`–`id_11`, `id_13`, `id_14`, `id_17`–`id_26`, `id_32` | 24 |
| Categorical id features | `id_12`, `id_15`, `id_16`, `id_23`, `id_27`–`id_31`, `id_33`–`id_38` | 14 |
| Device | `DeviceType`, `DeviceInfo` | 2 |

---

## 2. Dataset Join Strategy

```
train_transaction.csv  LEFT JOIN  train_identity.csv  ON TransactionID
```

- **Join type:** Left join — all 590,540 transaction rows are preserved.
- **Identity coverage:** 144,233 rows (24.42 %) have identity data after join.
- **Missing identity:** 449,730 rows (75.58 %) have `NULL` for all identity columns.
- **Result shape:** 590,540 rows × 434 columns (394 + 41 − 1 shared key).

Identity data must **not** be required for prediction. Features derived from identity columns must handle the 75.58 % missing-identity case with a defined default value.

---

## 3. Customer/History Identifier

The project's runtime API uses `customer_id` (UUID) to group transactions per customer. IEEE-CIS has no explicit customer ID column.

**Selected identifier: `card1`**

| Property | Value |
|---|---|
| Column | `card1` (in `train_transaction.csv`) |
| Data type | `int64` |
| Unique values | 13,553 |
| Missing values | 0 (0.00 %) |
| Interpretation | Hashed card number; each unique value represents one payment instrument and, by extension, one cardholder |

All history-dependent features group by `card1` and order by `TransactionDT` (ascending) to simulate chronological transaction sequences.

### Why not other columns?

| Alternative | Reason rejected |
|---|---|
| `card2` | 500 unique values; 1.51 % missing. Too few — represents card BIN, not individual cards. |
| `P_emaildomain` | 59 unique values; 15.99 % missing. Far too coarse. |
| `addr1` + `addr2` | 332 × 74 combinations; 11.13 % missing. Address-based grouping does not represent customers. |

---

## 4. Feature Mapping

The planned-features table in [`docs/ml-architecture.md`](ml-architecture.md#L127-L147) defines **19 features**. This mapping uses that table as the source of truth.

| # | Planned Feature | Source (per docs) | Status | IEEE-CIS Column(s) | Calculation Logic |
|---|---|---|---|---|---|
| 1 | `amount` | Transaction | **Direct** | `TransactionAmt` | Pass-through. No transformation. |
| 2 | `amount_deviation` | Transaction + history | **Derived** | `TransactionAmt`, `card1` | `(TransactionAmt − rolling_mean) / rolling_std` over all prior transactions for the same `card1`. |
| 3 | `amount_to_avg_ratio` | Transaction + history | **Derived** | `TransactionAmt`, `card1` | `TransactionAmt / rolling_mean` over all prior transactions for the same `card1`. |
| 4 | `location_country` | Transaction | **Partial** | `addr2` | Use `addr2` as a geographic proxy. Integer-encode. |
| 5 | `location_is_new` | Transaction + history | **Derived** | `addr2`, `card1` | Flag = 1 if this `addr2` has not appeared in any prior transaction for the same `card1`; else 0. |
| 6 | `location_change` | Transaction + history | **Derived** | `addr2`, `card1` | Flag = 1 if current `addr2` differs from the most recent prior `addr2` for the same `card1`; else 0. |
| 7 | `device_fingerprint` | Transaction | **Partial** | `id_19`, `id_20`, `DeviceType` (identity) | Construct a composite identifier from available identity columns. Encode to integer. |
| 8 | `is_new_device` | Transaction + history | **Derived** | Device fingerprint (constructed), `card1` | Flag = 1 if the constructed device fingerprint has not appeared in any prior transaction for the same `card1`; else 0. |
| 9 | `hour_of_day` | Transaction timestamp | **Derived** | `TransactionDT` | `hour = (TransactionDT % 86400) // 3600` → integer 0–23. Cyclical encoding: sin/cos. |
| 10 | `day_of_week` | Transaction timestamp | **Derived** | `TransactionDT` | `day = (TransactionDT // 86400) % 7` → integer 0–6. Cyclical encoding: sin/cos. |
| 11 | `is_unusual_hour` | Transaction + history | **Derived** | `TransactionDT`, `card1` | Flag = 1 if the current hour falls outside the set of "typical hours" for this `card1` (hours representing ≥ 10 % of prior transactions); else 0. |
| 12 | `tx_velocity_1h` | Transaction history | **Derived** | `TransactionDT`, `card1` | Count of prior transactions for the same `card1` where `TransactionDT ∈ [current_DT − 3600, current_DT)`. |
| 13 | `tx_velocity_24h` | Transaction history | **Derived** | `TransactionDT`, `card1` | Count of prior transactions for the same `card1` where `TransactionDT ∈ [current_DT − 86400, current_DT)`. |
| 14 | `tx_velocity_7d` | Transaction history | **Derived** | `TransactionDT`, `card1` | Count of prior transactions for the same `card1` where `TransactionDT ∈ [current_DT − 604800, current_DT)`. |
| 15 | `merchant_category` | Transaction | **Partial** | `ProductCD` | Integer-encode `ProductCD` (5 values: W, C, H, R, S). |
| 16 | `merchant_is_new` | Transaction + history | **Derived** | `ProductCD`, `card1` | Flag = 1 if this `card1` has not used this `ProductCD` in any prior transaction; else 0. |
| 17 | `transaction_type` | Transaction | **Unavailable** | — | No IEEE-CIS equivalent. Exclude from the ML feature set. |
| 18 | `avg_spend_30d` | Transaction history | **Derived** | `TransactionAmt`, `card1`, `TransactionDT` | Rolling mean of `TransactionAmt` for the same `card1` over prior transactions within `[current_DT − 2592000, current_DT)` (30 days). |
| 19 | `previous_suspicious_count` | Transaction history | **Partial** | `isFraud`, `card1` | Cumulative count of `isFraud == 1` for the same `card1` in all prior transactions. At inference, the backend provides `customer_history.previous_flagged_count`. |

### Availability Summary

| Status | Count | Features |
|---|---|---|
| Direct | 1 | `amount` |
| Derived | 13 | `amount_deviation`, `amount_to_avg_ratio`, `location_is_new`, `location_change`, `is_new_device`, `hour_of_day`, `day_of_week`, `is_unusual_hour`, `tx_velocity_1h`, `tx_velocity_24h`, `tx_velocity_7d`, `merchant_is_new`, `avg_spend_30d` |
| Partial | 4 | `location_country`, `device_fingerprint`, `merchant_category`, `previous_suspicious_count` |
| Unavailable | 1 | `transaction_type` |

---

## 5. Historical Feature Calculations

All historical features use `card1` as the grouping key and `TransactionDT` for temporal ordering.

The dataset must be sorted by `TransactionDT` (ascending) within each `card1` group before computing rolling features. For every transaction, historical statistics are computed using **only prior rows** (strictly earlier `TransactionDT` within the same `card1` group) to prevent data leakage.

### Features requiring only the current transaction

| Feature | Data Needed |
|---|---|
| `amount` | `TransactionAmt` |
| `location_country` | `addr2` |
| `hour_of_day` | `TransactionDT` |
| `day_of_week` | `TransactionDT` |
| `merchant_category` | `ProductCD` |

### Features requiring prior transaction history

| Feature | History Needed | Rolling Computation |
|---|---|---|
| `amount_deviation` | All prior `TransactionAmt` for this `card1` | Expanding mean and std |
| `amount_to_avg_ratio` | All prior `TransactionAmt` for this `card1` | Expanding mean |
| `location_is_new` | All prior `addr2` for this `card1` | Set membership check |
| `location_change` | Most recent prior `addr2` for this `card1` | Equality comparison with last row |
| `is_new_device` | All prior device fingerprints for this `card1` | Set membership check |
| `is_unusual_hour` | All prior hour-of-day values for this `card1` | Hour frequency distribution |
| `tx_velocity_1h` | Prior transactions within 3,600 s | Count |
| `tx_velocity_24h` | Prior transactions within 86,400 s | Count |
| `tx_velocity_7d` | Prior transactions within 604,800 s | Count |
| `merchant_is_new` | All prior `ProductCD` for this `card1` | Set membership check |
| `avg_spend_30d` | Prior `TransactionAmt` within 2,592,000 s | Rolling mean |
| `previous_suspicious_count` | All prior `isFraud` for this `card1` | Cumulative sum |

### Cold-Start Defaults

For the first transaction of each `card1` (no prior history):

| Feature type | Default value | Rationale |
|---|---|---|
| Rolling statistics (`amount_deviation`) | 0.0 | No deviation from a non-existent baseline |
| Ratios (`amount_to_avg_ratio`) | 1.0 | Amount equals its own value; no comparison possible |
| "Is new" flags (`location_is_new`, `is_new_device`, `merchant_is_new`) | 1 | Everything is new on the first transaction |
| "Change" flags (`location_change`) | 0 | No previous state to compare against |
| Velocity counts (`tx_velocity_*`) | 0 | No prior transactions |
| `avg_spend_30d` | Current `TransactionAmt` | Single data point as its own average |
| `is_unusual_hour` | 0 | No baseline established yet |
| `previous_suspicious_count` | 0 | No prior history |

---

## 6. Identity Feature Usage

Identity data covers only **24.42 %** of transactions (144,233 of 590,540).

### Planned features that use identity data

| Feature | Identity Columns | Coverage |
|---|---|---|
| `device_fingerprint` | `id_19` (522 unique, 3.41 % missing within identity), `id_20` (394 unique, 3.45 % missing), `DeviceType` (2 unique + NaN, 2.37 % missing) | 24.42 % of all transactions |
| `is_new_device` | Derived from `device_fingerprint` | Same 24.42 % coverage |

### Device fingerprint construction

The composite device fingerprint is constructed by combining:

- `id_19` — high-cardinality device identifier (522 unique values)
- `id_20` — secondary device identifier (394 unique values)
- `DeviceType` — device class (desktop, mobile)

These three columns are concatenated (as strings) and hashed to produce a single device fingerprint value. The theoretical cardinality is up to 522 × 394 × 2 ≈ 411,000 unique combinations, though the actual count will be lower due to real-world co-occurrence patterns.

### Missing identity handling

| Scenario | `device_fingerprint` | `is_new_device` |
|---|---|---|
| Identity data present | Constructed from `id_19` + `id_20` + `DeviceType` | Computed normally against prior history |
| Identity data absent (75.58 %) | Sentinel value: `0` (encoded "no_device_data") | Default: `0` (absence ≠ new device) |

### Optional supplementary feature

A binary flag `has_identity_data` (1 if identity exists, 0 otherwise) may be added to let the model learn whether identity availability itself correlates with fraud. This would bring the effective feature count from 18 to 19 (replacing the unavailable `transaction_type`).

---

## 7. Partial or Unavailable Features

### Unavailable: `transaction_type`

The project expects a transaction type label (`purchase`, `transfer`, `withdrawal`). IEEE-CIS has no equivalent column:

- `card6` (debit/credit) describes a payment method, not a transaction type.
- `ProductCD` (W/C/H/R/S) describes a product category, not a transaction type.

**Decision:** Exclude `transaction_type` from the ML feature set. The runtime system will still accept and use this field when real data is available, but it will not be a model feature during IEEE-CIS training.

### Partial: `location_country`

| Aspect | Detail |
|---|---|
| IEEE-CIS column | `addr2` (74 unique values, `int64`) |
| Project expectation | ISO country string (e.g., "US", "GB") |
| Gap | `addr2` is an obfuscated numeric code, not a country name |
| Impact | Still useful for distinguishing locations and computing location-based features. Semantic meaning is lost but structural utility is preserved. |

### Partial: `device_fingerprint`

| Aspect | Detail |
|---|---|
| IEEE-CIS columns | `id_19`, `id_20`, `DeviceType` (identity table) |
| Project expectation | A single hashed device fingerprint string |
| Gap | No real fingerprint hash exists. Must be constructed from multiple columns. Only 24.42 % coverage. |
| Impact | The constructed fingerprint has lower cardinality and higher missingness than a real system. |

### Partial: `merchant_category`

| Aspect | Detail |
|---|---|
| IEEE-CIS column | `ProductCD` (5 values: W, C, H, R, S) |
| Project expectation | Merchant Category Code (MCC), e.g., "5732" |
| Gap | ProductCD has only 5 broad product types, not hundreds of MCCs |
| Impact | Captures coarse product-type patterns but lacks merchant-category granularity. |

### Partial: `previous_suspicious_count`

| Aspect | Detail |
|---|---|
| Training source | Cumulative `isFraud == 1` count for same `card1` in prior rows |
| Inference source | `customer_history.previous_flagged_count` from backend API |
| Gap | Training uses the target label; inference uses a separate database field |
| Impact | Acceptable if training correctly restricts to prior rows only (no leakage). The semantics match: both count previous fraud flags for the same customer. |

---

## 8. Training vs Inference Consistency

Every feature must produce the same value given the same input, whether computed during offline training or online inference.

| Feature | Training (IEEE-CIS) | Inference (runtime API) | Consistent? |
|---|---|---|---|
| `amount` | `TransactionAmt` | `transaction.amount` | Yes |
| `amount_deviation` | Rolling z-score by `card1` (prior rows) | Z-score using `customer_history.avg_amount_30d` and `std_amount_30d` | Approximate — training uses all-time expanding window; inference uses 30-day window |
| `amount_to_avg_ratio` | `TransactionAmt / expanding_mean` | `amount / customer_history.avg_amount_30d` | Approximate — same window difference |
| `location_country` | `addr2` | `transaction.location_country` | Yes (different source field, same semantics) |
| `location_is_new` | Set check against prior `addr2` | Check against `customer_history` known locations | Yes |
| `location_change` | Compare with most recent prior `addr2` | Compare with `customer_history.last_transaction_country` | Yes |
| `device_fingerprint` | Constructed from `id_19`+`id_20`+`DeviceType` | `transaction.device_fingerprint` | Yes (different source, same role) |
| `is_new_device` | Set check against prior fingerprints | Check against `customer_history.known_device_fingerprints` | Yes |
| `hour_of_day` | `(TransactionDT % 86400) // 3600` | Extract from `transaction.timestamp` | Yes |
| `day_of_week` | `(TransactionDT // 86400) % 7` | Extract from `transaction.timestamp` | Yes |
| `is_unusual_hour` | Hour distribution check | Hour distribution check | Yes |
| `tx_velocity_1h` | Count within 3,600 s window | Count from recent transactions | Yes |
| `tx_velocity_24h` | Count within 86,400 s window | Count from recent transactions | Yes |
| `tx_velocity_7d` | Count within 604,800 s window | Count from recent transactions | Yes |
| `merchant_category` | `ProductCD` | `transaction.merchant_category` | Yes (different source, same role) |
| `merchant_is_new` | Set check against prior `ProductCD` | Check against `customer_history.known_merchant_ids` | Yes |
| `transaction_type` | **Unavailable** | `transaction.transaction_type` | N/A — excluded from training |
| `avg_spend_30d` | Rolling mean within 2,592,000 s | Computed from `customer_history` or recent transactions | Yes |
| `previous_suspicious_count` | Cumulative `isFraud` (prior only) | `customer_history.previous_flagged_count` | Yes (semantically equivalent) |

### Known approximation: `amount_deviation` and `amount_to_avg_ratio`

During training, these features use an **expanding window** (all prior transactions for the same `card1`) because IEEE-CIS spans 6 months and provides no explicit 30-day pre-aggregation. At inference, the backend provides 30-day statistics (`avg_amount_30d`, `std_amount_30d`). This difference is acceptable because:

1. The expanding window converges toward the 30-day window for active cards.
2. The model learns the statistical pattern, not the exact window.
3. Feature scaling during preprocessing will normalize both to a similar range.

---

## 9. Decisions Required Before Implementation

| # | Decision | Options | Recommended |
|---|---|---|---|
| 1 | **`transaction_type` handling** | (a) Exclude entirely. (b) Constant default. (c) Use `card6` proxy. | **(a) Exclude** — `card6` is not a valid transaction type. |
| 2 | **`device_fingerprint` construction** | (a) Hash of `id_19` + `id_20` + `DeviceType`. (b) `id_19` alone. (c) All low-missingness identity columns. | **(a) Hash of three columns** — balances cardinality and coverage. |
| 3 | **Missing identity: `is_new_device` default** | (a) 0 (not new). (b) 1 (new). (c) Separate "unknown" class. | **(a) 0** — absence of device data should not trigger fraud alerts. |
| 4 | **Add `has_identity_data` flag** | (a) Yes, as supplementary feature. (b) No. | **(a) Yes** — low cost, lets model learn identity-availability signal. |
| 5 | **Cold-start defaults** | (a) Fixed defaults (0.0, 1, 0). (b) Dataset-wide medians. | **(a) Fixed defaults** — deterministic, no cross-customer leakage. |
| 6 | **`addr1` as supplementary location** | (a) Use only `addr2`. (b) Also create `location_region` from `addr1` (332 values). | **(b) Also use `addr1`** — finer geographic granularity at minimal cost. |
| 7 | **Rolling window type for amount features** | (a) Expanding window (all prior). (b) Fixed 30-day window. | **(a) Expanding** — more stable for low-frequency cards. |
| 8 | **Effective feature count** | 18 (exclude `transaction_type`) or 19 (add `has_identity_data`). | **18 + 1 optional** — document both counts. |

---

## 10. Implementation Notes

### Data ordering

The dataset must be sorted by `TransactionDT` (ascending) before any rolling or historical computation. Within each `card1` group, this ensures that "prior" rows are strictly earlier in time.

### Memory considerations

With 590,540 rows and 434 columns (post-join), the dataset is approximately 2 GB in memory as a pandas DataFrame. Rolling computations should be implemented using vectorized pandas/numpy operations where possible. For velocity features, a groupby-apply pattern with sorted windows is recommended.

### Feature pipeline output

The feature engineering pipeline produces a DataFrame with one row per transaction and the following columns:

- `TransactionID` — preserved for join-back
- `isFraud` — target (preserved, not used in feature computation)
- 18 planned features (or 19 with `has_identity_data`)
- Any supplementary features decided in Section 9

### Encoding deferred

Final categorical encoding (label encoding, one-hot encoding, target encoding) and numerical scaling are **not** part of the feature engineering step. They are handled by the preprocessing pipeline during model training, where encoders are fitted on the training split and persisted with the model artifact.

### Files

| File | Purpose |
|---|---|
| `ml/data/loader.py` | Dataset loading and validation (exists) |
| `ml/features/engineer.py` | Feature engineering pipeline (to be created) |
| `ml/features/historical.py` | Rolling and history-dependent features (to be created) |
| `ml/features/identity.py` | Device fingerprint construction (to be created) |

### References

- Planned features: [`docs/ml-architecture.md`](ml-architecture.md) § 1 "Feature Engineering"
- Runtime request schema: [`docs/ml-architecture.md`](ml-architecture.md) § "Request Schema"
- Database schema: [`docs/database-design.md`](database-design.md) — `transactions` table
- ML module plan: [`ml/README.md`](../ml/README.md)
