# ML

Machine learning pipeline for fraud prediction, behavioural anomaly detection, and feature engineering.

## Planned Components

- **data/** — Data loading, validation, and preprocessing (synthetic/public datasets only)
- **features/** — Feature engineering from transaction and customer profile data
- **models/** — Model training, evaluation, and serialisation (scikit-learn, XGBoost)
- **inference/** — Real-time scoring service
- **behaviour/** — Behavioural anomaly analysis engine
- **explainability/** — Model explanation and feature importance extraction

## Fraud Signals

- Transaction amount and deviation from customer history
- Location and location change
- Device and new device flag
- Transaction time and frequency/velocity
- Merchant and transaction type
- Historical spending behaviour
- Previous suspicious activity

## Status

Not yet implemented. Foundation placeholder only.
