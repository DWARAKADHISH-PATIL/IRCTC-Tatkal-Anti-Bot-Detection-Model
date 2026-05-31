# IRCTC TATKAL BOT DETECTION MODEL - DEPLOYMENT GUIDE

## Model Overview
- **Model Type**: Ensemble (Logistic Regression, XGBoost, LightGBM)
- **Version**: 1.0.0
- **Created**: 2025-12-22 21:53:33
- **Target**: Detect bots in IRCTC Tatkal booking sessions
- **Best Model**: XGBoost (AUC-ROC = 1.0000)

## Performance Metrics
| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Features (53 total)
form_fill_time_sec, mouse_entropy, keystroke_interval_ms, scroll_depth_pct, page_view_time_sec, request_rate_per_sec, otp_solve_time_sec, device_consistency_score, tls_signature_entropy, geo_variance_km... and 43 more

## Model Files
- `models/xgboost_model.pkl` - Best performing model
- `models/feature_scaler.pkl` - StandardScaler for preprocessing
- `models/smote_balancer.pkl` - SMOTE balancer for training
- `models/model_config.json` - Complete model configuration

## Usage Example
```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Prepare features (must have all 53 engineered features)
X = pd.DataFrame([feature_dict])
X_scaled = scaler.transform(X)

# Predict
prediction = model.predict(X_scaled)
probability = model.predict_proba(X_scaled)[:, 1]

# Output
is_bot = prediction[0]
prob = probability[0]
```

## Inference Requirements
- **Latency**: <50ms per prediction
- **Memory**: ~100MB for all models loaded
- **Dependencies**: scikit-learn, xgboost, lightgbm, pandas, numpy, joblib

## Input Feature Requirements
All 53 features must be provided in the input dictionary with exact names matching model_config.json feature_names list.

## Maintenance & Updates
- Monitor model performance drift monthly
- Retrain on new IRCTC data quarterly
- Update feature importance analysis with new threat patterns
- Validate against live bot attack samples periodically

## Contact & Support
Model developed for IRCTC Tatkal bot detection during peak booking windows (first 5 minutes).
For updates or issues, refer to the project documentation.
