import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, r2_score, mean_squared_error,
                              precision_score, recall_score, f1_score, mean_absolute_error)

os.makedirs("models", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/startup_data.csv")
print(f"Loaded {len(df)} rows")

# ── Encode categoricals ────────────────────────────────────────────────────────
le_industry = LabelEncoder()
le_stage    = LabelEncoder()
df["industry_enc"]      = le_industry.fit_transform(df["industry"])
df["funding_stage_enc"] = le_stage.fit_transform(df["funding_stage"])

# ── Features ───────────────────────────────────────────────────────────────────
FEATURES_CLASS = [
    "funding_amount_m", "team_size", "years_active", "market_size_m",
    "num_competitors", "product_score", "customer_count",
    "marketing_spend_m", "industry_enc", "funding_stage_enc"
]
FEATURES_REG = [
    "funding_amount_m", "team_size", "years_active", "market_size_m",
    "revenue_year1_m", "burn_rate_m", "product_score",
    "customer_count", "marketing_spend_m", "industry_enc", "funding_stage_enc"
]

X_cls = df[FEATURES_CLASS]
y_cls = df["success"]

X_reg = df[FEATURES_REG]
y_reg = df["future_revenue_m"]

# ── Train / test split ─────────────────────────────────────────────────────────
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# ── Scale ──────────────────────────────────────────────────────────────────────
sc_cls = StandardScaler()
X_tr_c_sc = sc_cls.fit_transform(X_tr_c)
X_te_c_sc = sc_cls.transform(X_te_c)

sc_reg = StandardScaler()
X_tr_r_sc = sc_reg.fit_transform(X_tr_r)
X_te_r_sc = sc_reg.transform(X_te_r)

# ── Logistic Regression ────────────────────────────────────────────────────────
log_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
log_model.fit(X_tr_c_sc, y_tr_c)

y_pred_c = log_model.predict(X_te_c_sc)
acc = accuracy_score(y_te_c, y_pred_c)
print(f"\n── Logistic Regression ──")
print(f"Accuracy : {acc:.4f}")
print(classification_report(y_te_c, y_pred_c, target_names=["Fail","Success"]))

# ── Linear Regression ─────────────────────────────────────────────────────────
lin_model = LinearRegression()
lin_model.fit(X_tr_r_sc, y_tr_r)

y_pred_r = lin_model.predict(X_te_r_sc)
r2  = r2_score(y_te_r, y_pred_r)
mse = mean_squared_error(y_te_r, y_pred_r)
rmse = np.sqrt(mse)
print(f"\n── Linear Regression ──")
print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")

# ── Save metrics for Streamlit app ─────────────────────────────────────────────
# Additional classification metrics
precision = precision_score(y_te_c, y_pred_c)
recall = recall_score(y_te_c, y_pred_c)
f1 = f1_score(y_te_c, y_pred_c)
cm = confusion_matrix(y_te_c, y_pred_c)

# Additional regression metrics
mae = mean_absolute_error(y_te_r, y_pred_r)
mape = np.mean(np.abs((y_te_r - y_pred_r) / y_te_r)) * 100

# Classification report as dict
class_report = classification_report(y_te_c, y_pred_c, target_names=["Fail", "Success"], output_dict=True)

metrics = {
    'accuracy': acc,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'confusion_matrix': cm,
    'r2': r2,
    'rmse': rmse,
    'mae': mae,
    'mape': mape,
    'classification_report_dict': class_report
}

joblib.dump(metrics, "models/model_metrics.pkl")

# ── Save everything ────────────────────────────────────────────────────────────
joblib.dump(log_model,  "models/logistic_model.pkl")
joblib.dump(lin_model,  "models/linear_model.pkl")
joblib.dump(sc_cls,     "models/scaler_cls.pkl")
joblib.dump(sc_reg,     "models/scaler_reg.pkl")
joblib.dump(le_industry,"models/le_industry.pkl")
joblib.dump(le_stage,   "models/le_stage.pkl")
joblib.dump(FEATURES_CLASS, "models/features_cls.pkl")
joblib.dump(FEATURES_REG,   "models/features_reg.pkl")

print("\n✅ All models and scalers saved to models/")
print("   logistic_model.pkl | linear_model.pkl")
print("   scaler_cls.pkl     | scaler_reg.pkl")
print("   le_industry.pkl    | le_stage.pkl")
print("   model_metrics.pkl   | features files")