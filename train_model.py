import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
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
# (Scaling isn't required for tree-based models, but we reuse the same scaled
#  features for every model so all classifiers/regressors share one input
#  pipeline and the app doesn't need to branch its preprocessing per model.)
sc_cls = StandardScaler()
X_tr_c_sc = sc_cls.fit_transform(X_tr_c)
X_te_c_sc = sc_cls.transform(X_te_c)

sc_reg = StandardScaler()
X_tr_r_sc = sc_reg.fit_transform(X_tr_r)
X_te_r_sc = sc_reg.transform(X_te_r)

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION MODELS
# ══════════════════════════════════════════════════════════════════════════════
classification_models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    ),
}

cls_metrics = {}
cls_fitted  = {}

print("\n" + "=" * 60)
print("CLASSIFICATION MODELS (Success / Fail)")
print("=" * 60)

for name, model in classification_models.items():
    model.fit(X_tr_c_sc, y_tr_c)
    y_pred = model.predict(X_te_c_sc)

    acc  = accuracy_score(y_te_c, y_pred)
    prec = precision_score(y_te_c, y_pred)
    rec  = recall_score(y_te_c, y_pred)
    f1   = f1_score(y_te_c, y_pred)
    cm   = confusion_matrix(y_te_c, y_pred)
    report = classification_report(y_te_c, y_pred, target_names=["Fail", "Success"], output_dict=True)

    cls_metrics[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report_dict": report,
    }
    cls_fitted[name] = model

    print(f"\n── {name} ──")
    print(f"Accuracy : {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION MODELS
# ══════════════════════════════════════════════════════════════════════════════
regression_models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    ),
}

reg_metrics = {}
reg_fitted  = {}

print("\n" + "=" * 60)
print("REGRESSION MODELS (Future Revenue)")
print("=" * 60)

for name, model in regression_models.items():
    model.fit(X_tr_r_sc, y_tr_r)
    y_pred = model.predict(X_te_r_sc)

    r2   = r2_score(y_te_r, y_pred)
    mse  = mean_squared_error(y_te_r, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_te_r, y_pred)
    mape = np.mean(np.abs((y_te_r - y_pred) / y_te_r)) * 100

    reg_metrics[name] = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }
    reg_fitted[name] = model

    print(f"\n── {name} ──")
    print(f"R² : {r2:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}  MAPE: {mape:.2f}%")

# ── Save metrics for Streamlit app ─────────────────────────────────────────────
# New structure: metrics comparison across all models, keyed by model name.
# We also mirror the *best* classifier/regressor (by F1 / R²) into the old
# top-level keys so nothing else that reads model_metrics.pkl breaks.
best_cls_name = max(cls_metrics, key=lambda k: cls_metrics[k]["f1"])
best_reg_name = max(reg_metrics, key=lambda k: reg_metrics[k]["r2"])

metrics = {
    "classification_models": cls_metrics,
    "regression_models": reg_metrics,
    "best_classification_model": best_cls_name,
    "best_regression_model": best_reg_name,
    # Backwards-compatible top-level keys (mirrors logistic/linear, the
    # original defaults) so any older code paths keep working.
    "accuracy": cls_metrics["logistic_regression"]["accuracy"],
    "precision": cls_metrics["logistic_regression"]["precision"],
    "recall": cls_metrics["logistic_regression"]["recall"],
    "f1": cls_metrics["logistic_regression"]["f1"],
    "confusion_matrix": cls_metrics["logistic_regression"]["confusion_matrix"],
    "r2": reg_metrics["linear_regression"]["r2"],
    "rmse": reg_metrics["linear_regression"]["rmse"],
    "mae": reg_metrics["linear_regression"]["mae"],
    "mape": reg_metrics["linear_regression"]["mape"],
    "classification_report_dict": cls_metrics["logistic_regression"]["classification_report_dict"],
}

joblib.dump(metrics, "models/model_metrics.pkl")

# ── Save everything ────────────────────────────────────────────────────────────
joblib.dump(cls_fitted["logistic_regression"], "models/logistic_model.pkl")
joblib.dump(cls_fitted["random_forest"],       "models/rf_classifier.pkl")
joblib.dump(cls_fitted["gradient_boosting"],   "models/gb_classifier.pkl")

joblib.dump(reg_fitted["linear_regression"],   "models/linear_model.pkl")
joblib.dump(reg_fitted["random_forest"],       "models/rf_regressor.pkl")
joblib.dump(reg_fitted["gradient_boosting"],   "models/gb_regressor.pkl")

joblib.dump(sc_cls,     "models/scaler_cls.pkl")
joblib.dump(sc_reg,     "models/scaler_reg.pkl")
joblib.dump(le_industry,"models/le_industry.pkl")
joblib.dump(le_stage,   "models/le_stage.pkl")
joblib.dump(FEATURES_CLASS, "models/features_cls.pkl")
joblib.dump(FEATURES_REG,   "models/features_reg.pkl")

print("\n✅ All models and scalers saved to models/")
print("   logistic_model.pkl | rf_classifier.pkl | gb_classifier.pkl")
print("   linear_model.pkl   | rf_regressor.pkl   | gb_regressor.pkl")
print("   scaler_cls.pkl     | scaler_reg.pkl")
print("   le_industry.pkl    | le_stage.pkl")
print("   model_metrics.pkl  | features files")
print(f"\n🏆 Best classifier (by F1): {best_cls_name}")
print(f"🏆 Best regressor  (by R²): {best_reg_name}")