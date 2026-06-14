# %% [markdown]
# # 03 — Model Training (XGBoost + LightGBM Ensemble)

# %%
import os
import sys
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.path.abspath(".")
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "outputs", "models"), exist_ok=True)

from Fraud_detection.src.feature_engineering import build_features
from Fraud_detection.src.model import XGBoostFraudModel, LightGBMFraudModel, FraudEnsemble
from Fraud_detection.src.evaluation import fraud_metrics_at_threshold, plot_fraud_report, recall_at_fpr

logging.basicConfig(level=logging.INFO, format="%(message)s")
TARGET = "isFraud"
DROP_COLS = ["TransactionID", TARGET]

# %%
with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)

transactions = pd.read_csv(os.path.join(ROOT, "data", "raw", "train_transaction.csv"))
identity = pd.read_csv(os.path.join(ROOT, "data", "raw", "train_identity.csv"))
df = build_features(transactions, identity=identity)

FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]
for col in df[FEATURE_COLS].select_dtypes(include=["object"]).columns:
    df[col] = pd.Categorical(df[col]).codes

X = df[FEATURE_COLS].fillna(-999)
y = df[TARGET]
split_idx = int(len(X) * 0.80)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")

# %%
import joblib

ensemble_path = os.path.join(ROOT, "outputs", "models", "fraud_ensemble.pkl")
if os.path.exists(ensemble_path):
    print(f"Loading saved ensemble from {ensemble_path}")
    ensemble = joblib.load(ensemble_path)
    xgb_model = ensemble.xgb_model
else:
    xgb_model = XGBoostFraudModel(
        params=config.get("xgboost"),
        fn_cost=config["costs"]["fn_cost"],
        fp_cost=config["costs"]["fp_cost"],
    )
    xgb_model.fit(X_train, y_train, X_val, y_val)
    lgb_model = LightGBMFraudModel(params=config.get("lightgbm"))
    lgb_model.fit(X_train, y_train, X_val, y_val)
    ensemble = FraudEnsemble(xgb_model, lgb_model, xgb_weight=0.6)
    ensemble.calibrate_threshold(X_val, y_val)
    joblib.dump(ensemble, ensemble_path)

# %%
opt_thresh = ensemble.threshold_optimizer.optimal_threshold_
y_ens_score_val = ensemble.predict_proba(X_val)
metrics = fraud_metrics_at_threshold(y_val.values, y_ens_score_val, opt_thresh)
metrics_df = pd.DataFrame([metrics])
display(metrics_df.T.rename(columns={0: "value"}))

recall_5fpr = recall_at_fpr(y_val.values, y_ens_score_val, 0.05)
print(f"Recall @ 5% FPR: {recall_5fpr:.4f}")

# %%
plot_fraud_report(
    y_train.values, xgb_model.predict_proba(X_train),
    y_val.values, y_ens_score_val,
    optimal_threshold=opt_thresh,
    model_name="XGBoost+LightGBM Ensemble",
    save_path=os.path.join(ROOT, "outputs", "fraud_model_report.png"),
)
plt.show()

fi = xgb_model.feature_importance()
fi.to_csv(os.path.join(ROOT, "outputs", "feature_importance.csv"), index=False)
display(fi.head(10))

fig, ax = plt.subplots(figsize=(8, 6))
top = fi.head(15)
ax.barh(top["feature"][::-1], top["importance"][::-1], color="#2563EB")
ax.set_title("Top 15 XGBoost feature importances")
plt.tight_layout()
out = os.path.join(ROOT, "outputs", "feature_importance_top15.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}")

# %%
import joblib
joblib.dump(ensemble, os.path.join(ROOT, "outputs", "models", "fraud_ensemble.pkl"))
print("Ensemble saved to outputs/models/fraud_ensemble.pkl")
