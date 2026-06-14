# %% [markdown]
# # 04 — Threshold Optimization (Business Cost)

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.path.abspath(".")
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

from Fraud_detection.src.feature_engineering import build_features
from Fraud_detection.src.model import FraudEnsemble, LightGBMFraudModel, XGBoostFraudModel
from Fraud_detection.src.evaluation import fraud_metrics_at_threshold

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

import joblib

ensemble_path = os.path.join(ROOT, "outputs", "models", "fraud_ensemble.pkl")
if os.path.exists(ensemble_path):
    print(f"Loading saved ensemble from {ensemble_path}")
    ensemble = joblib.load(ensemble_path)
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

y_score = ensemble.predict_proba(X_val)
cost_curve = ensemble.threshold_optimizer.cost_curve_
display(cost_curve.head(10))

# %%
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(cost_curve["threshold"], cost_curve["total_cost"], color="#DC2626", lw=2, label="Total cost")
ax2.plot(cost_curve["threshold"], cost_curve["recall"], color="#2563EB", lw=1.5, linestyle="--", label="Recall")
opt = ensemble.threshold_optimizer.optimal_threshold_
ax1.axvline(opt, color="green", linestyle="--", lw=2, label=f"Optimal={opt:.3f}")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Total cost ($)", color="#DC2626")
ax2.set_ylabel("Recall", color="#2563EB")
ax1.set_title("Cost-based threshold optimization")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
plt.tight_layout()
out = os.path.join(ROOT, "outputs", "threshold_cost_curve.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}")

# %%
thresholds = [0.1, 0.2, 0.3, 0.5, opt]
rows = [fraud_metrics_at_threshold(y_val.values, y_score, t) for t in thresholds]
comparison = pd.DataFrame(rows)
display(comparison[["threshold", "recall", "precision", "fpr", "total_cost", "cost_savings_vs_baseline"]])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(comparison["threshold"], comparison["recall"], marker="o", label="Recall")
ax.plot(comparison["threshold"], comparison["precision"], marker="s", label="Precision")
ax.set_xlabel("Threshold")
ax.set_title("Precision / Recall vs threshold")
ax.legend()
plt.tight_layout()
out2 = os.path.join(ROOT, "outputs", "threshold_precision_recall.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out2}")
