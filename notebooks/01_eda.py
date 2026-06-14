# %% [markdown]
# # 01 — Exploratory Data Analysis (IEEE-CIS Fraud Detection)

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.path.abspath(".")
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

sns.set_theme(style="whitegrid")
DATA_DIR = os.path.join(ROOT, "data", "raw")

# %%
transactions = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))
identity = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))
print(f"Transactions: {transactions.shape}")
print(f"Identity: {identity.shape}")
print(f"Fraud rate: {transactions['isFraud'].mean():.4%}")

# %%
display_cols = ["TransactionAmt", "isFraud", "ProductCD", "card4", "TransactionDT"]
display(transactions[display_cols].head(10))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fraud_rate = transactions["isFraud"].mean()
axes[0].bar(["Legit", "Fraud"], [1 - fraud_rate, fraud_rate], color=["#2563EB", "#DC2626"])
axes[0].set_title("Class balance")
axes[0].set_ylabel("Share of transactions")

log_amt = np.log1p(transactions["TransactionAmt"])
axes[1].hist(log_amt[transactions["isFraud"] == 0], bins=60, alpha=0.6, label="Legit", density=True)
axes[1].hist(log_amt[transactions["isFraud"] == 1], bins=60, alpha=0.6, label="Fraud", density=True)
axes[1].set_title("Log transaction amount by class")
axes[1].legend()
plt.tight_layout()
out = os.path.join(ROOT, "outputs", "eda_class_balance_amount.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}")

# %%
if "ProductCD" in transactions.columns:
    fraud_by_product = (
        transactions.groupby("ProductCD")["isFraud"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fraud_rate", "count": "n"})
        .sort_values("fraud_rate", ascending=False)
    )
    display(fraud_by_product.head(10))

    fig, ax = plt.subplots(figsize=(8, 4))
    fraud_by_product["fraud_rate"].plot(kind="bar", ax=ax, color="#7C3AED")
    ax.set_title("Fraud rate by ProductCD")
    ax.set_ylabel("Fraud rate")
    plt.tight_layout()
    out = os.path.join(ROOT, "outputs", "eda_fraud_rate_by_product.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")

# %%
merged = transactions.merge(identity, on="TransactionID", how="left", indicator=True)
coverage = merged["_merge"].value_counts(normalize=True)
print("Identity join coverage:")
print(coverage)
display(coverage.to_frame("share"))
