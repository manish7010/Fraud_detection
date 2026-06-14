# %% [markdown]
# # 02 — Feature Engineering

# %%
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.path.abspath(".")
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

from Fraud_detection.src.data_utils import load_raw
from Fraud_detection.src.feature_engineering import build_features, get_feature_groups

# %%
transactions, identity = load_raw(os.path.join(ROOT, "data", "raw"))
df = build_features(transactions, identity=identity)
print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")

# %%
groups = get_feature_groups()
rows = []
for name, cols in groups.items():
    present = [c for c in cols if c in df.columns]
    rows.append({"group": name, "features_present": len(present), "features": ", ".join(present[:5])})
feat_summary = pd.DataFrame(rows)
display(feat_summary)

# %%
engineered = [c for c in df.columns if c not in transactions.columns]
sample = df[engineered + ["isFraud"]].describe().T
display(sample.head(15))

# %%
plot_cols = [c for c in ["amt_zscore_from_card_mean", "w1h_txn_count", "unique_cards_on_device"] if c in df.columns]
if plot_cols:
    fig, axes = plt.subplots(1, len(plot_cols), figsize=(5 * len(plot_cols), 4))
    if len(plot_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, plot_cols):
        for label, color in [(0, "#2563EB"), (1, "#DC2626")]:
            subset = df.loc[df["isFraud"] == label, col].dropna()
            ax.hist(subset, bins=50, alpha=0.5, density=True, label="Fraud" if label else "Legit", color=color)
        ax.set_title(col)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(ROOT, "outputs", "feature_engineering_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")
