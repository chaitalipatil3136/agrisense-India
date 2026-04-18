"""
AgriSense India — Crop Recommendation ML Model (Day 8)
File: notebooks/05_crop_model.py

Trains two classifiers on crop_recommendation.csv:
  1. Random Forest  (target: ~99% accuracy)
  2. XGBoost        (target: ~98% accuracy)

Saves models + label encoder + comparison report.

Input : data/raw/crop_recommendation.csv
Output: models/rf_crop_model.pkl
        models/xgb_crop_model.pkl
        models/label_encoder.pkl
        models/scaler.pkl
        models/model_comparison.csv
        assets/model_comparison_chart.png
        assets/feature_importance.png

Run: python notebooks/05_crop_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.metrics          import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

os.makedirs("models",  exist_ok=True)
os.makedirs("assets",  exist_ok=True)

print("=" * 60)
print("AgriSense India — Crop Recommendation Model (Day 8)")
print("=" * 60)


# ── Step 1: Load crop_recommendation.csv directly ────────────
REC_PATHS = [
    "data/raw/crop_recommendation.csv",
    "data/raw/Crop_recommendation.csv",
]
df = None
for p in REC_PATHS:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"\nLoaded: {p}  ({df.shape[0]} rows, {df.shape[1]} cols)")
        break

if df is None:
    print("\nERROR: crop_recommendation.csv not found in data/raw/")
    print("Download from: kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
    exit(1)

# Standardize column names
df.columns = df.columns.str.strip()
if "label" in df.columns:
    df = df.rename(columns={"label": "crop"})
if "Label" in df.columns:
    df = df.rename(columns={"Label": "crop"})

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL   = "crop"

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"\nERROR: Missing columns: {missing}")
    print(f"Available: {list(df.columns)}")
    exit(1)

print(f"Columns   : {FEATURE_COLS + [TARGET_COL]}")
print(f"Crops     : {sorted(df[TARGET_COL].unique())}")
print(f"Missing   : {df.isnull().sum().sum()}")


# ── Step 2: Encode labels ────────────────────────────────────
le = LabelEncoder()
df["crop_encoded"] = le.fit_transform(df[TARGET_COL])

print(f"\nLabel encoding:")
for i, cls in enumerate(le.classes_):
    print(f"  {i:2d} → {cls}")


# ── Step 3: Features / target / split ────────────────────────
X = df[FEATURE_COLS].values
y = df["crop_encoded"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standard scaler (used by XGBoost — RF doesn't need it but keeps consistent)
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain set : {len(X_train)} samples")
print(f"Test set  : {len(X_test)}  samples")


# ── Step 4: Train Random Forest ──────────────────────────────
print("\n" + "-" * 40)
print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_acc  = accuracy_score(y_test, y_pred_rf)
rf_f1   = f1_score(y_test, y_pred_rf, average="weighted")
rf_prec = precision_score(y_test, y_pred_rf, average="weighted", zero_division=0)
rf_rec  = recall_score(y_test, y_pred_rf, average="weighted", zero_division=0)

# Cross-validation
rf_cv = cross_val_score(rf, X, y, cv=5, scoring="accuracy")

print(f"  Accuracy  : {rf_acc:.4f}  ({rf_acc*100:.2f}%)")
print(f"  F1 score  : {rf_f1:.4f}")
print(f"  Precision : {rf_prec:.4f}")
print(f"  Recall    : {rf_rec:.4f}")
print(f"  CV (5-fold): {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")


# ── Step 5: Train XGBoost ────────────────────────────────────
print("\n" + "-" * 40)
print("Training XGBoost...")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb.fit(X_train_sc, y_train)
y_pred_xgb = xgb.predict(X_test_sc)

xgb_acc  = accuracy_score(y_test, y_pred_xgb)
xgb_f1   = f1_score(y_test, y_pred_xgb, average="weighted")
xgb_prec = precision_score(y_test, y_pred_xgb, average="weighted", zero_division=0)
xgb_rec  = recall_score(y_test, y_pred_xgb, average="weighted", zero_division=0)

xgb_cv = cross_val_score(
    XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                  use_label_encoder=False, eval_metric="mlogloss",
                  random_state=42, verbosity=0),
    X_train_sc, y_train, cv=5, scoring="accuracy"
)

print(f"  Accuracy  : {xgb_acc:.4f}  ({xgb_acc*100:.2f}%)")
print(f"  F1 score  : {xgb_f1:.4f}")
print(f"  Precision : {xgb_prec:.4f}")
print(f"  Recall    : {xgb_rec:.4f}")
print(f"  CV (5-fold): {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")


# ── Step 6: Classification report ────────────────────────────
print("\n" + "-" * 40)
print("Classification report (Random Forest — best model):")
print(classification_report(y_test, y_pred_rf,
                             target_names=le.classes_))


# ── Step 7: Save models ───────────────────────────────────────
joblib.dump(rf,     "models/rf_crop_model.pkl")
joblib.dump(xgb,    "models/xgb_crop_model.pkl")
joblib.dump(le,     "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\nSaved:")
print("  models/rf_crop_model.pkl")
print("  models/xgb_crop_model.pkl")
print("  models/label_encoder.pkl")
print("  models/scaler.pkl")


# ── Step 8: Save comparison CSV ──────────────────────────────
comparison = pd.DataFrame({
    "model":     ["Random Forest", "XGBoost"],
    "accuracy":  [round(rf_acc, 4),  round(xgb_acc, 4)],
    "f1_score":  [round(rf_f1, 4),   round(xgb_f1, 4)],
    "precision": [round(rf_prec, 4), round(xgb_prec, 4)],
    "recall":    [round(rf_rec, 4),  round(xgb_rec, 4)],
    "cv_mean":   [round(rf_cv.mean(), 4), round(xgb_cv.mean(), 4)],
    "cv_std":    [round(rf_cv.std(), 4),  round(xgb_cv.std(), 4)],
})
comparison.to_csv("models/model_comparison.csv", index=False)
print("  models/model_comparison.csv")


# ── Step 9: Model comparison chart ───────────────────────────
metrics = ["accuracy", "f1_score", "precision", "recall"]
rf_vals  = [rf_acc,  rf_f1,  rf_prec,  rf_rec]
xgb_vals = [xgb_acc, xgb_f1, xgb_prec, xgb_rec]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, rf_vals,  width, label="Random Forest",
               color="#1D9E75", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, xgb_vals, width, label="XGBoost",
               color="#7F77DD", alpha=0.85, edgecolor="white")

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=11)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model comparison — Random Forest vs XGBoost\n(Crop Recommendation)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig("assets/model_comparison_chart.png", dpi=200, bbox_inches="tight")
plt.close()
print("  assets/model_comparison_chart.png")


# ── Step 10: Feature importance chart ────────────────────────
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [FEATURE_COLS[i] for i in indices]
sorted_imp      = importances[indices]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(len(sorted_features)), sorted_imp[::-1],
        color="#1D9E75", alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(sorted_features)))
ax.set_yticklabels([f.capitalize() for f in sorted_features[::-1]], fontsize=11)
ax.set_xlabel("Feature importance (Mean decrease in impurity)", fontsize=10)
ax.set_title("Random Forest — Feature importance for crop recommendation",
             fontsize=12, fontweight="bold", pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig("assets/feature_importance.png", dpi=200, bbox_inches="tight")
plt.close()
print("  assets/feature_importance.png")


# ── Quick prediction test ────────────────────────────────────
print("\n" + "-" * 40)
print("Quick prediction test:")
sample = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]])   # should be rice
pred_encoded = rf.predict(sample)[0]
pred_crop    = le.inverse_transform([pred_encoded])[0]
proba        = rf.predict_proba(sample)[0]
top3_idx     = np.argsort(proba)[::-1][:3]
print(f"  Input: N=90, P=42, K=43, Temp=20.8, Humidity=82, pH=6.5, Rainfall=202.9")
print(f"  Prediction: {pred_crop.upper()}")
print(f"  Top 3 crops:")
for idx in top3_idx:
    print(f"    {le.inverse_transform([idx])[0]:15s} {proba[idx]*100:.1f}%")

print("\n" + "=" * 60)
print("Day 8 complete! Models saved.")
print("Next: python notebooks/06_yield_shap.py")
print("=" * 60)
