"""
AgriSense India — Yield Prediction + SHAP Explainability (Day 9)
File: notebooks/06_yield_shap.py

Part A: Yield regression model (XGBoost Regressor)
Part B: SHAP waterfall + beeswarm + feature importance charts

Input : data/raw/crop_recommendation.csv (always available)
        data/processed/india_production_clean.csv (optional — richer features)
Output: models/yield_model.pkl
        models/yield_scaler.pkl
        assets/shap_waterfall.png
        assets/shap_beeswarm.png
        assets/shap_feature_importance.png

Run: python notebooks/06_yield_shap.py
"""
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — avoids display errors
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error
from xgboost                 import XGBRegressor
import shap

os.makedirs("models", exist_ok=True)
os.makedirs("assets", exist_ok=True)

print("=" * 60)
print("AgriSense India — Yield Model + SHAP (Day 9)")
print("=" * 60)


# ════════════════════════════════════════════════════════════
# PART A — YIELD REGRESSION MODEL
# ════════════════════════════════════════════════════════════

# ── Load crop_recommendation.csv ────────────────────────────
REC_PATHS = [
    "data/raw/crop_recommendation.csv",
    "data/raw/Crop_recommendation.csv",
]
rec_df = None
for p in REC_PATHS:
    if os.path.exists(p):
        rec_df = pd.read_csv(p)
        rec_df.columns = rec_df.columns.str.strip()
        if "label" in rec_df.columns:
            rec_df = rec_df.rename(columns={"label": "crop"})
        if "Label" in rec_df.columns:
            rec_df = rec_df.rename(columns={"Label": "crop"})
        rec_df["crop"] = rec_df["crop"].str.strip().str.lower()
        print(f"\nLoaded rec_df: {rec_df.shape}")
        break

if rec_df is None:
    print("ERROR: crop_recommendation.csv not found.")
    exit(1)

# ── Try to load production data for richer yield target ─────
prod_path = "data/processed/india_production_clean.csv"
use_prod  = False

if os.path.exists(prod_path):
    prod_df = pd.read_csv(prod_path)
    # Check if yield_kg_per_ha column exists
    if "yield_kg_per_ha" in prod_df.columns:
        prod_df = prod_df.dropna(subset=["yield_kg_per_ha"])
        prod_df = prod_df[prod_df["yield_kg_per_ha"] > 0]
        prod_df = prod_df[prod_df["yield_kg_per_ha"] < 100000]  # remove outliers
        print(f"Loaded prod_df: {prod_df.shape}")
        use_prod = True
    else:
        print("prod_df missing yield_kg_per_ha — computing from area/production...")
        area_col = next((c for c in prod_df.columns if "area" in c.lower()), None)
        prod_col = next((c for c in prod_df.columns if "production" in c.lower()), None)
        if area_col and prod_col:
            prod_df[area_col]  = pd.to_numeric(prod_df[area_col],  errors="coerce")
            prod_df[prod_col]  = pd.to_numeric(prod_df[prod_col],  errors="coerce")
            prod_df = prod_df[prod_df[area_col] > 0]
            prod_df["yield_kg_per_ha"] = (prod_df[prod_col] * 1000 / prod_df[area_col]).round(2)
            prod_df = prod_df[prod_df["yield_kg_per_ha"].between(10, 100000)]
            print(f"  Computed yield for {len(prod_df)} rows")
            use_prod = True

# ── Build training data ──────────────────────────────────────
# Strategy: use rec_df features (N,P,K,temp,hum,ph,rain) + encoded crop
# Simulate yield as a function of these features if prod data unavailable

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

le_crop = LabelEncoder()
rec_df["crop_enc"] = le_crop.fit_transform(rec_df["crop"])

if use_prod:
    # Map crop names from production to rec encoding
    crop_col = next((c for c in prod_df.columns if c.lower() == "crop"), None)
    if crop_col:
        prod_df[crop_col] = prod_df[crop_col].str.strip().str.lower()
        # Merge on crop name to get soil/climate features from rec_df averages
        rec_means = rec_df.groupby("crop")[FEATURE_COLS].mean().reset_index()
        merged = prod_df.merge(rec_means, left_on=crop_col,
                               right_on="crop", how="inner")
        if len(merged) > 500:
            X_yield = merged[FEATURE_COLS].values
            y_yield = merged["yield_kg_per_ha"].values
            print(f"\nUsing production-merged data: {len(merged)} samples")
        else:
            use_prod = False

if not use_prod:
    # Simulate yield using rec_df (deterministic, reproducible)
    print("\nSimulating yield from crop_recommendation features...")
    np.random.seed(42)
    # Yield roughly correlates with N, rainfall, temp in known ways
    y_yield = (
        rec_df["N"] * 8.5
        + rec_df["rainfall"] * 0.6
        + rec_df["K"] * 3.0
        - np.abs(rec_df["ph"] - 6.5) * 80
        + rec_df["temperature"] * 15
        + np.random.normal(0, 200, len(rec_df))
    ).clip(100, 8000)

    X_yield = rec_df[FEATURE_COLS].values
    print(f"  Simulated yield: mean={y_yield.mean():.0f}  "
          f"std={y_yield.std():.0f}  range=[{y_yield.min():.0f}, {y_yield.max():.0f}]")

# ── Train/test split ─────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X_yield, y_yield, test_size=0.2, random_state=42
)

yield_scaler = StandardScaler()
X_tr_sc = yield_scaler.fit_transform(X_tr)
X_te_sc = yield_scaler.transform(X_te)

print(f"\nTrain: {len(X_tr)} | Test: {len(X_te)}")

# ── Train XGBoost Regressor ──────────────────────────────────
print("\nTraining XGBoost Regressor...")
xgb_reg = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
xgb_reg.fit(X_tr_sc, y_tr)
y_pred = xgb_reg.predict(X_te_sc)

rmse = np.sqrt(mean_squared_error(y_te, y_pred))
mae  = mean_absolute_error(y_te, y_pred)
r2   = r2_score(y_te, y_pred)

print(f"  RMSE : {rmse:.1f} kg/ha")
print(f"  MAE  : {mae:.1f} kg/ha")
print(f"  R²   : {r2:.4f}")

# Save models
joblib.dump(xgb_reg,      "models/yield_model.pkl")
joblib.dump(yield_scaler, "models/yield_scaler.pkl")
joblib.dump(FEATURE_COLS, "models/yield_feature_cols.pkl")
print("\nSaved: models/yield_model.pkl")
print("Saved: models/yield_scaler.pkl")


# ════════════════════════════════════════════════════════════
# PART B — SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════
print("\n" + "-" * 40)
print("Computing SHAP values...")

# Use a smaller background sample for speed
background_size = min(100, len(X_tr_sc))
X_bg = X_tr_sc[:background_size]

explainer   = shap.TreeExplainer(xgb_reg)
shap_values = explainer.shap_values(X_te_sc)

print(f"  SHAP values computed for {len(X_te_sc)} test samples")
print(f"  Shape: {np.array(shap_values).shape}")

feature_names_display = ["N (nitrogen)", "P (phosphorus)", "K (potassium)",
                          "Temperature", "Humidity", "Soil pH", "Rainfall"]


# ── SHAP Waterfall — single prediction explanation ───────────
print("\n[SHAP 1] Waterfall plot (single prediction)...")

sample_idx = 0
sample_shap = shap_values[sample_idx]
sample_feat = X_te_sc[sample_idx]
base_val    = explainer.expected_value
pred_val    = float(xgb_reg.predict(X_te_sc[sample_idx:sample_idx+1])[0])

# Sort by absolute SHAP value
order = np.argsort(np.abs(sample_shap))
sorted_names = [feature_names_display[i] for i in order]
sorted_shap  = sample_shap[order]

# Build waterfall manually (compatible with all shap versions)
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in sorted_shap]
y_pos  = range(len(sorted_names))

ax.barh(list(y_pos), sorted_shap, color=colors,
        edgecolor="white", height=0.65, alpha=0.85)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(sorted_names, fontsize=10)
ax.axvline(x=0, color="#555", linewidth=0.8)
ax.set_xlabel("SHAP value (impact on yield prediction kg/ha)", fontsize=10)
ax.set_title(
    f"Why did the model predict {pred_val:.0f} kg/ha?\n"
    f"(Base value: {base_val:.0f} kg/ha)",
    fontsize=12, fontweight="bold", pad=12,
)
# Add value labels
for y, v in zip(y_pos, sorted_shap):
    ax.text(v + (0.5 if v >= 0 else -0.5),
            y, f"{v:+.1f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9, color="#333")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2, linestyle="--")

green_patch = mpatches.Patch(color="#1D9E75", alpha=0.85, label="Increases yield")
red_patch   = mpatches.Patch(color="#D85A30", alpha=0.85, label="Decreases yield")
ax.legend(handles=[green_patch, red_patch], fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig("assets/shap_waterfall.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: assets/shap_waterfall.png")


# ── SHAP Beeswarm — global feature importance ────────────────
print("[SHAP 2] Beeswarm / summary plot...")

# Mean absolute SHAP per feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)
order_global  = np.argsort(mean_abs_shap)
sorted_feat_g = [feature_names_display[i] for i in order_global]
sorted_shap_g = mean_abs_shap[order_global]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(len(sorted_feat_g)), sorted_shap_g,
        color="#7F77DD", alpha=0.85, edgecolor="white", height=0.65)
ax.set_yticks(range(len(sorted_feat_g)))
ax.set_yticklabels(sorted_feat_g, fontsize=11)
ax.set_xlabel("Mean |SHAP value| — average impact on yield (kg/ha)", fontsize=10)
ax.set_title("Global feature importance — SHAP values\n(Which features drive yield prediction most?)",
             fontsize=12, fontweight="bold", pad=12)
for i, v in enumerate(sorted_shap_g):
    ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=9, color="#555")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("assets/shap_beeswarm.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: assets/shap_beeswarm.png")


# ── SHAP Feature importance (XGBoost native) ─────────────────
print("[SHAP 3] XGBoost native feature importance...")

xgb_imp = xgb_reg.feature_importances_
order_xgb = np.argsort(xgb_imp)
feat_sorted = [FEATURE_COLS[i] for i in order_xgb]
imp_sorted  = xgb_imp[order_xgb]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(range(len(feat_sorted)), imp_sorted,
               color="#EF9F27", alpha=0.85, edgecolor="white", height=0.65)
ax.set_yticks(range(len(feat_sorted)))
ax.set_yticklabels([f.capitalize() for f in feat_sorted], fontsize=11)
ax.set_xlabel("Feature importance score (XGBoost)", fontsize=10)
ax.set_title("XGBoost yield model — feature importance",
             fontsize=12, fontweight="bold", pad=12)
for i, v in enumerate(imp_sorted):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9, color="#555")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("assets/shap_feature_importance.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: assets/shap_feature_importance.png")


# ── Plain-English SHAP summary ───────────────────────────────
print("\n" + "-" * 40)
print("Plain-English SHAP summary (what the model learned):")
top3 = np.argsort(mean_abs_shap)[::-1][:3]
for i, idx in enumerate(top3, 1):
    print(f"  {i}. {feature_names_display[idx]:20s} — avg impact: "
          f"{mean_abs_shap[idx]:.1f} kg/ha on yield prediction")


print("\n" + "=" * 60)
print("Day 9 complete!")
print("Files saved:")
for f in ["models/yield_model.pkl", "models/yield_scaler.pkl",
          "assets/shap_waterfall.png", "assets/shap_beeswarm.png",
          "assets/shap_feature_importance.png"]:
    status = "OK" if os.path.exists(f) else "MISSING"
    print(f"  [{status}] {f}")
print("\nNext: python notebooks/07_model_report.py")
print("=" * 60)
