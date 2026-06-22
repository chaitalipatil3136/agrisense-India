"""
AgriSense India — Model Evaluation Report (Day 10)
File: notebooks/07_model_report.py

Loads both trained models and prints a complete evaluation report.
Also verifies all Week 2 deliverables are in place.

Run: python notebooks/07_model_report.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("AgriSense India — Model Evaluation Report (Day 10)")
print("=" * 60)


# ── Load models ───────────────────────────────────────────────
model_files = {
    "rf_crop_model.pkl" : "Random Forest Classifier",
    "xgb_crop_model.pkl": "XGBoost Classifier",
    "label_encoder.pkl" : "Label Encoder",
    "scaler.pkl"        : "Feature Scaler",
    "yield_model.pkl"   : "Yield Regressor (XGBoost)",
    "yield_scaler.pkl"  : "Yield Scaler",
}

loaded = {}
print("\n[1] Loading models:")
for fname, label in model_files.items():
    fpath = f"models/{fname}"
    if os.path.exists(fpath):
        loaded[fname] = joblib.load(fpath)
        size_kb = os.path.getsize(fpath) // 1024
        print(f"  OK  {label:30s} ({size_kb} KB)")
    else:
        print(f"  MISSING  {label} — run notebooks/05_crop_model.py first")

if "rf_crop_model.pkl" not in loaded:
    print("\nERROR: Core models not found. Run Day 8 and Day 9 scripts first.")
    exit(1)

rf  = loaded.get("rf_crop_model.pkl")
xgb = loaded.get("xgb_crop_model.pkl")
le  = loaded.get("label_encoder.pkl")
sc  = loaded.get("scaler.pkl")


# ── Load test data ────────────────────────────────────────────
REC_PATHS = [
    "data/raw/crop_recommendation.csv",
    "data/raw/Crop_recommendation.csv",
]
df = None
for p in REC_PATHS:
    if os.path.exists(p):
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        if "label" in df.columns:
            df = df.rename(columns={"label": "crop"})
        if "Label" in df.columns:
            df = df.rename(columns={"Label": "crop"})
        df["crop"] = df["crop"].str.strip().str.lower()
        break

if df is None:
    print("ERROR: crop_recommendation.csv not found.")
    exit(1)

from sklearn.model_selection import train_test_split
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[FEATURE_COLS].values
y = le.transform(df["crop"])

_, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_te_sc = sc.transform(X_te)


# ── Model comparison ──────────────────────────────────────────
print("\n[2] Model performance comparison:")
print(f"\n  {'Metric':<15} {'Random Forest':>15} {'XGBoost':>15}")
print(f"  {'-'*15} {'-'*15} {'-'*15}")

from sklearn.metrics import f1_score, precision_score, recall_score

y_rf  = rf.predict(X_te)
y_xgb = xgb.predict(X_te_sc)

metrics = {
    "Accuracy" : (accuracy_score(y_te, y_rf),        accuracy_score(y_te, y_xgb)),
    "F1 (weighted)": (f1_score(y_te, y_rf, average="weighted"),
                      f1_score(y_te, y_xgb, average="weighted")),
    "Precision": (precision_score(y_te, y_rf, average="weighted", zero_division=0),
                  precision_score(y_te, y_xgb, average="weighted", zero_division=0)),
    "Recall"   : (recall_score(y_te, y_rf, average="weighted", zero_division=0),
                  recall_score(y_te, y_xgb, average="weighted", zero_division=0)),
}

for metric, (rf_val, xgb_val) in metrics.items():
    winner = "<-- best" if rf_val >= xgb_val else ""
    print(f"  {metric:<15} {rf_val:>15.4f} {xgb_val:>15.4f}  {winner}")

best_model = "Random Forest" if metrics["Accuracy"][0] >= metrics["Accuracy"][1] else "XGBoost"
print(f"\n  Best model: {best_model}")
print(f"  Recommendation: Use {best_model} as primary model in the Streamlit app.")


# ── Per-crop performance ──────────────────────────────────────
print("\n[3] Per-crop classification report (Random Forest):")
print(classification_report(y_te, y_rf, target_names=le.classes_))


# ── Feature importance summary ────────────────────────────────
print("[4] Feature importance (Random Forest):")
importances = rf.feature_importances_
for feat, imp in sorted(zip(FEATURE_COLS, importances),
                        key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:12s} {imp:.4f}  {bar}")


# ── Prediction demo ───────────────────────────────────────────
print("\n[5] Live prediction demo — 3 sample inputs:")
samples = [
    ([90, 42, 43, 20.8, 82.0, 6.5, 202.9], "Expected: rice"),
    ([20, 11, 11, 22.5, 82.0, 6.0, 201.0], "Expected: pigeonpea"),
    ([40, 20, 20, 23.0, 60.0, 7.0,  80.0], "Expected: wheat"),
]
for feat, note in samples:
    arr        = np.array([feat])
    pred_enc   = rf.predict(arr)[0]
    pred_crop  = le.inverse_transform([pred_enc])[0]
    proba      = rf.predict_proba(arr)[0]
    confidence = proba.max() * 100
    print(f"  Input: {feat}")
    print(f"  Predicted: {pred_crop.upper():15s}  Confidence: {confidence:.1f}%  ({note})")
    print()


# ── Deliverable checklist ─────────────────────────────────────
print("[6] Week 2 deliverable checklist:")
deliverables = [
    "models/rf_crop_model.pkl",
    "models/xgb_crop_model.pkl",
    "models/label_encoder.pkl",
    "models/scaler.pkl",
    "models/yield_model.pkl",
    "models/yield_scaler.pkl",
    "models/model_comparison.csv",
    "assets/model_comparison_chart.png",
    "assets/feature_importance.png",
    "assets/shap_waterfall.png",
    "assets/shap_beeswarm.png",
    "assets/shap_feature_importance.png",
    "assets/chart1_crop_dist.png",
    "assets/chart2_correlation.png",
    "assets/chart3_npk_boxplots.png",
    "assets/chart4_rainfall_temp.png",
    "assets/chart5_ph_distribution.png",
    "assets/chart6_humidity.png",
    "assets/india_map.html",
]

done  = sum(1 for f in deliverables if os.path.exists(f))
total = len(deliverables)

for f in deliverables:
    status = "OK" if os.path.exists(f) else "MISSING"
    print(f"  [{status}] {f}")

print(f"\n{done}/{total} deliverables complete.")

if done == total:
    print("\nAll Week 2 deliverables present!")
    print("Ready to commit and start Week 3 (CNN + carbon + rotation).")
else:
    missing = total - done
    print(f"\n{missing} file(s) missing.")
    print("Run the corresponding script to generate them.")

print("\nGit commit command:")
print('  git add assets/ models/model_comparison.csv notebooks/')
print('  git commit -m "Week 2 complete: RF+XGBoost, yield model, SHAP"')
print("  git push")
print("\n" + "=" * 60)
