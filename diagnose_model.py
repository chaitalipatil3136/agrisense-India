"""
AgriSense India — Model Diagnosis Script
File: diagnose_model.py (place in project ROOT, same level as app.py)

Run this BEFORE retraining if Crop Advisor predicts the same crop every time.
It tells you exactly what is wrong with your current models — no guessing.

Run: python diagnose_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("AgriSense India — Model Diagnosis")
print("=" * 60)

# ── Step 1: Check which CSV files exist ──────────────────────
print("\n[1] Checking data files...")

paths_to_check = [
    "data/raw/crop_recommendation.csv",
    "data/raw/Crop_recommendation.csv",
    "data/processed/master_dataset.csv",
]

rec_path = None
for p in paths_to_check:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"  FOUND: {p}")
        print(f"         Shape: {df.shape}")
        print(f"         Columns: {list(df.columns)}")
        cols_upper = [c.upper() for c in df.columns]
        if "N" in cols_upper:
            print(f"         ✅ This looks like crop_recommendation.csv (has N,P,K)")
            if rec_path is None:
                rec_path = p
        else:
            print(f"         ❌ This does NOT have N,P,K — wrong file for training")
    else:
        print(f"  missing: {p}")

# ── Step 2: Check model files ─────────────────────────────────
print("\n[2] Checking model files...")

model_files = {
    "models/rf_crop_model.pkl":  "Random Forest",
    "models/label_encoder.pkl":  "Label Encoder",
    "models/scaler.pkl":         "Scaler",
}

rf, le, sc = None, None, None
for path, label in model_files.items():
    if os.path.exists(path):
        size_kb = os.path.getsize(path) // 1024
        print(f"  FOUND: {path} ({size_kb} KB)")
        try:
            obj = joblib.load(path)
            if "rf" in path:     rf = obj
            if "label" in path:  le = obj
            if "scaler" in path: sc = obj
        except Exception as e:
            print(f"         ❌ Cannot load: {e}")
    else:
        print(f"  MISSING: {path}")

# ── Step 3: Inspect label encoder ────────────────────────────
print("\n[3] Label Encoder classes (what crops the model knows)...")
if le is not None:
    classes = list(le.classes_)
    print(f"  Total crops: {len(classes)}")
    print(f"  All classes: {classes}")
    print(f"  Class 0 (default when model is confused): {classes[0]}")
    if classes[0].lower() in ["kidney beans", "kidneybeans"]:
        print(f"  🚨 CLASS 0 IS KIDNEY BEANS — this is why every prediction = kidney beans")
        print(f"     When scaler makes inputs look like outliers, model defaults to class 0")
else:
    print("  ❌ label_encoder.pkl not loaded")

# ── Step 4: Inspect scaler ────────────────────────────────────
print("\n[4] Scaler inspection...")
if sc is not None:
    print(f"  Scaler type: {type(sc).__name__}")
    if hasattr(sc, "mean_"):
        feat_names = ["N","P","K","temperature","humidity","ph","rainfall"]
        print(f"  Feature means (what the scaler was trained on):")
        for name, mean, std in zip(feat_names, sc.mean_, sc.scale_):
            print(f"    {name:12s}: mean={mean:.2f}  std={std:.2f}")

        print(f"\n  Expected values from crop_recommendation.csv:")
        print(f"    N           : mean≈50   std≈36")
        print(f"    P           : mean≈53   std≈33")
        print(f"    K           : mean≈48   std≈50")
        print(f"    temperature : mean≈25   std≈5")
        print(f"    humidity    : mean≈71   std≈22")
        print(f"    ph          : mean≈6.5  std≈0.77")
        print(f"    rainfall    : mean≈104  std≈55")

        n_mean = sc.mean_[0]
        if n_mean > 100:
            print(f"\n  🚨 SCALER WAS TRAINED ON WRONG DATA!")
            print(f"     N mean = {n_mean:.1f} (should be ~50)")
            print(f"     This confirms: scaler was fit on production data, not crop_recommendation.csv")
        elif abs(n_mean - 50) < 15:
            print(f"\n  ✅ Scaler means look correct (N mean = {n_mean:.1f} ≈ 50)")
        else:
            print(f"\n  ⚠️  Scaler N mean = {n_mean:.1f} — unclear, may be wrong data")
    else:
        print(f"  Scaler has no mean_ attribute")
else:
    print("  ❌ scaler.pkl not loaded")

# ── Step 5: Test prediction with known inputs ─────────────────
print("\n[5] Test prediction (input that should give RICE)...")
if rf is not None and le is not None and sc is not None:
    test_rice = np.array([[80, 40, 40, 25, 82, 6.0, 220]])
    test_sc   = sc.transform(test_rice)
    proba     = rf.predict_proba(test_sc)[0]
    top3_idx  = np.argsort(proba)[::-1][:3]

    print(f"  Input: N=80, P=40, K=40, Temp=25, Humid=82, pH=6.0, Rain=220mm")
    print(f"  (This should predict: rice)")
    print(f"  Scaled input: {test_sc[0].round(2)}")
    print(f"  Top 3 predictions:")
    for idx in top3_idx:
        crop = le.inverse_transform([idx])[0]
        conf = proba[idx] * 100
        print(f"    {crop:20s} {conf:.1f}%")

    best = le.inverse_transform([top3_idx[0]])[0]
    if best.lower() in ["kidney beans", "kidneybeans"]:
        print(f"\n  🚨 CONFIRMED: Model always predicts kidney beans")
        print(f"     Root cause: scaler mismatch — retrain needed")
    elif best.lower() == "rice":
        print(f"\n  ✅ Model correctly predicts rice — scaler looks fine")
    else:
        print(f"\n  ⚠️  Predicted {best} instead of rice — may have minor issue")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)

if rec_path:
    print(f"✅ Training CSV found at: {rec_path}")
else:
    print(f"❌ crop_recommendation.csv NOT FOUND in data/raw/")
    print(f"   Download from: kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
    print(f"   Place at: data/raw/crop_recommendation.csv")

if rf and le and sc:
    print(f"✅ All 3 model files found")
else:
    print(f"❌ Some model files missing — run notebooks/05_crop_model_fixed.py")

print(f"\nNext step: python notebooks/05_crop_model_fixed.py")
print("=" * 60)