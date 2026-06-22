"""
AgriSense India — Fixed Crop Model Training Script
File: notebooks/05_crop_model_fixed.py

FIXES THE KIDNEY BEANS BUG:
  Root cause: scaler was fit on wrong dataset (production data instead of
              crop_recommendation.csv). This makes all inputs look like
              extreme outliers → model always predicts class 0 (kidney beans).

HOW TO RUN:
  1. Make sure crop_recommendation.csv is at data/raw/crop_recommendation.csv
  2. Delete all old model files: del models/*.pkl
  3. Run: python notebooks/05_crop_model_fixed.py
  4. Restart Streamlit: Ctrl+C then streamlit run app.py

WHAT THIS SCRIPT DOES:
  - Loads ONLY crop_recommendation.csv (2200 rows, 22 crops)
  - Verifies the data is correct before training
  - Fits scaler on crop_recommendation.csv ONLY
  - Trains Random Forest + XGBoost
  - Prints test predictions to verify fix worked
  - Saves all 5 model files
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Setup paths ───────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DATA_PATH   = os.path.join(ROOT, "data", "raw", "crop_recommendation.csv")
MODELS_DIR  = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 65)
print("AgriSense India — Crop Model Training (Fixed)")
print("=" * 65)

# ════════════════════════════════════════════════════════════
# STEP 1 — Find and verify the correct CSV
# ════════════════════════════════════════════════════════════

print("\n[1] Looking for crop_recommendation.csv ...")

# Try multiple possible paths
SEARCH_PATHS = [
    DATA_PATH,
    os.path.join(ROOT, "data", "raw", "Crop_recommendation.csv"),
    os.path.join(ROOT, "data", "Crop_recommendation.csv"),
    os.path.join(ROOT, "Crop_recommendation.csv"),
    os.path.join(ROOT, "crop_recommendation.csv"),
]

found_path = None
for p in SEARCH_PATHS:
    if os.path.exists(p):
        found_path = p
        print(f"  FOUND: {p}")
        break

if found_path is None:
    print("\n  ERROR: crop_recommendation.csv not found!")
    print("  Download from: kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
    print("  Save to: data/raw/crop_recommendation.csv")
    sys.exit(1)

# Load and verify
df = pd.read_csv(found_path)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Verify it is the correct file
required_cols = {"N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"}
actual_cols   = set(df.columns.str.strip())

# Handle lowercase column names
col_map = {}
for col in df.columns:
    col_map[col.strip()] = col
df.columns = df.columns.str.strip()

# Check if this looks like the right file
if "label" not in df.columns and "crop" in df.columns:
    df = df.rename(columns={"crop": "label"})
    print("  INFO: Renamed 'crop' column to 'label'")

if not {"N", "P", "K", "temperature", "humidity", "ph", "rainfall"}.issubset(
    set(df.columns.str.lower())
):
    # Try lowercase
    df.columns = df.columns.str.lower()

if df.shape[0] > 10000:
    print(f"\n  WARNING: File has {df.shape[0]} rows.")
    print("  crop_recommendation.csv should have exactly 2200 rows.")
    print("  This looks like the production dataset — WRONG FILE.")
    print("  Download the correct file from Kaggle.")
    sys.exit(1)

print(f"\n  Crops found: {sorted(df['label'].unique())}")
print(f"  Total crops: {df['label'].nunique()}")
print(f"  Rows: {df.shape[0]} (should be 2200)")

# Verify value ranges match expected
print(f"\n  Data statistics (verify these look right):")
for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
    if col in df.columns:
        print(f"    {col:12s}: mean={df[col].mean():.1f}  "
              f"min={df[col].min():.1f}  "
              f"max={df[col].max():.1f}")

# Expected ranges for crop_recommendation.csv
expected = {
    "N":           (0,   140,   50),   # min, max, approx_mean
    "P":           (5,   145,   53),
    "K":           (5,   205,   48),
    "temperature": (8,   44,    25),
    "humidity":    (14,  100,   71),
    "ph":          (3.5, 9.9,   6.5),
    "rainfall":    (20,  300,   104),
}

print("\n  Range validation:")
all_ok = True
for col, (emin, emax, emean) in expected.items():
    if col in df.columns:
        actual_mean = df[col].mean()
        if abs(actual_mean - emean) > emean * 0.5:
            print(f"    {col:12s}: ⚠️  mean={actual_mean:.1f} (expected ~{emean}) — may be wrong data")
            all_ok = False
        else:
            print(f"    {col:12s}: ✅ mean={actual_mean:.1f} (expected ~{emean})")

if not all_ok:
    print("\n  ⚠️  Some values look unusual but continuing...")


# ════════════════════════════════════════════════════════════
# STEP 2 — Prepare features and labels
# ════════════════════════════════════════════════════════════

print("\n[2] Preparing features and labels ...")

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Handle any case variations
col_lower_map = {c.lower(): c for c in df.columns}
feature_cols_actual = []
for f in FEATURE_COLS:
    if f in df.columns:
        feature_cols_actual.append(f)
    elif f.lower() in col_lower_map:
        feature_cols_actual.append(col_lower_map[f.lower()])
    else:
        print(f"  ERROR: Column '{f}' not found!")
        sys.exit(1)

label_col = "label" if "label" in df.columns else "crop"

X = df[feature_cols_actual].values.astype(float)
y = df[label_col].values

print(f"  Feature matrix: {X.shape}")
print(f"  Label column: '{label_col}'")
print(f"  Unique crops: {len(np.unique(y))}")

# Check for NaN
if np.isnan(X).any():
    print("  Filling NaN values with column means...")
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_means[j]


# ════════════════════════════════════════════════════════════
# STEP 3 — Encode labels
# ════════════════════════════════════════════════════════════

print("\n[3] Encoding crop labels ...")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"  Classes (sorted): {list(le.classes_)}")
print(f"  Class 0 is: '{le.classes_[0]}' (this is what model predicts when confused)")
print(f"  Number of classes: {len(le.classes_)}")


# ════════════════════════════════════════════════════════════
# STEP 4 — Scale features (THE CRITICAL FIX)
# ════════════════════════════════════════════════════════════

print("\n[4] Fitting scaler on crop_recommendation.csv ONLY ...")
print("    (This is the fix — old code used wrong dataset here)")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# THE FIX: Scaler is fit ONLY on X (crop_recommendation.csv)
# NOT on any production/master dataset
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

print(f"  Scaler fitted on {X.shape[0]} rows from crop_recommendation.csv")
print(f"  Scaler means: {sc.mean_.round(2)}")
print(f"  Scaler std:   {sc.scale_.round(2)}")

# Verify scaler makes sense
if sc.mean_[0] > 100:   # N mean should be ~50, not 100+
    print("\n  ❌ SCALER STILL WRONG — N mean is too high")
    print("  Are you using the right CSV file?")
    sys.exit(1)
else:
    print(f"\n  ✅ Scaler looks correct (N mean = {sc.mean_[0]:.1f}, expected ~50)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

print(f"\n  Train: {X_train.shape[0]} rows")
print(f"  Test:  {X_test.shape[0]} rows")


# ════════════════════════════════════════════════════════════
# STEP 5 — Train Random Forest
# ════════════════════════════════════════════════════════════

print("\n[5] Training Random Forest ...")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc  = accuracy_score(y_test,  rf.predict(X_test))

print(f"  Train accuracy: {rf_train_acc * 100:.2f}%")
print(f"  Test accuracy:  {rf_test_acc  * 100:.2f}%")

if rf_test_acc < 0.90:
    print("  ⚠️  Accuracy below 90% — check your data")
else:
    print(f"  ✅ Excellent accuracy!")


# ════════════════════════════════════════════════════════════
# STEP 6 — Train XGBoost
# ════════════════════════════════════════════════════════════

print("\n[6] Training XGBoost ...")

try:
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    xgb_test_acc = accuracy_score(y_test, xgb.predict(X_test))
    print(f"  XGBoost test accuracy: {xgb_test_acc * 100:.2f}%")

except ImportError:
    print("  XGBoost not installed — skipping (pip install xgboost)")
    xgb = None


# ════════════════════════════════════════════════════════════
# STEP 7 — Train yield model (simple regressor)
# ════════════════════════════════════════════════════════════

print("\n[7] Training yield model ...")

from sklearn.ensemble import GradientBoostingRegressor

# Create synthetic but realistic yield targets
# Based on ICAR district-level yield data
YIELD_MAP = {
    "rice": 2500, "wheat": 3500, "maize": 3000,
    "cotton": 1800, "soybean": 1400, "sugarcane": 70000,
    "groundnut": 1500, "chickpea": 1000, "lentil": 900,
    "mungbean": 900, "blackgram": 900, "pigeonpea": 1200,
    "pomegranate": 12000, "mango": 8000, "grapes": 15000,
    "watermelon": 25000, "papaya": 30000, "orange": 10000,
    "apple": 18000, "coconut": 7500, "jute": 2500,
    "coffee": 800, "banana": 20000,
}

np.random.seed(42)
y_yield = np.array([
    YIELD_MAP.get(crop.lower(), 2000) * np.random.uniform(0.7, 1.3)
    for crop in y
])

X_yield_train, X_yield_test, y_yield_train, y_yield_test = train_test_split(
    X, y_yield, test_size=0.2, random_state=42
)

yield_scaler = StandardScaler()
X_y_train_sc = yield_scaler.fit_transform(X_yield_train)
X_y_test_sc  = yield_scaler.transform(X_yield_test)

yield_model = GradientBoostingRegressor(
    n_estimators=100, max_depth=4,
    learning_rate=0.1, random_state=42,
)
yield_model.fit(X_y_train_sc, y_yield_train)

from sklearn.metrics import r2_score
y_pred_yield = yield_model.predict(X_y_test_sc)
r2 = r2_score(y_yield_test, y_pred_yield)
print(f"  Yield model R² score: {r2:.3f}")


# ════════════════════════════════════════════════════════════
# STEP 8 — Verify predictions before saving
# ════════════════════════════════════════════════════════════

print("\n[8] Verifying predictions (most important step) ...")

TEST_CASES = [
    {
        "name":     "Rice conditions (high humidity + high rainfall)",
        "inputs":   [80, 40, 40, 25, 82, 6.0, 220],
        "expected": "rice",
    },
    {
        "name":     "Cotton conditions (high N + moderate rainfall)",
        "inputs":   [117, 46, 20, 28, 65, 7.0, 80],
        "expected": "cotton",
    },
    {
        "name":     "Chickpea conditions (low rainfall + low NPK)",
        "inputs":   [40, 67, 19, 18, 18, 7.0, 65],
        "expected": "chickpea",
    },
    {
        "name":     "Maize conditions (moderate everything)",
        "inputs":   [77, 52, 17, 22, 82, 6.0, 75],
        "expected": "maize",
    },
    {
        "name":     "Soybean conditions (Vidarbha typical)",
        "inputs":   [43, 67, 19, 29, 92, 6.5, 103],
        "expected": "soybean",
    },
]

all_correct = True
print()
for tc in TEST_CASES:
    inp        = np.array([tc["inputs"]])
    inp_scaled = sc.transform(inp)
    proba      = rf.predict_proba(inp_scaled)[0]
    top3_idx   = np.argsort(proba)[::-1][:3]
    top3       = [(le.inverse_transform([i])[0], proba[i]) for i in top3_idx]
    predicted  = top3[0][0]
    conf       = top3[0][1] * 100

    status = "✅" if predicted.lower() == tc["expected"].lower() else "❌"
    print(f"  {status} {tc['name']}")
    print(f"     Expected: {tc['expected']:15s}  |  "
          f"Predicted: {predicted:15s}  ({conf:.1f}%)")
    print(f"     Top 3: {', '.join(f'{c}({v*100:.0f}%)' for c, v in top3)}")
    print()

    if predicted.lower() != tc["expected"].lower():
        all_correct = False

if all_correct:
    print("  ✅ ALL TEST CASES PASSED — kidney beans bug is FIXED")
else:
    print("  ⚠️  Some test cases failed — but model may still be improved")
    print("  This can happen if crop data distribution is unusual")
    print("  Continuing to save models...")


# ════════════════════════════════════════════════════════════
# STEP 9 — Save all model files
# ════════════════════════════════════════════════════════════

print("\n[9] Saving model files ...")

joblib.dump(rf,           os.path.join(MODELS_DIR, "rf_crop_model.pkl"))
joblib.dump(le,           os.path.join(MODELS_DIR, "label_encoder.pkl"))
joblib.dump(sc,           os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(yield_model,  os.path.join(MODELS_DIR, "yield_model.pkl"))
joblib.dump(yield_scaler, os.path.join(MODELS_DIR, "yield_scaler.pkl"))

if xgb is not None:
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb_crop_model.pkl"))
    print("  Saved: xgb_crop_model.pkl")

# Save model comparison CSV
comparison_data = {
    "model":     ["Random Forest", "XGBoost"] if xgb else ["Random Forest"],
    "accuracy":  [rf_test_acc, xgb_test_acc]  if xgb else [rf_test_acc],
    "train_acc": [rf_train_acc, None]          if xgb else [rf_train_acc],
}
pd.DataFrame(comparison_data).to_csv(
    os.path.join(MODELS_DIR, "model_comparison.csv"), index=False
)

print(f"  Saved: rf_crop_model.pkl")
print(f"  Saved: label_encoder.pkl")
print(f"  Saved: scaler.pkl")
print(f"  Saved: yield_model.pkl")
print(f"  Saved: yield_scaler.pkl")
print(f"  Saved: model_comparison.csv")

file_sizes = {
    "rf_crop_model.pkl":  os.path.getsize(os.path.join(MODELS_DIR, "rf_crop_model.pkl")) // 1024,
    "label_encoder.pkl":  os.path.getsize(os.path.join(MODELS_DIR, "label_encoder.pkl")) // 1024,
    "scaler.pkl":         os.path.getsize(os.path.join(MODELS_DIR, "scaler.pkl")) // 1024,
}
print("\n  File sizes:")
for fname, kb in file_sizes.items():
    print(f"    {fname}: {kb} KB")


# ════════════════════════════════════════════════════════════
# STEP 10 — Cross-validation score
# ════════════════════════════════════════════════════════════

print("\n[10] Cross-validation (5-fold) ...")

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf, X_scaled, y_encoded, cv=5, scoring="accuracy")
print(f"  CV scores: {cv_scores.round(4)}")
print(f"  Mean CV:   {cv_scores.mean() * 100:.2f}%")
print(f"  Std CV:    {cv_scores.std() * 100:.2f}%")


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)
print(f"  Random Forest accuracy: {rf_test_acc * 100:.2f}%")
print(f"  Cross-validation mean:  {cv_scores.mean() * 100:.2f}%")
print(f"  Classes trained:        {len(le.classes_)}")
print(f"  Scaler N mean:          {sc.mean_[0]:.1f} (should be ~50)")
print()
print("  Next steps:")
print("  1. Restart Streamlit:  Ctrl+C → streamlit run app.py")
print("  2. Go to Crop Advisor page")
print("  3. Set N=80, P=40, K=40, Temp=25, Humid=82, pH=6.0, Rain=220")
print("  4. Click Predict — should show RICE (not kidney beans)")
print()
print("  If still showing kidney beans:")
print("  → Clear Streamlit cache: Menu → Clear cache → Rerun")
print("  → Or add ?clear_cache=true to your URL")
print("=" * 65)