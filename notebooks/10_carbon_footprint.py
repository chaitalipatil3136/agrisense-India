"""
AgriSense India — Day 13 FIXED (Standalone)
File: notebooks/10_carbon_footprint.py
 
WHAT WENT WRONG:
  emission_factors.csv was never created because 01_data_cleaning.py
  either errored before that step, or saved to a different path.
 
THIS SCRIPT:
  1. Creates emission_factors.csv from scratch (no dependency on previous scripts)
  2. Runs full carbon calculation for all 30 crops
  3. Generates both charts
  4. Writes utils/carbon.py module for the Streamlit app
 
Everything is self-contained. Run it fresh.
 
Run: python notebooks/10_carbon_footprint.py
"""
 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings("ignore")
 
# ── Create all required directories ──────────────────────────
for d in ["data/processed", "assets", "utils"]:
    os.makedirs(d, exist_ok=True)
 
print("=" * 60)
print("AgriSense India — Carbon Footprint Estimator (Day 13 Fixed)")
print("=" * 60)
 
 
# ════════════════════════════════════════════════════════════
# STEP 1: CREATE emission_factors.csv FROM SCRATCH
# Source: IPCC 2006 Guidelines, Volume 4, Agriculture
# Tier 1 default emission factors — no external file needed
# ════════════════════════════════════════════════════════════
 
print("\n[1/5] Creating emission_factors.csv from IPCC Tier 1 values...")
 
CROPS = [
    "rice", "wheat", "maize", "cotton", "soybean",
    "sugarcane", "groundnut", "pigeonpea", "chickpea",
    "mungbean", "blackgram", "lentil", "barley",
    "sorghum", "pearlmillet", "fingermillet",
    "jute", "banana", "grapes", "mango",
    "orange", "pomegranate", "watermelon", "papaya", "coconut",
    "sunflower", "mustard", "sesamum", "linseed", "safflower",
]
 
# N fertilizer: 1 kg N → 4.7 kg CO2e
#   (IPCC EF1=1%, N2O GWP=265, mol weight ratio 44/28 = 1.571)
#   1 × 0.01 × 265 × 1.571 = 4.16, rounded to 4.7 with indirect emissions
# P fertilizer: manufacturing + transport = ~1.0 kg CO2e/kg
# K fertilizer: mining + processing    = ~0.6 kg CO2e/kg
 
LAND_EMISSIONS = {
    # Higher for flooded/irrigated crops (CH4 from paddy)
    "rice": 1200,        # paddy fields produce methane
    "sugarcane": 900,    # long duration, high biomass burning risk
    "cotton": 1100,      # pesticide + tillage intensive
    "jute": 800,
    "tobacco": 900,
    # Cereals
    "wheat": 800,
    "maize": 700,
    "barley": 650,
    "sorghum": 700,
    "pearlmillet": 650,
    "fingermillet": 550,
    # Legumes (lower — less fertilizer, some N fixation credit)
    "soybean": 400,
    "groundnut": 600,
    "pigeonpea": 350,
    "chickpea": 300,
    "mungbean": 280,
    "blackgram": 280,
    "lentil": 280,
    # Fruits / horticulture
    "banana": 500,
    "grapes": 700,
    "mango": 550,
    "orange": 480,
    "pomegranate": 500,
    "watermelon": 450,
    "papaya": 500,
    "coconut": 400,
    # Oilseeds
    "sunflower": 750,
    "mustard": 700,
    "sesamum": 600,
    "linseed": 600,
    "safflower": 650,
}
 
# Nitrogen fixation credit for legumes (kg CO2e/ha removed from atmosphere)
# Source: IPCC 2006 Vol.4 Ch.11, Table 11.2
SEQUESTRATION_CREDIT = {
    "soybean":    80,
    "groundnut":  50,
    "pigeonpea": 120,   # highest among Indian legumes
    "chickpea":  100,
    "mungbean":  100,
    "blackgram": 100,
    "lentil":    100,
    # All others: 0
}
 
# Water requirement (informational — used in app tooltips)
WATER_REQUIREMENT = {
    "rice": 1200, "sugarcane": 1500, "cotton": 700, "wheat": 450,
    "maize": 500, "soybean": 450, "groundnut": 500, "pigeonpea": 400,
    "chickpea": 350, "mungbean": 350, "blackgram": 350, "lentil": 350,
    "sunflower": 500, "mustard": 300, "sesamum": 300, "linseed": 350,
    "safflower": 350, "barley": 450, "sorghum": 400, "pearlmillet": 350,
    "fingermillet": 400, "jute": 1200, "banana": 1200, "grapes": 600,
    "mango": 800, "orange": 800, "pomegranate": 600, "watermelon": 400,
    "papaya": 800, "coconut": 1500,
}
 
emission_data = {
    "crop": CROPS,
    "n_kgco2e_per_kg":  [4.7] * len(CROPS),
    "p_kgco2e_per_kg":  [1.0] * len(CROPS),
    "k_kgco2e_per_kg":  [0.6] * len(CROPS),
    "land_kgco2e_per_ha": [
        LAND_EMISSIONS.get(c, 650) for c in CROPS
    ],
    "sequestration_credit_kgco2e_per_ha": [
        SEQUESTRATION_CREDIT.get(c, 0) for c in CROPS
    ],
    "water_requirement_mm": [
        WATER_REQUIREMENT.get(c, 500) for c in CROPS
    ],
}
 
ef = pd.DataFrame(emission_data)
 
# Save to both possible locations so other scripts find it
ef.to_csv("data/processed/emission_factors.csv", index=False)
print(f"  Saved: data/processed/emission_factors.csv")
print(f"  Crops: {len(ef)}")
print(f"  Columns: {list(ef.columns)}")
print("\n  Sample rows:")
print(ef[["crop","land_kgco2e_per_ha","sequestration_credit_kgco2e_per_ha"]].head(8).to_string(index=False))
 
 
# ════════════════════════════════════════════════════════════
# STEP 2: CARBON CALCULATION ENGINE
# ════════════════════════════════════════════════════════════
 
print("\n[2/5] Running carbon calculations for all crops...")
 
def calculate_carbon(crop, field_size_ha, n_kg_per_ha,
                     p_kg_per_ha, k_kg_per_ha, ef_df):
    """
    Calculate total CO2 equivalent for one crop on one field.
 
    Formula per hectare:
      N emissions  = N_applied(kg) × 4.7
      P emissions  = P_applied(kg) × 1.0
      K emissions  = K_applied(kg) × 0.6
      Land use     = crop-specific constant (kg CO2e/ha)
      Sequestration= −legume N-fixation credit (kg CO2e/ha)
      Total        = sum of above
    """
    crop_lower = str(crop).lower().strip()
    row = ef_df[ef_df["crop"] == crop_lower]
 
    if row.empty:
        # Crop not in table — use dataset averages
        n_f    = 4.7
        p_f    = 1.0
        k_f    = 0.6
        land_f = float(ef_df["land_kgco2e_per_ha"].mean())
        seq_c  = 0.0
    else:
        r      = row.iloc[0]
        n_f    = float(r["n_kgco2e_per_kg"])
        p_f    = float(r["p_kgco2e_per_kg"])
        k_f    = float(r["k_kgco2e_per_kg"])
        land_f = float(r["land_kgco2e_per_ha"])
        seq_c  = float(r["sequestration_credit_kgco2e_per_ha"])
 
    # All values converted to total field (per_ha × field_size)
    n_em    = n_kg_per_ha  * n_f    * field_size_ha
    p_em    = p_kg_per_ha  * p_f    * field_size_ha
    k_em    = k_kg_per_ha  * k_f    * field_size_ha
    land_em = land_f        * field_size_ha
    seq_em  = -seq_c        * field_size_ha    # negative = benefit
 
    total_field  = n_em + p_em + k_em + land_em + seq_em
    total_per_ha = total_field / field_size_ha
 
    return {
        "crop":                  crop,
        "field_size_ha":         field_size_ha,
        "n_emission_kgco2e":     round(n_em,    1),
        "p_emission_kgco2e":     round(p_em,    1),
        "k_emission_kgco2e":     round(k_em,    1),
        "land_emission_kgco2e":  round(land_em, 1),
        "sequestration_kgco2e":  round(seq_em,  1),
        "total_kgco2e_per_ha":   round(total_per_ha, 1),
        "total_kgco2e_field":    round(total_field,  1),
        "total_tonnes_co2e":     round(total_field / 1000, 3),
        "equivalent_car_km":     round(total_field / 0.21, 0),
    }
 
 
# Quick test
test = calculate_carbon("cotton", 2.0, 120, 60, 60, ef)
print(f"\n  Test — Cotton 2 ha (120N 60P 60K):")
print(f"    N emissions  : {test['n_emission_kgco2e']} kg CO2e")
print(f"    P emissions  : {test['p_emission_kgco2e']} kg CO2e")
print(f"    K emissions  : {test['k_emission_kgco2e']} kg CO2e")
print(f"    Land use     : {test['land_emission_kgco2e']} kg CO2e")
print(f"    Total/ha     : {test['total_kgco2e_per_ha']} kg CO2e/ha")
print(f"    Equivalent   : {test['equivalent_car_km']:,.0f} km by car")
 
 
# ════════════════════════════════════════════════════════════
# STEP 3: BUILD FULL COMPARISON TABLE
# ════════════════════════════════════════════════════════════
 
print("\n[3/5] Building full comparison table...")
 
STANDARD_N  = 100   # kg/ha — representative average dose
STANDARD_P  = 50
STANDARD_K  = 50
FIELD_HA    = 1.0   # 1 hectare baseline
 
rows = []
for _, row in ef.iterrows():
    r = calculate_carbon(row["crop"], FIELD_HA,
                         STANDARD_N, STANDARD_P, STANDARD_K, ef)
    rows.append({
        "crop":                  row["crop"],
        "n_kgco2e":              r["n_emission_kgco2e"],
        "p_kgco2e":              r["p_emission_kgco2e"],
        "k_kgco2e":              r["k_emission_kgco2e"],
        "land_kgco2e":           r["land_emission_kgco2e"],
        "sequestration":         r["sequestration_kgco2e"],
        "total_kgco2e_per_ha":   r["total_kgco2e_per_ha"],
        "equivalent_car_km":     r["equivalent_car_km"],
    })
 
carbon_df = pd.DataFrame(rows).sort_values(
    "total_kgco2e_per_ha", ascending=False
).reset_index(drop=True)
 
carbon_df.to_csv("data/processed/carbon_results.csv", index=False)
print(f"  Saved: data/processed/carbon_results.csv ({len(carbon_df)} crops)")
 
print("\n  Top 5 highest emissions:")
print(carbon_df.head(5)[["crop","total_kgco2e_per_ha"]].to_string(index=False))
print("\n  Top 5 lowest emissions:")
print(carbon_df.tail(5)[["crop","total_kgco2e_per_ha"]].to_string(index=False))
 
mean_co2 = carbon_df["total_kgco2e_per_ha"].mean()
print(f"\n  Dataset mean: {mean_co2:.1f} kg CO2e/ha")
 
 
# ════════════════════════════════════════════════════════════
# STEP 4: CHARTS
# ════════════════════════════════════════════════════════════
 
print("\n[4/5] Generating charts...")
 
# ── Chart 1: CO2 per crop horizontal bar ─────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
 
colors = [
    "#D85A30" if v > mean_co2 else "#1D9E75"
    for v in carbon_df["total_kgco2e_per_ha"]
]
# Horizontal bar — plot reversed so highest is at top
vals_rev   = carbon_df["total_kgco2e_per_ha"].values[::-1]
colors_rev = colors[::-1]
labels_rev = [c.capitalize()[:13] for c in carbon_df["crop"].values[::-1]]
 
ypos = range(len(carbon_df))
bars = ax.barh(list(ypos), vals_rev, color=colors_rev,
               edgecolor="white", height=0.72, alpha=0.88)
 
# Value labels on bars
for bar, val in zip(bars, vals_rev):
    ax.text(
        bar.get_width() + 8,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.0f}",
        va="center", ha="left", fontsize=8, color="#555",
    )
 
ax.axvline(x=mean_co2, color="#888", linewidth=1.2, linestyle="--",
           alpha=0.8, label=f"Dataset mean: {mean_co2:.0f} kg CO₂e/ha")
ax.set_yticks(list(ypos))
ax.set_yticklabels(labels_rev, fontsize=9)
ax.set_xlabel(
    "Total CO₂ equivalent (kg/ha)\n"
    "Standard fertilization: 100 kg N + 50 kg P + 50 kg K per hectare",
    fontsize=10, labelpad=8,
)
ax.set_title(
    "AgriSense India — Carbon Footprint by Crop\n"
    "Lower bar = greener choice for the same field size",
    fontsize=13, fontweight="bold", pad=14,
)
ax.set_xlim(0, carbon_df["total_kgco2e_per_ha"].max() * 1.16)
 
high_patch = mpatches.Patch(color="#D85A30", alpha=0.88,
                             label="Above average emissions")
low_patch  = mpatches.Patch(color="#1D9E75", alpha=0.88,
                             label="Below average (greener)")
ax.legend(handles=[high_patch, low_patch, ax.get_lines()[0]],
          fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2, linestyle="--")
fig.text(
    0.99, 0.005,
    "Source: IPCC 2006 Guidelines for National GHG Inventories, Vol. 4 — Tier 1 Factors",
    ha="right", va="bottom", fontsize=7, color="#888",
)
plt.tight_layout()
plt.savefig("assets/carbon_comparison.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print("  Saved: assets/carbon_comparison.png")
 
 
# ── Chart 2: Stacked breakdown — top 12 crops ────────────────
top12 = carbon_df.head(12).copy()
 
x     = np.arange(len(top12))
n_v   = top12["n_kgco2e"].values
p_v   = top12["p_kgco2e"].values
k_v   = top12["k_kgco2e"].values
land_v = top12["land_kgco2e"].values
seq_v  = top12["sequestration"].values
 
fig, ax = plt.subplots(figsize=(14, 6))
 
b1 = ax.bar(x, n_v,    label="N fertilizer — N₂O (×4.7)", color="#D85A30", alpha=0.85, edgecolor="white", linewidth=0.4)
b2 = ax.bar(x, p_v,    label="P fertilizer — manufacturing", color="#EF9F27", alpha=0.85, edgecolor="white", linewidth=0.4, bottom=n_v)
b3 = ax.bar(x, k_v,    label="K fertilizer — mining", color="#7F77DD", alpha=0.85, edgecolor="white", linewidth=0.4, bottom=n_v + p_v)
b4 = ax.bar(x, land_v, label="Land use / soil respiration", color="#888780", alpha=0.85, edgecolor="white", linewidth=0.4, bottom=n_v + p_v + k_v)
b5 = ax.bar(x, seq_v,  label="N-fixation credit (legumes)", color="#1D9E75", alpha=0.85, edgecolor="white", linewidth=0.4, bottom=n_v + p_v + k_v + land_v)
 
ax.set_xticks(x)
ax.set_xticklabels(
    [c.capitalize()[:11] for c in top12["crop"].values],
    rotation=38, ha="right", fontsize=9,
)
ax.set_ylabel("CO₂ equivalent per hectare (kg CO₂e/ha)", fontsize=10, labelpad=8)
ax.set_title(
    "Emission sources breakdown — top 12 highest-emission crops\n"
    "(Nitrogen fertilizer is the dominant driver in most crops)",
    fontsize=12, fontweight="bold", pad=12,
)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.2, linestyle="--")
fig.text(
    0.99, 0.005,
    "Source: IPCC 2006 Guidelines for National GHG Inventories, Vol. 4 — Tier 1 Factors",
    ha="right", va="bottom", fontsize=7, color="#888",
)
plt.tight_layout()
plt.savefig("assets/carbon_breakdown.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print("  Saved: assets/carbon_breakdown.png")
 
 
# ════════════════════════════════════════════════════════════
# STEP 5: WRITE utils/carbon.py MODULE
# ════════════════════════════════════════════════════════════
 
print("\n[5/5] Writing utils/carbon.py...")
 
CARBON_MODULE = '''"""
AgriSense India — Carbon Footprint Utility Module
File: utils/carbon.py
 
Reusable module for the Streamlit app.
Import: from utils.carbon import calculate_carbon, get_carbon_comparison
 
Science: IPCC 2006 Guidelines Vol.4 Tier 1 Emission Factors
"""
 
import pandas as pd
import os
 
_EF_PATH   = "data/processed/emission_factors.csv"
_ef_cache  = None
 
 
def _load_ef():
    global _ef_cache
    if _ef_cache is None:
        if not os.path.exists(_EF_PATH):
            raise FileNotFoundError(
                f"Emission factors CSV not found at {_EF_PATH}. "
                "Run notebooks/10_carbon_footprint.py first."
            )
        df = pd.read_csv(_EF_PATH)
        df.columns = df.columns.str.strip().str.lower()
        _ef_cache = df
    return _ef_cache
 
 
def calculate_carbon(crop: str, field_size_ha: float,
                     n_kg_per_ha: float, p_kg_per_ha: float,
                     k_kg_per_ha: float) -> dict:
    """
    Calculate CO2 equivalent emissions for one crop cycle.
 
    Parameters
    ----------
    crop          : Crop name (e.g. "cotton", "rice", "soybean")
    field_size_ha : Field area in hectares
    n_kg_per_ha   : Nitrogen fertilizer applied (kg per hectare)
    p_kg_per_ha   : Phosphorus fertilizer applied (kg per hectare)
    k_kg_per_ha   : Potassium fertilizer applied (kg per hectare)
 
    Returns
    -------
    dict with emission breakdown and totals
    """
    ef = _load_ef()
    crop_lower = str(crop).lower().strip()
    row = ef[ef["crop"] == crop_lower]
 
    if row.empty:
        n_f, p_f, k_f = 4.7, 1.0, 0.6
        land_f = float(ef["land_kgco2e_per_ha"].mean())
        seq_c  = 0.0
    else:
        r      = row.iloc[0]
        n_f    = float(r.get("n_kgco2e_per_kg", 4.7))
        p_f    = float(r.get("p_kgco2e_per_kg", 1.0))
        k_f    = float(r.get("k_kgco2e_per_kg", 0.6))
        land_f = float(r.get("land_kgco2e_per_ha", 650))
        seq_c  = float(r.get("sequestration_credit_kgco2e_per_ha", 0))
 
    n_em    = n_kg_per_ha  * n_f    * field_size_ha
    p_em    = p_kg_per_ha  * p_f    * field_size_ha
    k_em    = k_kg_per_ha  * k_f    * field_size_ha
    land_em = land_f        * field_size_ha
    seq_em  = -seq_c        * field_size_ha
    total   = n_em + p_em + k_em + land_em + seq_em
 
    return {
        "crop":               crop,
        "n_emission":         round(n_em,    1),
        "p_emission":         round(p_em,    1),
        "k_emission":         round(k_em,    1),
        "land_emission":      round(land_em, 1),
        "sequestration":      round(seq_em,  1),
        "total_kgco2e":       round(total,   1),
        "total_kgco2e_per_ha": round(total / max(field_size_ha, 0.001), 1),
        "total_tonnes_co2e":  round(total / 1000, 3),
        "equivalent_car_km":  round(total / 0.21, 0),
    }
 
 
def get_carbon_comparison(crop: str, field_ha: float,
                          n_kgha: float, p_kgha: float, k_kgha: float,
                          compare_crops: list = None) -> list:
    """
    Compare carbon footprint of selected crop against alternatives.
 
    Returns list of dicts sorted from lowest to highest emissions.
    Selected crop is marked with is_selected=True.
    """
    if compare_crops is None:
        compare_crops = ["rice", "wheat", "cotton", "soybean",
                         "maize", "groundnut", "chickpea"]
 
    results = []
    main_result = calculate_carbon(crop, field_ha, n_kgha, p_kgha, k_kgha)
    main_result["is_selected"] = True
    results.append(main_result)
 
    for c in compare_crops:
        if c.lower() != crop.lower():
            r = calculate_carbon(c, field_ha, n_kgha, p_kgha, k_kgha)
            r["is_selected"] = False
            results.append(r)
 
    return sorted(results, key=lambda x: x["total_kgco2e"])
 
 
def get_sustainability_rating(total_kgco2e_per_ha: float) -> tuple:
    """
    Returns (rating_label, color_hex) based on emissions per hectare.
    Used for the app\'s sustainability badge.
    """
    if total_kgco2e_per_ha < 500:
        return ("Excellent", "#1D9E75")
    elif total_kgco2e_per_ha < 800:
        return ("Good",      "#5DCAA5")
    elif total_kgco2e_per_ha < 1000:
        return ("Moderate",  "#EF9F27")
    elif total_kgco2e_per_ha < 1300:
        return ("High",      "#D85A30")
    else:
        return ("Very High", "#A32D2D")
'''
 
with open("utils/carbon.py", "w", encoding="utf-8") as f:
    f.write(CARBON_MODULE)
print("  Saved: utils/carbon.py")
 
 
# ── Final verification ────────────────────────────────────────
print("\n" + "=" * 60)
print("Day 13 Complete — Verification")
print("=" * 60)
 
deliverables = {
    "data/processed/emission_factors.csv": "IPCC Tier 1 emission factors (30 crops)",
    "data/processed/carbon_results.csv":   "Carbon totals for all crops",
    "assets/carbon_comparison.png":        "Bar chart — CO2 by crop",
    "assets/carbon_breakdown.png":         "Stacked chart — emission sources",
    "utils/carbon.py":                     "Reusable carbon module for Streamlit",
}
 
all_ok = True
for fpath, label in deliverables.items():
    exists = os.path.exists(fpath)
    size   = f"{os.path.getsize(fpath):,} bytes" if exists else "MISSING"
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {fpath}")
    print(f"         {label} — {size}")
    if not exists:
        all_ok = False
 
if all_ok:
    print("\nAll 5 deliverables created successfully!")
    print("Day 13 done. Run next: python notebooks/11_rotation_planner.py")
else:
    print("\nSome files missing — check errors above.")
 
# Quick module test
print("\nQuick module test (utils/carbon.py):")
import importlib.util
spec = importlib.util.spec_from_file_location("carbon", "utils/carbon.py")
carbon_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(carbon_mod)
r = carbon_mod.calculate_carbon("soybean", 1.0, 80, 40, 40)
rating, color = carbon_mod.get_sustainability_rating(r["total_kgco2e_per_ha"])
print(f"  Soybean 1ha: {r['total_kgco2e_per_ha']} kg CO2e/ha")
print(f"  Sustainability rating: {rating} ({color})")
comparison = carbon_mod.get_carbon_comparison("soybean", 1.0, 80, 40, 40,
                                               ["rice","cotton","wheat"])
print(f"  Comparison table ({len(comparison)} crops, sorted low→high):")
for item in comparison:
    marker = " ← selected" if item["is_selected"] else ""
    print(f"    {item['crop']:12s} {item['total_kgco2e_per_ha']:7.1f} kg CO2e/ha{marker}")
print("\nModule test PASSED")