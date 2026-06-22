"""
AgriSense India — EDA Visualizations (FIXED VERSION)
File: notebooks/02_eda.py

ROOT CAUSE OF PREVIOUS ERROR:
  master_dataset.csv was the production dataset (246k rows, 7 cols).
  Charts 3-6 need crop_recommendation.csv (2200 rows, 8 cols: N,P,K,etc).

This fixed version loads BOTH datasets explicitly:
  - rec_df  : crop_recommendation.csv  → Charts 2,3,4,5,6
  - prod_df : india_production_clean.csv → Chart 1 (124 crops)

Run: python notebooks/02_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("assets", exist_ok=True)

# ── Global style ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

C_GREEN  = "#1D9E75"
C_TEAL   = "#5DCAA5"
C_AMBER  = "#EF9F27"
C_CORAL  = "#D85A30"
C_PURPLE = "#7F77DD"

CROP_COLORS = [
    "#1D9E75","#0F6E56","#5DCAA5","#9FE1CB",
    "#EF9F27","#BA7517","#D85A30","#F0997B",
    "#7F77DD","#AFA9EC","#378ADD","#85B7EB",
    "#639922","#97C459","#E24B4A","#F09595",
    "#D4537E","#888780","#5F5E5A","#2C2C2A",
    "#FAC775","#B5D4F4",
]

print("=" * 60)
print("AgriSense India — EDA Visualizations (Fixed)")
print("=" * 60)


# ════════════════════════════════════════════════════════════
# LOAD DATASETS — two separate files, explicit paths
# ════════════════════════════════════════════════════════════

# ── Dataset 1: crop_recommendation.csv ───────────────────────
# Has: N, P, K, temperature, humidity, ph, rainfall, crop
# Used for: Charts 2, 3, 4, 5, 6
REC_PATHS = [
    "data/raw/crop_recommendation.csv",
    "data/raw/Crop_recommendation.csv",
    "data/raw/crop_rec.csv",
]
rec_df = None
for p in REC_PATHS:
    if os.path.exists(p):
        rec_df = pd.read_csv(p)
        # Standardize column names
        rec_df.columns = rec_df.columns.str.strip()
        # Rename label → crop if needed
        if "label" in rec_df.columns:
            rec_df = rec_df.rename(columns={"label": "crop"})
        if "Label" in rec_df.columns:
            rec_df = rec_df.rename(columns={"Label": "crop"})
        # Lowercase crop values
        rec_df["crop"] = rec_df["crop"].str.strip().str.lower()
        print(f"\nLoaded rec_df from: {p}")
        print(f"  Shape  : {rec_df.shape}")
        print(f"  Columns: {list(rec_df.columns)}")
        print(f"  Crops  : {sorted(rec_df['crop'].unique())}")
        break

if rec_df is None:
    print("\nERROR: crop_recommendation.csv not found in data/raw/")
    print("Download from: kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
    print("Place at: data/raw/crop_recommendation.csv")
    exit(1)

# Verify required columns exist
REQUIRED_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "crop"]
missing = [c for c in REQUIRED_COLS if c not in rec_df.columns]
if missing:
    print(f"\nERROR: Missing columns in crop_recommendation.csv: {missing}")
    print(f"Available columns: {list(rec_df.columns)}")
    print("Check that you downloaded the correct dataset (Atharva Ingle).")
    exit(1)

print(f"\n  All required columns present: {REQUIRED_COLS}")

# ── Dataset 2: india_production_clean.csv ────────────────────
# Has: state, district, year, season, crop, area, production
# Used for: Chart 1 (crop frequency from real India data)
PROD_PATHS = [
    "data/processed/india_production_clean.csv",
    "data/processed/master_dataset.csv",     # fallback in case it was saved here
]
prod_df = None
for p in PROD_PATHS:
    if os.path.exists(p):
        tmp = pd.read_csv(p)
        # Check it's actually production data (has 'crop' but NOT 'N')
        if "crop" in tmp.columns and "N" not in tmp.columns:
            prod_df = tmp
            print(f"\nLoaded prod_df from: {p}")
            print(f"  Shape  : {prod_df.shape}")
            print(f"  Columns: {list(prod_df.columns)}")
            break

if prod_df is None:
    print("\nWARNING: india_production_clean.csv not found.")
    print("Chart 1 will use crop_recommendation.csv data instead.")
    prod_df = rec_df.copy()
    use_prod_for_chart1 = False
else:
    use_prod_for_chart1 = True

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# ════════════════════════════════════════════════════════════
# CHART 1 — Crop frequency
# Source: production data (124 real Indian crops) if available,
#         else crop_recommendation.csv (22 crops)
# ════════════════════════════════════════════════════════════
print("\n[Chart 1] Crop frequency distribution...")

if use_prod_for_chart1:
    # Use production data — top 25 crops by record count
    crop_col = "crop" if "crop" in prod_df.columns else prod_df.columns[4]
    crop_counts = prod_df[crop_col].value_counts().head(25).sort_values(ascending=True)
    chart1_source = "India Crop Production Statistics — data.gov.in / Govt of India"
    chart1_title  = "Top 25 crops by record count — India production dataset (1997–2015)"
else:
    crop_counts = rec_df["crop"].value_counts().sort_values(ascending=True)
    chart1_source = "Crop Recommendation Dataset — Atharva Ingle (Kaggle)"
    chart1_title  = "Crop frequency — training dataset (22 crops)"

colors_bar = [CROP_COLORS[i % len(CROP_COLORS)] for i in range(len(crop_counts))]

fig, ax = plt.subplots(figsize=(12, 9))
bars = ax.barh(
    crop_counts.index,
    crop_counts.values,
    color=colors_bar,
    edgecolor="white",
    linewidth=0.5,
    height=0.72,
)
for bar, val in zip(bars, crop_counts.values):
    ax.text(
        bar.get_width() + crop_counts.max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,}", va="center", ha="left", fontsize=8, color="#555",
    )

ax.set_xlabel("Number of records", fontsize=11, labelpad=8)
ax.set_title(chart1_title, fontsize=13, fontweight="bold", pad=14)
ax.set_xlim(0, crop_counts.max() * 1.15)
ax.tick_params(axis="y", labelsize=9)
fig.text(0.99, 0.01, f"Source: {chart1_source}",
         ha="right", va="bottom", fontsize=7, color="#888")
plt.tight_layout()
plt.savefig("assets/chart1_crop_dist.png")
plt.close()
print("  Saved: assets/chart1_crop_dist.png")


# ════════════════════════════════════════════════════════════
# CHART 2 — Feature correlation heatmap
# Source: crop_recommendation.csv (rec_df) — needs N,P,K,etc
# ════════════════════════════════════════════════════════════
print("[Chart 2] Feature correlation heatmap...")

corr = rec_df[FEATURE_COLS].corr().round(2)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(
    corr, ax=ax,
    annot=True, fmt=".2f",
    cmap="RdYlGn",
    vmin=-1, vmax=1, center=0,
    linewidths=0.6, linecolor="white",
    annot_kws={"size": 10},
    square=True,
    cbar_kws={"shrink": 0.78, "label": "Pearson r"},
)
ax.set_title("Feature correlation — soil & climate variables",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
fig.text(0.99, 0.01, "Source: Crop Recommendation Dataset — Atharva Ingle (Kaggle)",
         ha="right", va="bottom", fontsize=7, color="#888")
plt.tight_layout()
plt.savefig("assets/chart2_correlation.png")
plt.close()
print("  Saved: assets/chart2_correlation.png")


# ════════════════════════════════════════════════════════════
# CHART 3 — NPK boxplots by crop
# Source: crop_recommendation.csv (rec_df)
# ════════════════════════════════════════════════════════════
print("[Chart 3] NPK boxplots by crop...")

# Top 6 crops — keeps labels readable
top6 = rec_df["crop"].value_counts().head(6).index.tolist()
df_top = rec_df[rec_df["crop"].isin(top6)].copy()
short_names = [c.capitalize()[:11] for c in top6]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
nutrients  = ["N",            "P",           "K"]
ylabels    = ["Nitrogen (kg/ha)", "Phosphorus (kg/ha)", "Potassium (kg/ha)"]
box_colors = [C_GREEN,           C_AMBER,               C_PURPLE]

for ax, nutrient, ylabel, bcolor in zip(axes, nutrients, ylabels, box_colors):
    # Build list of value arrays — one per crop
    data_groups = [
        df_top.loc[df_top["crop"] == c, nutrient].dropna().values
        for c in top6
    ]
    # Safety check — skip empty groups
    data_groups = [g if len(g) > 0 else np.array([0]) for g in data_groups]

    bp = ax.boxplot(
        data_groups,
        patch_artist=True,
        labels=short_names,
        medianprops=dict(color="white", linewidth=2.2),
        whiskerprops=dict(linewidth=1.2, color="#555"),
        capprops=dict(linewidth=1.2, color="#555"),
        flierprops=dict(marker="o", markersize=3,
                        markerfacecolor=bcolor, alpha=0.4),
        widths=0.55,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(bcolor)
        patch.set_alpha(0.75)

    ax.set_title(f"{nutrient} — nutrient requirement",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)

fig.suptitle("Soil NPK requirements by crop — AgriSense India",
             fontsize=13, fontweight="bold", y=1.01)
fig.text(0.99, -0.03, "Source: Crop Recommendation Dataset — Atharva Ingle (Kaggle)",
         ha="right", va="bottom", fontsize=7, color="#888")
plt.tight_layout()
plt.savefig("assets/chart3_npk_boxplots.png")
plt.close()
print("  Saved: assets/chart3_npk_boxplots.png")


# ════════════════════════════════════════════════════════════
# CHART 4 — Rainfall vs temperature scatter
# Source: crop_recommendation.csv (rec_df)
# ════════════════════════════════════════════════════════════
print("[Chart 4] Rainfall vs temperature scatter...")

top10 = rec_df["crop"].value_counts().head(10).index.tolist()
df10  = rec_df[rec_df["crop"].isin(top10)].copy()

fig, ax = plt.subplots(figsize=(12, 7))

for i, crop in enumerate(top10):
    sub = df10[df10["crop"] == crop]
    ax.scatter(
        sub["rainfall"], sub["temperature"],
        label=crop.capitalize(),
        color=CROP_COLORS[i],
        alpha=0.60, s=28, edgecolors="none",
    )

# Quadrant guide lines
ax.axvline(x=df10["rainfall"].median(),   color="#ccc", linewidth=0.8, linestyle="--")
ax.axhline(y=df10["temperature"].median(), color="#ccc", linewidth=0.8, linestyle="--")

ax.set_xlabel("Annual rainfall (mm)", fontsize=11, labelpad=8)
ax.set_ylabel("Average temperature (°C)", fontsize=11, labelpad=8)
ax.set_title("Climate preferences — rainfall vs temperature by crop",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
          fontsize=9, framealpha=0.9, title="Crop", title_fontsize=9)
fig.text(0.99, 0.01, "Source: Crop Recommendation Dataset — Atharva Ingle (Kaggle)",
         ha="right", va="bottom", fontsize=7, color="#888")
plt.tight_layout()
plt.savefig("assets/chart4_rainfall_temp.png")
plt.close()
print("  Saved: assets/chart4_rainfall_temp.png")


# ════════════════════════════════════════════════════════════
# CHART 5 — pH violin plot
# Source: crop_recommendation.csv (rec_df)
# ════════════════════════════════════════════════════════════
print("[Chart 5] pH violin distribution by crop...")

top12 = rec_df["crop"].value_counts().head(12).index.tolist()
df12  = rec_df[rec_df["crop"].isin(top12)].copy()

# Filter crops that have enough data for a violin (>=5 points)
valid_crops = [
    c for c in top12
    if len(df12[df12["crop"] == c]["ph"].dropna()) >= 5
]
ph_data   = [df12[df12["crop"] == c]["ph"].dropna().values for c in valid_crops]
positions = list(range(len(valid_crops)))

fig, ax = plt.subplots(figsize=(14, 6))

parts = ax.violinplot(
    ph_data, positions=positions,
    showmedians=True, showextrema=True, widths=0.7,
)

for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(CROP_COLORS[i % len(CROP_COLORS)])
    pc.set_alpha(0.65)
    pc.set_edgecolor("white")
    pc.set_linewidth(0.5)

parts["cmedians"].set_color("white")
parts["cmedians"].set_linewidth(2)
parts["cmins"].set_color("#aaa")
parts["cmins"].set_linewidth(0.8)
parts["cmaxes"].set_color("#aaa")
parts["cmaxes"].set_linewidth(0.8)
parts["cbars"].set_color("#ccc")
parts["cbars"].set_linewidth(0.6)

ax.axhline(y=7.0, color=C_CORAL,  linewidth=1.2, linestyle="--",
           alpha=0.8, label="Neutral pH 7.0")
ax.axhline(y=6.0, color=C_AMBER,  linewidth=0.8, linestyle=":",
           alpha=0.7, label="Slightly acidic 6.0")

ax.set_xticks(positions)
ax.set_xticklabels(
    [c.capitalize()[:10] for c in valid_crops],
    rotation=35, ha="right", fontsize=9,
)
ax.set_ylabel("Soil pH", fontsize=11, labelpad=8)
ax.set_ylim(3.0, 10.5)
ax.set_title("Soil pH tolerance — top crops (violin width = data density)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(fontsize=9, loc="upper right")
ax.tick_params(axis="y", labelsize=9)
fig.text(0.99, 0.01, "Source: Crop Recommendation Dataset — Atharva Ingle (Kaggle)",
         ha="right", va="bottom", fontsize=7, color="#888")
plt.tight_layout()
plt.savefig("assets/chart5_ph_distribution.png")
plt.close()
print("  Saved: assets/chart5_ph_distribution.png")


# ════════════════════════════════════════════════════════════
# CHART 6 — Humidity requirements bar chart
# Source: crop_recommendation.csv (rec_df)
# ════════════════════════════════════════════════════════════
print("[Chart 6] Humidity requirements by crop...")

crop_hum  = (
    rec_df.groupby("crop")["humidity"]
    .agg(["mean", "std"])
    .sort_values("mean", ascending=False)
)
mean_line = crop_hum["mean"].mean()
bar_colors = [C_GREEN if v >= mean_line else C_TEAL
              for v in crop_hum["mean"]]

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(
    range(len(crop_hum)),
    crop_hum["mean"],
    color=bar_colors,
    edgecolor="white",
    linewidth=0.5,
    width=0.72,
    yerr=crop_hum["std"],
    capsize=3,
    error_kw={"elinewidth": 0.8, "ecolor": "#bbb"},
)
ax.axhline(y=mean_line, color=C_CORAL, linewidth=1.4,
           linestyle="--", alpha=0.85,
           label=f"Dataset mean: {mean_line:.1f}%")
ax.set_xticks(range(len(crop_hum)))
ax.set_xticklabels(
    [c.capitalize()[:10] for c in crop_hum.index],
    rotation=40, ha="right", fontsize=9,
)
ax.set_ylabel("Average humidity requirement (%)", fontsize=11, labelpad=8)
ax.set_ylim(0, 110)
ax.set_title("Humidity requirements by crop — mean ± std deviation",
             fontsize=13, fontweight="bold", pad=14)
ax.tick_params(axis="y", labelsize=9)

legend_patches = [
    mpatches.Patch(color=C_GREEN, label=f"Above mean ({mean_line:.0f}%)"),
    mpatches.Patch(color=C_TEAL,  label=f"Below mean ({mean_line:.0f}%)"),
]
ax.legend(handles=legend_patches + [ax.get_lines()[0]],
          fontsize=9, loc="upper right")
fig.text(0.99, 0.01, "Source: Crop Recommendation Dataset — Atharva Ingle (Kaggle)",
         ha="right", va="bottom", fontsize=7, color="#888")
plt.tight_layout()
plt.savefig("assets/chart6_humidity.png")
plt.close()
print("  Saved: assets/chart6_humidity.png")


# ════════════════════════════════════════════════════════════
# WEEK 1 — EDA SUMMARY (PRODUCTION DATASET)
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("AgriSense India — Week 1 EDA Summary")
print("="*60)

# Total records
print(f"Total records: {len(prod_df)}")

# Unique crops
print(f"Number of unique crops: {prod_df['crop'].nunique()}")

# Most common crop
most_common = prod_df['crop'].value_counts().idxmax()
print(f"Most common crop: {most_common}")

# Top 5 crops
print("\nTop 5 crops:")
print(prod_df['crop'].value_counts().head(5))

# Production stats
print(f"\nTotal production: {prod_df['production'].sum():,.2f}")
print(f"Average production: {prod_df['production'].mean():,.2f}")

# Area stats
print(f"Average area: {prod_df['area'].mean():,.2f}")

# Yield (important metric)
prod_df["yield"] = prod_df["production"] / prod_df["area"]
print(f"Average yield: {prod_df['yield'].mean():,.4f}")

# Missing + duplicates
print(f"\nMissing values: {prod_df.isnull().sum().sum()}")
print(f"Duplicate rows: {prod_df.duplicated().sum()}")

print("="*60)
print("EDA Completed Successfully 🚀")
print("="*60)