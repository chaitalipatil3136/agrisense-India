"""
AgriSense India — Crop Rotation Planner (Day 14)
File: notebooks/11_rotation_planner.py

Builds the scientific rotation rules engine.
Sources: ICAR Crop Rotation Guidelines + Soil Fertility Research.

Output: assets/rotation_rules.json    (knowledge base)
        utils/rotation.py             (Streamlit utility module)
        assets/rotation_calendar.png  (sample 3-season calendar)

Run: python notebooks/11_rotation_planner.py
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

os.makedirs("assets", exist_ok=True)
os.makedirs("utils",  exist_ok=True)

print("=" * 60)
print("AgriSense India — Crop Rotation Planner (Day 14)")
print("=" * 60)


# ════════════════════════════════════════════════════════════
# PART 1 — Build rotation_rules.json
# Scientific basis: ICAR Crop Rotation and Soil Health Guidelines
# ════════════════════════════════════════════════════════════

# Rotation rules structure:
#   For each crop, define:
#   - avoid_after: crops that SHOULD NOT precede this crop
#   - prefer_after: crops that are IDEAL to precede this crop
#   - nitrogen_impact: effect on soil N (+= adds, -= removes, 0 = neutral)
#   - disease_family: crops in the same family (share pathogens, avoid rotation)
#   - seasons: which Indian seasons this crop grows in
#   - rationale: plain-English explanation for the farmer

ROTATION_RULES = {
    "rice": {
        "avoid_after": ["rice", "wheat"],
        "prefer_after": ["groundnut", "blackgram", "pigeonpea", "soybean"],
        "nitrogen_impact": -80,
        "disease_family": "gramineae",
        "seasons": ["kharif"],
        "rationale": "Rice is a heavy nitrogen consumer. Growing legumes before rice replenishes soil N and breaks pest cycles. Avoid rice-rice rotation as it depletes nutrients and builds up blast disease."
    },
    "wheat": {
        "avoid_after": ["wheat", "barley"],
        "prefer_after": ["rice", "maize", "soybean", "lentil", "chickpea"],
        "nitrogen_impact": -60,
        "disease_family": "gramineae",
        "seasons": ["rabi"],
        "rationale": "Wheat after legumes reduces N fertilizer requirement by 30-40 kg/ha. Wheat-wheat rotation builds up Karnal bunt and rust diseases. Rice-wheat is India's most common rotation."
    },
    "cotton": {
        "avoid_after": ["cotton", "okra", "tomato"],
        "prefer_after": ["wheat", "sorghum", "maize", "groundnut"],
        "nitrogen_impact": -100,
        "disease_family": "malvaceae",
        "seasons": ["kharif"],
        "rationale": "Cotton is Maharashtra's major Kharif crop but a heavy soil exhaustor. Continuous cotton causes Verticillium wilt buildup. Groundnut before cotton adds nitrogen and breaks disease cycle."
    },
    "soybean": {
        "avoid_after": ["soybean", "groundnut", "lentil"],
        "prefer_after": ["cotton", "maize", "sorghum", "wheat"],
        "nitrogen_impact": 80,
        "disease_family": "leguminosae",
        "seasons": ["kharif"],
        "rationale": "Soybean fixes 80-120 kg N/ha from atmosphere — making it valuable before cereals. Avoid back-to-back legumes as they share root diseases. Soybean-wheat is a highly productive rotation in MP and Maharashtra."
    },
    "maize": {
        "avoid_after": ["maize", "sorghum"],
        "prefer_after": ["soybean", "groundnut", "chickpea", "lentil"],
        "nitrogen_impact": -90,
        "disease_family": "gramineae",
        "seasons": ["kharif", "rabi", "zaid"],
        "rationale": "Maize is a high-yield but nutrient-intensive crop. After soybean, maize needs 30 kg less nitrogen fertilizer. Avoid maize after sorghum — they share stalk borer pests and grey leaf spot."
    },
    "groundnut": {
        "avoid_after": ["groundnut", "soybean", "blackgram"],
        "prefer_after": ["cotton", "sorghum", "maize", "wheat"],
        "nitrogen_impact": 50,
        "disease_family": "leguminosae",
        "seasons": ["kharif", "rabi"],
        "rationale": "Groundnut fixes 50-70 kg N/ha. After cereals, it improves soil structure and nitrogen. Back-to-back groundnut causes tikka leaf spot and collar rot buildup. Cotton-groundnut-cotton is a highly sustainable rotation."
    },
    "pigeonpea": {
        "avoid_after": ["pigeonpea", "blackgram", "soybean"],
        "prefer_after": ["cotton", "maize", "rice", "sorghum"],
        "nitrogen_impact": 120,
        "disease_family": "leguminosae",
        "seasons": ["kharif"],
        "rationale": "Pigeonpea (Tur Dal) fixes 120 kg N/ha — the highest of all pulse crops. Roots reach 2m deep, breaking plough pan and improving soil structure. After pigeonpea, cereals and cotton show 20-30% yield increases."
    },
    "chickpea": {
        "avoid_after": ["chickpea", "lentil", "pea"],
        "prefer_after": ["rice", "maize", "cotton", "sorghum"],
        "nitrogen_impact": 100,
        "disease_family": "leguminosae",
        "seasons": ["rabi"],
        "rationale": "Chickpea (Chana) is the classic Rabi legume after Kharif cereals. Fixes 80-100 kg N/ha. Avoid after lentil — shared Ascochyta blight pathogen. Rice-chickpea is a productive, sustainable rotation."
    },
    "blackgram": {
        "avoid_after": ["blackgram", "pigeonpea", "soybean"],
        "prefer_after": ["rice", "maize", "cotton", "sorghum"],
        "nitrogen_impact": 60,
        "disease_family": "leguminosae",
        "seasons": ["kharif", "zaid"],
        "rationale": "Blackgram (Urad) is a short-duration legume perfect for Zaid or as a break crop. Fixes 60 kg N/ha. Excellent for diversifying the rotation and improving soil health quickly."
    },
    "sugarcane": {
        "avoid_after": ["sugarcane"],
        "prefer_after": ["soybean", "groundnut", "wheat", "maize"],
        "nitrogen_impact": -120,
        "disease_family": "gramineae",
        "seasons": ["kharif", "annual"],
        "rationale": "Sugarcane is a 10-12 month crop and the heaviest soil nutrient exhaustor. Ratoon (regrowth) crop possible for 2-3 years. After sugarcane, grow nitrogen-fixing legumes for at least one season to restore soil fertility."
    },
    "mustard": {
        "avoid_after": ["mustard", "cabbage", "cauliflower"],
        "prefer_after": ["rice", "cotton", "maize", "groundnut"],
        "nitrogen_impact": -40,
        "disease_family": "brassicaceae",
        "seasons": ["rabi"],
        "rationale": "Mustard is a valuable oilseed Rabi crop. Avoid continuous Brassica crops — they share Sclerotinia and Alternaria diseases. After Kharif cereals, mustard fits perfectly and has allelopathic weed suppression."
    },
    "lentil": {
        "avoid_after": ["lentil", "chickpea", "pea"],
        "prefer_after": ["rice", "maize", "cotton", "wheat"],
        "nitrogen_impact": 80,
        "disease_family": "leguminosae",
        "seasons": ["rabi"],
        "rationale": "Lentil (Masoor) is a cool-season Rabi legume. Fixes 80 kg N/ha. Ideal after Kharif rice in eastern India (Bihar, West Bengal). Avoid after other legumes due to shared soil pathogens."
    },
    "mung bean": {
        "avoid_after": ["mung bean", "blackgram", "cowpea"],
        "prefer_after": ["wheat", "rice", "maize", "cotton"],
        "nitrogen_impact": 60,
        "disease_family": "leguminosae",
        "seasons": ["kharif", "zaid"],
        "rationale": "Mung bean (Moong) is the fastest rotating legume — matures in 60-70 days. Perfect Zaid crop between Rabi and Kharif. Fixes 60 kg N/ha and its residue decomposes quickly, benefiting the next crop."
    },
    "jowar": {
        "avoid_after": ["jowar", "maize"],
        "prefer_after": ["soybean", "groundnut", "lentil", "blackgram"],
        "nitrogen_impact": -70,
        "disease_family": "gramineae",
        "seasons": ["kharif", "rabi"],
        "rationale": "Sorghum (Jowar) is drought-tolerant but competes with maize for soil resources. After legumes, yields improve by 15-20%. Jowar roots are very deep and improve soil structure for subsequent crops."
    },
    "bajra": {
        "avoid_after": ["bajra", "jowar"],
        "prefer_after": ["groundnut", "blackgram", "mung bean"],
        "nitrogen_impact": -60,
        "disease_family": "gramineae",
        "seasons": ["kharif"],
        "rationale": "Pearl millet (Bajra) is the primary Kharif cereal in arid Rajasthan. After legumes, downy mildew pressure decreases. Bajra-mustard is the classic rotation in Rajasthan and Haryana arid zones."
    },
    "sunflower": {
        "avoid_after": ["sunflower", "safflower"],
        "prefer_after": ["chickpea", "lentil", "wheat", "maize"],
        "nitrogen_impact": -60,
        "disease_family": "asteraceae",
        "seasons": ["rabi", "kharif"],
        "rationale": "Sunflower is a good catch crop. Avoid back-to-back planting — Sclerotinia head rot persists in soil 5+ years. After Rabi pulses, sunflower benefits from residual nitrogen and improved soil water retention."
    },
}

# Save rotation_rules.json
rules_path = "assets/rotation_rules.json"
with open(rules_path, "w", encoding="utf-8") as f:
    json.dump(ROTATION_RULES, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {rules_path} ({len(ROTATION_RULES)} crop rotation rules)")


# ════════════════════════════════════════════════════════════
# PART 2 — Test the rotation engine
# ════════════════════════════════════════════════════════════

def get_rotation_plan(current_crop: str, rules: dict, n_seasons: int = 3) -> list:
    """
    Given current crop, recommend next n_seasons crops.
    Logic:
    1. Avoid crops in avoid_after list
    2. Prefer crops in prefer_after list
    3. Prefer crops from different disease_family
    4. Alternate N-fixing and N-consuming crops
    """
    plan = [current_crop.lower()]
    current = current_crop.lower()

    seasons = ["kharif", "rabi", "zaid", "kharif", "rabi"]
    season_idx = 1   # start from next season

    for i in range(n_seasons - 1):
        current_rule = rules.get(current, {})
        avoid   = current_rule.get("avoid_after", [])
        prefer  = current_rule.get("prefer_after", [])
        cur_fam = current_rule.get("disease_family", "unknown")

        # Score all crops
        scored = []
        for crop, rule in rules.items():
            if crop == current:
                continue
            if crop in avoid:
                continue

            score = 0
            if crop in prefer:
                score += 10
            if rule.get("disease_family") != cur_fam:
                score += 5
            n_impact = rule.get("nitrogen_impact", 0)
            if n_impact > 0:    # nitrogen fixer
                score += 3
            # Prefer crops that grow in the target season
            target_season = seasons[season_idx % len(seasons)]
            if target_season in rule.get("seasons", []):
                score += 4

            scored.append((crop, score))

        scored.sort(key=lambda x: -x[1])
        if scored:
            next_crop = scored[0][0]
        else:
            # Fallback: any legume
            next_crop = next(
                (c for c, r in rules.items()
                 if r.get("nitrogen_impact", 0) > 0 and c not in plan),
                list(rules.keys())[0]
            )

        plan.append(next_crop)
        current = next_crop
        season_idx += 1

    return plan


# Test on 5 common starting crops
print("\nRotation recommendations (3 seasons):")
test_crops = ["cotton", "rice", "wheat", "maize", "sugarcane"]
SEASON_LABELS = ["Season 1 (Kharif)", "Season 2 (Rabi)", "Season 3 (Zaid/Kharif)"]

for crop in test_crops:
    if crop not in ROTATION_RULES:
        continue
    plan = get_rotation_plan(crop, ROTATION_RULES, 3)
    print(f"\n  Starting crop: {crop.upper()}")
    for i, (season, c) in enumerate(zip(SEASON_LABELS, plan)):
        n_imp = ROTATION_RULES.get(c, {}).get("nitrogen_impact", 0)
        n_str = f"(+{n_imp} kg N/ha)" if n_imp > 0 else f"({n_imp} kg N/ha)"
        print(f"    {season}: {c.capitalize():15s} {n_str}")
    rationales = [ROTATION_RULES.get(c, {}).get("rationale", "")
                  for c in plan[1:]]
    print(f"    Key reason: {rationales[0][:80]}..." if rationales[0] else "")


# ════════════════════════════════════════════════════════════
# PART 3 — Rotation calendar chart
# ════════════════════════════════════════════════════════════
print("\nGenerating rotation calendar chart...")

SEASON_MONTHS = {
    "kharif":  ("June", "October"),
    "rabi":    ("October", "March"),
    "zaid":    ("March", "June"),
}
SEASON_COLORS = {
    "kharif":  "#1D9E75",
    "rabi":    "#EF9F27",
    "zaid":    "#7F77DD",
}

example_rotation = get_rotation_plan("cotton", ROTATION_RULES, 3)
example_seasons  = ["kharif", "rabi", "zaid"]

fig, ax = plt.subplots(figsize=(12, 5))

for i, (crop, season) in enumerate(zip(example_rotation, example_seasons)):
    months = SEASON_MONTHS[season]
    color  = SEASON_COLORS[season]
    rule   = ROTATION_RULES.get(crop, {})
    n_imp  = rule.get("nitrogen_impact", 0)
    n_str  = f"N: +{n_imp}" if n_imp > 0 else f"N: {n_imp}"

    rect = plt.Rectangle(
        (i * 4, 0.5), 3.6, 1.5,
        color=color, alpha=0.80,
    )
    ax.add_patch(rect)
    ax.text(i * 4 + 1.8, 1.55, crop.capitalize(),
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")
    ax.text(i * 4 + 1.8, 1.15, f"{months[0]} – {months[1]}",
            ha="center", va="center",
            fontsize=9, color="white", alpha=0.9)
    ax.text(i * 4 + 1.8, 0.80, n_str + " kg N/ha",
            ha="center", va="center",
            fontsize=9, color="white", alpha=0.9)

    if i < len(example_rotation) - 1:
        ax.annotate(
            "", xy=(i * 4 + 4, 1.25), xytext=(i * 4 + 3.62, 1.25),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
        )

ax.set_xlim(-0.3, 12.3)
ax.set_ylim(0, 2.5)
ax.axis("off")
ax.set_title(
    "AgriSense India — Recommended 3-Season Rotation\n"
    "Starting crop: Cotton (Maharashtra, Kharif)",
    fontsize=13, fontweight="bold", pad=14,
)

season_patches = [
    mpatches.Patch(color=SEASON_COLORS["kharif"], alpha=0.8, label="Kharif (Jun–Oct)"),
    mpatches.Patch(color=SEASON_COLORS["rabi"],   alpha=0.8, label="Rabi (Oct–Mar)"),
    mpatches.Patch(color=SEASON_COLORS["zaid"],   alpha=0.8, label="Zaid (Mar–Jun)"),
]
ax.legend(handles=season_patches, fontsize=9, loc="lower center",
          ncol=3, bbox_to_anchor=(0.5, -0.05))

fig.text(0.99, 0.01, "Source: ICAR Crop Rotation and Soil Fertility Guidelines",
         ha="right", va="bottom", fontsize=7, color="#888")
plt.tight_layout()
plt.savefig("assets/rotation_calendar.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: assets/rotation_calendar.png")


# ════════════════════════════════════════════════════════════
# PART 4 — Write utils/rotation.py
# ════════════════════════════════════════════════════════════
print("\nWriting utils/rotation.py...")

rotation_module = '''"""
AgriSense India — Crop Rotation Utility Module
File: utils/rotation.py

Import: from utils.rotation import get_rotation_plan, get_rotation_benefits
"""

import json, os

_RULES_PATH = "assets/rotation_rules.json"
_rules_cache = None


def _load_rules():
    global _rules_cache
    if _rules_cache is None:
        if not os.path.exists(_RULES_PATH):
            raise FileNotFoundError(f"Rotation rules not found: {_RULES_PATH}")
        with open(_RULES_PATH, encoding="utf-8") as f:
            _rules_cache = json.load(f)
    return _rules_cache


def get_rotation_plan(current_crop, n_seasons=3):
    rules = _load_rules()
    plan  = [current_crop.lower()]
    current = current_crop.lower()
    seasons = ["kharif","rabi","zaid","kharif","rabi"]
    season_idx = 1

    for _ in range(n_seasons - 1):
        rule   = rules.get(current, {})
        avoid  = rule.get("avoid_after", [])
        prefer = rule.get("prefer_after", [])
        cur_fam = rule.get("disease_family", "")

        scored = []
        for crop, r in rules.items():
            if crop == current or crop in avoid:
                continue
            score = 0
            if crop in prefer: score += 10
            if r.get("disease_family","") != cur_fam: score += 5
            if r.get("nitrogen_impact", 0) > 0: score += 3
            if seasons[season_idx % len(seasons)] in r.get("seasons",[]): score += 4
            scored.append((crop, score))

        scored.sort(key=lambda x: -x[1])
        next_crop = scored[0][0] if scored else list(rules.keys())[0]
        plan.append(next_crop)
        current = next_crop
        season_idx += 1

    return plan


def get_rotation_benefits(plan):
    rules = _load_rules()
    benefits = []
    total_n  = 0
    for i, crop in enumerate(plan):
        r = rules.get(crop, {})
        n_imp = r.get("nitrogen_impact", 0)
        total_n += n_imp
        benefits.append({
            "season": i + 1,
            "crop": crop,
            "nitrogen_impact": n_imp,
            "rationale": r.get("rationale", ""),
            "seasons": r.get("seasons", []),
        })
    return {"plan": benefits, "total_n_balance": total_n}
'''

with open("utils/rotation.py", "w", encoding="utf-8") as f:
    f.write(rotation_module)
print("  Saved: utils/rotation.py")


# ── Final summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Day 14 — Week 3 complete!")
files = [
    "assets/rotation_rules.json",
    "utils/rotation.py",
    "assets/rotation_calendar.png",
    "utils/carbon.py",
    "assets/carbon_comparison.png",
    "assets/carbon_breakdown.png",
]
for f in files:
    print(f"  [{'OK' if os.path.exists(f) else 'MISSING'}] {f}")

print("\nWeek 3 git commit:")
print('  git add assets/ utils/ notebooks/ models/')
print('  git commit -m "Week 3 complete: CNN disease detector, carbon estimator, rotation planner"')
print("  git push")
print("\nWeek 4 starts tomorrow — Streamlit app assembly!")
