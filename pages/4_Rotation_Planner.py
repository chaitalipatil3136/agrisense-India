"""
AgriSense India — Enriched Crop Rotation Planner
File: pages/4_Rotation_Planner.py

Shows a complete 3-season farming calendar with:
- Exact sowing window
- Seed rate per acre
- Nitrogen savings from previous crop (in rupees)
- Which disease is broken by this rotation
- Water requirement vs previous crop
- Expected income range based on MSP 2024-25
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="Rotation Planner — AgriSense India",
    page_icon="🗓️",
    layout="wide",
)

# ════════════════════════════════════════════════════════════
# COMPLETE CROP DATABASE
# All values from ICAR published guidelines
# ════════════════════════════════════════════════════════════

CROP_DATA = {
    "cotton": {
        "display":       "Cotton",
        "season":        "Kharif",
        "sow_window":    "June 15 – July 15",
        "harvest_window":"October – December",
        "duration_days": 180,
        "seed_rate_kg_acre": 1.5,
        "seed_variety":  "Bt hybrid varieties (NHH-44, RCH-2)",
        "water_mm":      700,
        "n_uptake":      -100,   # negative = takes from soil
        "p_uptake":      -50,
        "k_uptake":      -80,
        "disease_family":"vascular_wilt",
        "breaks_disease":"Fusarium wilt, Root rot",
        "msп_per_quintal": 6620,
        "yield_range_kg_acre": (400, 700),
        "icar_n_rec":    120,
        "icar_p_rec":    60,
        "icar_k_rec":    60,
        "colour":        "#EF9F27",
        "icon":          "🌿",
        "key_tip":       "Sow in lines — do not broadcast. Use pink root rot resistant varieties in heavy soils.",
    },
    "soybean": {
        "display":       "Soybean",
        "season":        "Kharif",
        "sow_window":    "June 20 – July 10",
        "harvest_window":"October – November",
        "duration_days": 100,
        "seed_rate_kg_acre": 30,
        "seed_variety":  "JS-335, JS-9560, NRC-86 (ICAR approved)",
        "water_mm":      450,
        "n_uptake":      80,    # positive = fixes nitrogen
        "p_uptake":      -30,
        "k_uptake":      -40,
        "disease_family":"yellow_mosaic",
        "breaks_disease":"Cotton bollworm cycle, Root rot",
        "msп_per_quintal": 4600,
        "yield_range_kg_acre": (500, 900),
        "icar_n_rec":    30,    # low because it fixes its own N
        "icar_p_rec":    60,
        "icar_k_rec":    40,
        "colour":        "#1D9E75",
        "icon":          "🫘",
        "key_tip":       "Seed treatment with Rhizobium culture 25g/kg seed is mandatory — skipping it loses the nitrogen-fixing benefit.",
    },
    "wheat": {
        "display":       "Wheat",
        "season":        "Rabi",
        "sow_window":    "November 1 – November 25",
        "harvest_window":"March – April",
        "duration_days": 120,
        "seed_rate_kg_acre": 40,
        "seed_variety":  "GW-496, HD-2781, PBW-343 (zone specific)",
        "water_mm":      450,
        "n_uptake":      -60,
        "p_uptake":      -35,
        "k_uptake":      -30,
        "disease_family":"rust_smut",
        "breaks_disease":"Cotton pink bollworm, Soybean girdle beetle",
        "msп_per_quintal": 2275,
        "yield_range_kg_acre": (1200, 1800),
        "icar_n_rec":    120,
        "icar_p_rec":    60,
        "icar_k_rec":    40,
        "colour":        "#EF9F27",
        "icon":          "🌾",
        "key_tip":       "Timely sowing (before Nov 25) is the single most important factor for wheat yield. Late sowing reduces yield 1.5% per day.",
    },
    "chickpea": {
        "display":       "Chickpea (Chana)",
        "season":        "Rabi",
        "sow_window":    "October 20 – November 15",
        "harvest_window":"February – March",
        "duration_days": 110,
        "seed_rate_kg_acre": 30,
        "seed_variety":  "JG-11, JAKI-9218, Vijay (for Vidarbha)",
        "water_mm":      350,
        "n_uptake":      100,   # fixes 100 kg N/ha
        "p_uptake":      -30,
        "k_uptake":      -20,
        "disease_family":"wilt_blight",
        "breaks_disease":"Cotton bollworm Helicoverpa cycle, Stem borer",
        "msп_per_quintal": 5440,
        "yield_range_kg_acre": (500, 900),
        "icar_n_rec":    20,
        "icar_p_rec":    50,
        "icar_k_rec":    30,
        "colour":        "#BA7517",
        "icon":          "🫘",
        "key_tip":       "Inoculate seed with Rhizobium + PSB culture. Apply 20 kg N/ha as starter dose only — chickpea fixes rest.",
    },
    "rice": {
        "display":       "Rice (Paddy)",
        "season":        "Kharif",
        "sow_window":    "June 15 – July 15 (transplanting)",
        "harvest_window":"October – November",
        "duration_days": 130,
        "seed_rate_kg_acre": 20,
        "seed_variety":  "Swarna, MTU-1010, BPT-5204 (state specific)",
        "water_mm":      1200,
        "n_uptake":      -80,
        "p_uptake":      -40,
        "k_uptake":      -50,
        "disease_family":"blast_blight",
        "breaks_disease":"Wheat Karnal bunt, Yellow rust",
        "msп_per_quintal": 2183,
        "yield_range_kg_acre": (800, 1400),
        "icar_n_rec":    100,
        "icar_p_rec":    50,
        "icar_k_rec":    50,
        "colour":        "#5DCAA5",
        "icon":          "🌾",
        "key_tip":       "Maintain 2-3 cm standing water during tillering. Do not use DAP before transplanting — apply after establishment.",
    },
    "maize": {
        "display":       "Maize (Corn)",
        "season":        "Kharif / Rabi",
        "sow_window":    "June 15 – July 15 (Kharif) / Oct 15 – Nov 15 (Rabi)",
        "harvest_window":"September–October (Kharif) / February–March (Rabi)",
        "duration_days": 100,
        "seed_rate_kg_acre": 8,
        "seed_variety":  "DK-9144, Pioneer 30V92, Bioseed 9681",
        "water_mm":      500,
        "n_uptake":      -90,
        "p_uptake":      -40,
        "k_uptake":      -60,
        "disease_family":"northern_blight",
        "breaks_disease":"Cotton Fusarium wilt, Soybean yellow mosaic",
        "msп_per_quintal": 2090,
        "yield_range_kg_acre": (1000, 1600),
        "icar_n_rec":    120,
        "icar_p_rec":    60,
        "icar_k_rec":    40,
        "colour":        "#F5C842",
        "icon":          "🌽",
        "key_tip":       "Apply 1/3 N at sowing + 1/3 at knee height + 1/3 at tasseling for best results. Watch for Fall Armyworm in whorl stage.",
    },
    "pigeonpea": {
        "display":       "Pigeonpea (Tur Dal)",
        "season":        "Kharif",
        "sow_window":    "June 15 – July 5",
        "harvest_window":"January – February",
        "duration_days": 180,
        "seed_rate_kg_acre": 8,
        "seed_variety":  "BSMR-736, Maruti, ICPL-87119",
        "water_mm":      400,
        "n_uptake":      120,   # highest N fixer among Indian pulses
        "p_uptake":      -35,
        "k_uptake":      -25,
        "disease_family":"sterility_mosaic",
        "breaks_disease":"Cotton root rot, Soybean stem fly",
        "msп_per_quintal": 7000,
        "yield_range_kg_acre": (300, 600),
        "icar_n_rec":    20,
        "icar_p_rec":    50,
        "icar_k_rec":    30,
        "colour":        "#8B5CF6",
        "icon":          "🫘",
        "key_tip":       "Tur Dal fixes 120 kg N/ha — the most valuable rotation crop for Vidarbha. Deep taproot breaks hardpan and improves soil structure.",
    },
    "mustard": {
        "display":       "Mustard (Sarson)",
        "season":        "Rabi",
        "sow_window":    "October 1 – October 25",
        "harvest_window":"February – March",
        "duration_days": 110,
        "seed_rate_kg_acre": 2,
        "seed_variety":  "Pusa Bold, Kranti, Rohini",
        "water_mm":      300,
        "n_uptake":      -40,
        "p_uptake":      -25,
        "k_uptake":      -20,
        "disease_family":"alternaria_blight",
        "breaks_disease":"Wheat yellow rust, Barley stripe disease",
        "msп_per_quintal": 5650,
        "yield_range_kg_acre": (400, 700),
        "icar_n_rec":    100,
        "icar_p_rec":    40,
        "icar_k_rec":    40,
        "colour":        "#F59E0B",
        "icon":          "🌻",
        "key_tip":       "Earliest sowing (Oct 1-10) gives best yield. Mustard suppresses weeds through allelopathy — reduces weeding cost.",
    },
    "groundnut": {
        "display":       "Groundnut (Mungfali)",
        "season":        "Kharif",
        "sow_window":    "June 15 – July 10",
        "harvest_window":"October – November",
        "duration_days": 120,
        "seed_rate_kg_acre": 50,
        "seed_variety":  "TAG-24, GG-20, TG-37A",
        "water_mm":      500,
        "n_uptake":      50,
        "p_uptake":      -35,
        "k_uptake":      -30,
        "disease_family":"tikka_leaf_spot",
        "breaks_disease":"Cotton Fusarium wilt, Soybean girdle beetle",
        "msп_per_quintal": 6377,
        "yield_range_kg_acre": (600, 1000),
        "icar_n_rec":    20,
        "icar_p_rec":    40,
        "icar_k_rec":    40,
        "colour":        "#D97706",
        "icon":          "🥜",
        "key_tip":       "Apply gypsum 200 kg/acre at pegging stage — critical for pod filling. Groundnut needs calcium for pegs.",
    },
    "bajra": {
        "display":       "Bajra (Pearl Millet)",
        "season":        "Kharif",
        "sow_window":    "June 20 – July 15",
        "harvest_window":"September – October",
        "duration_days": 90,
        "seed_rate_kg_acre": 2.5,
        "seed_variety":  "HHB-67, Kaveri Super Boss, GHB-732",
        "water_mm":      350,
        "n_uptake":      -60,
        "p_uptake":      -30,
        "k_uptake":      -40,
        "disease_family":"downy_mildew",
        "breaks_disease":"Cotton bollworm, Soybean yellow mosaic",
        "msп_per_quintal": 2500,
        "yield_range_kg_acre": (600, 1000),
        "icar_n_rec":    80,
        "icar_p_rec":    40,
        "icar_k_rec":    30,
        "colour":        "#78716C",
        "icon":          "🌾",
        "key_tip":       "Bajra is the most drought-tolerant cereal crop. Ideal for low-rainfall areas (<400mm). Seed treatment with Thiram 4g/kg is mandatory.",
    },
}

# MSP 2024-25 Urea price
UREA_PRICE_PER_KG_N = 7.2    # ₹/kg N equivalent (Urea 45 kg bag = ₹242)
SEASON_COLORS = {
    "Kharif":       "#1D9E75",
    "Rabi":         "#7F77DD",
    "Kharif / Rabi":"#EF9F27",
    "Annual":       "#5DCAA5",
}

SEASON_MONTHS = {
    "Kharif": "Jun–Oct",
    "Rabi":   "Oct–Mar",
    "Zaid":   "Mar–Jun",
}


def get_rotation_suggestions(current_crop: str, n_seasons: int) -> list:
    """
    Returns a list of dicts representing the rotation plan.
    Each dict = one season's full information.
    """
    current   = current_crop.lower()
    plan      = [current]
    cur_fam   = CROP_DATA.get(current, {}).get("disease_family", "")

    for _ in range(n_seasons - 1):
        cur_data = CROP_DATA.get(plan[-1], {})
        avoid    = [plan[-1]]   # avoid same crop consecutively

        scored = []
        for crop, data in CROP_DATA.items():
            if crop in avoid:
                continue
            score = 0
            # Bonus: breaks disease family
            if data.get("disease_family","") != cur_fam:       score += 8
            # Bonus: nitrogen fixer after heavy feeder
            if data.get("n_uptake", 0) > 0:                    score += 6
            # Bonus: different season (Kharif after Rabi)
            if data.get("season","") != cur_data.get("season",""): score += 4
            # Bonus: lower water requirement
            if data.get("water_mm",999) < cur_data.get("water_mm",999): score += 2
            scored.append((crop, score))

        scored.sort(key=lambda x: -x[1])
        next_crop = scored[0][0] if scored else list(CROP_DATA.keys())[0]
        plan.append(next_crop)
        cur_fam = CROP_DATA.get(next_crop, {}).get("disease_family", "")

    # Build full season details
    result = []
    for i, crop_key in enumerate(plan):
        data = CROP_DATA.get(crop_key, {})
        prev = CROP_DATA.get(plan[i-1], {}) if i > 0 else {}

        # N savings from previous crop
        prev_n_added   = prev.get("n_uptake", 0) if i > 0 else 0
        n_saved_kg_ha  = max(0, prev_n_added)         # only positive = savings
        n_saved_rupees = n_saved_kg_ha * UREA_PRICE_PER_KG_N * 2.47  # per acre

        # Recommended N for this crop (reduced by previous crop's N contribution)
        this_n_rec     = data.get("icar_n_rec", 80)
        adjusted_n     = max(0, this_n_rec - n_saved_kg_ha)

        # Income estimate
        ymin, ymax     = data.get("yield_range_kg_acre", (500, 800))
        msp            = data.get("msп_per_quintal", 2000)
        income_min     = int(ymin * msp / 100)
        income_max     = int(ymax * msp / 100)

        # Water comparison
        prev_water     = prev.get("water_mm", 0) if i > 0 else 0
        this_water     = data.get("water_mm", 500)
        water_diff     = this_water - prev_water if i > 0 else 0

        result.append({
            "season_num":       i + 1,
            "crop_key":         crop_key,
            "display":          data.get("display", crop_key.capitalize()),
            "season":           data.get("season", ""),
            "sow_window":       data.get("sow_window", ""),
            "harvest_window":   data.get("harvest_window", ""),
            "duration_days":    data.get("duration_days", 100),
            "seed_rate":        data.get("seed_rate_kg_acre", 10),
            "seed_variety":     data.get("seed_variety", ""),
            "water_mm":         this_water,
            "water_diff":       water_diff,
            "n_uptake":         data.get("n_uptake", 0),
            "prev_n_added":     prev_n_added,
            "n_saved_kg_ha":    n_saved_kg_ha,
            "n_saved_rupees":   int(n_saved_rupees),
            "icar_n_rec":       this_n_rec,
            "adjusted_n":       adjusted_n,
            "icar_p_rec":       data.get("icar_p_rec", 40),
            "icar_k_rec":       data.get("icar_k_rec", 30),
            "breaks_disease":   data.get("breaks_disease", ""),
            "disease_family":   data.get("disease_family", ""),
            "msp":              msp,
            "yield_min":        ymin,
            "yield_max":        ymax,
            "income_min":       income_min,
            "income_max":       income_max,
            "colour":           data.get("colour", "#888"),
            "icon":             data.get("icon", "🌱"),
            "key_tip":          data.get("key_tip", ""),
        })

    return result


# ════════════════════════════════════════════════════════════
# PAGE UI
# ════════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(135deg,#0A2218,#1D9E75);
            border-radius:16px;padding:24px 28px;margin-bottom:20px;">
  <h1 style="color:white;font-size:28px;margin:0 0 6px">
    🗓️ Crop Rotation Planner
  </h1>
  <p style="color:#C8F0DF;font-size:14px;margin:0">
    Plan your next 3 seasons — get sowing dates, seed rates, fertilizer savings,
    disease protection, water needs, and income estimates for each crop.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────
col_in1, col_in2, col_in3 = st.columns([1.2, 0.8, 0.8])

with col_in1:
    current_crop = st.selectbox(
        "Your current crop this season",
        options=list(CROP_DATA.keys()),
        format_func=lambda k: f"{CROP_DATA[k]['icon']} {CROP_DATA[k]['display']}",
        index=0,
        help="The crop you are growing right now or just harvested",
    )

with col_in2:
    n_seasons = st.radio(
        "Plan for how many seasons?",
        options=[2, 3],
        index=1,
        horizontal=True,
    )

with col_in3:
    field_size = st.number_input(
        "Field size (acres)",
        min_value=0.5, max_value=50.0,
        value=2.0, step=0.5,
        help="Used to calculate total seed and fertilizer quantities",
    )

plan_btn = st.button(
    "🗓️ Generate My Rotation Plan",
    type="primary",
    use_container_width=True,
)

if not plan_btn:
    # Show current crop info
    cd = CROP_DATA[current_crop]
    st.markdown("---")
    st.info(
        f"**{cd['display']} — current season summary:**  \n"
        f"Season: {cd['season']} · "
        f"Sow: {cd['sow_window']} · "
        f"Duration: {cd['duration_days']} days · "
        f"Water: {cd['water_mm']}mm · "
        f"MSP: ₹{cd['msп_per_quintal']}/quintal"
    )
    st.markdown(f"💡 **Tip:** {cd['key_tip']}")
    st.stop()

# ── Generate plan ─────────────────────────────────────────────
plan = get_rotation_plan_data = get_rotation_suggestions(current_crop, n_seasons)

st.markdown("---")

# ════════════════════════════════════════════════════════════
# GANTT-STYLE CALENDAR CHART
# ════════════════════════════════════════════════════════════

st.markdown("### 📅 Your 3-Season Farming Calendar")
st.caption("Each block = one season. Hover over it for details.")

fig = go.Figure()

for i, s in enumerate(plan):
    fig.add_trace(go.Bar(
        x=[s["duration_days"]],
        y=[f"Season {s['season_num']}: {s['display']}"],
        orientation="h",
        marker=dict(
            color=s["colour"],
            opacity=0.85,
            line=dict(color="white", width=2),
        ),
        text=(
            f"<b>{s['display']}</b><br>"
            f"Sow: {s['sow_window']}<br>"
            f"Harvest: {s['harvest_window']}<br>"
            f"Income: ₹{s['income_min']:,}–₹{s['income_max']:,}/acre"
        ),
        textposition="inside",
        hovertemplate=(
            f"<b>{s['display']}</b><br>"
            f"Season: {s['season']}<br>"
            f"Sow: {s['sow_window']}<br>"
            f"Harvest: {s['harvest_window']}<br>"
            f"Duration: {s['duration_days']} days<br>"
            f"Income range: ₹{s['income_min']:,}–₹{s['income_max']:,}/acre<br>"
            f"<extra></extra>"
        ),
        name=s["display"],
    ))

fig.update_layout(
    barmode="stack",
    height=200 + len(plan) * 60,
    margin=dict(t=20, b=40, l=20, r=20),
    xaxis=dict(
        title="Duration (days)",
        showgrid=True,
        gridcolor="#f0f0f0",
    ),
    yaxis=dict(title=""),
    paper_bgcolor="white",
    plot_bgcolor="white",
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TOTAL N BALANCE BANNER
# ════════════════════════════════════════════════════════════

total_n   = sum(s["n_uptake"] for s in plan)
total_sav = sum(s["n_saved_rupees"] for s in plan)
total_inc = sum(s["income_min"] for s in plan)
total_inc_max = sum(s["income_max"] for s in plan)

n_color  = "#1D9E75" if total_n >= 0 else "#D85A30"
n_msg    = (
    f"This rotation ADDS {total_n} kg N/ha to your soil — "
    f"saving ₹{int(total_n * UREA_PRICE_PER_KG_N * 2.47):,}/acre on future Urea cost."
    if total_n > 0 else
    f"This rotation uses {abs(total_n)} kg N/ha from soil — "
    f"apply extra Urea in next season to compensate."
)

st.markdown(f"""
<div style="background:#F0FBF6;border-radius:12px;padding:14px 20px;
            border-left:5px solid {n_color};margin-bottom:16px;">
  <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;">
    <div>
      <p style="font-size:12px;color:#555;margin:0">Total nitrogen balance across all seasons</p>
      <p style="font-size:24px;font-weight:700;color:{n_color};margin:2px 0">
        {'+' if total_n >= 0 else ''}{total_n} kg N/ha
      </p>
      <p style="font-size:13px;color:#333;margin:0">{n_msg}</p>
    </div>
    <div style="text-align:right">
      <p style="font-size:12px;color:#555;margin:0">Total income range ({n_seasons} seasons)</p>
      <p style="font-size:22px;font-weight:700;color:#0A2218;margin:2px 0">
        ₹{total_inc:,} – ₹{total_inc_max:,}/acre
      </p>
      <p style="font-size:12px;color:#555;margin:0">Based on MSP 2024-25</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# DETAILED SEASON CARDS
# ════════════════════════════════════════════════════════════

st.markdown("### 📋 Season-by-Season Complete Guide")

for s in plan:
    season_label = (
        "🟢 Current season"
        if s["season_num"] == 1
        else f"Season {s['season_num']} — {s['season']}"
    )

    with st.expander(
        f"{s['icon']} **Season {s['season_num']}: {s['display']}** — "
        f"Sow: {s['sow_window']} · "
        f"Income: ₹{s['income_min']:,}–₹{s['income_max']:,}/acre",
        expanded=True,
    ):

        # ── Row 1: Sowing + Harvest + Duration ───────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""
        <div style="background:#F0FBF6;border-radius:8px;padding:10px 12px;
                    border-left:3px solid {s['colour']};">
          <p style="font-size:10px;color:#666;margin:0">📅 Sow between</p>
          <p style="font-size:14px;font-weight:600;color:#0A2218;margin:2px 0">
            {s['sow_window']}
          </p>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div style="background:#F0FBF6;border-radius:8px;padding:10px 12px;
                    border-left:3px solid {s['colour']};">
          <p style="font-size:10px;color:#666;margin:0">🌾 Harvest window</p>
          <p style="font-size:14px;font-weight:600;color:#0A2218;margin:2px 0">
            {s['harvest_window']}
          </p>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div style="background:#F0FBF6;border-radius:8px;padding:10px 12px;
                    border-left:3px solid {s['colour']};">
          <p style="font-size:10px;color:#666;margin:0">⏱️ Crop duration</p>
          <p style="font-size:14px;font-weight:600;color:#0A2218;margin:2px 0">
            {s['duration_days']} days
          </p>
        </div>
        """, unsafe_allow_html=True)

        c4.markdown(f"""
        <div style="background:#F0FBF6;border-radius:8px;padding:10px 12px;
                    border-left:3px solid {s['colour']};">
          <p style="font-size:10px;color:#666;margin:0">💰 Expected income</p>
          <p style="font-size:14px;font-weight:600;color:#0A2218;margin:2px 0">
            ₹{s['income_min']:,}–₹{s['income_max']:,}/acre
          </p>
          <p style="font-size:9px;color:#888;margin:0">
            MSP ₹{s['msp']}/quintal
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            # ── Seed information ──────────────────────────────
            st.markdown("**🌱 Seed & Sowing**")
            total_seed = s["seed_rate"] * field_size
            st.markdown(f"""
            | Detail | Value |
            |--------|-------|
            | Seed rate | {s['seed_rate']} kg/acre |
            | **Total seed needed** | **{total_seed:.1f} kg for {field_size} acres** |
            | Recommended variety | {s['seed_variety']} |
            | Season type | {s['season']} |
            """)

            # ── Nitrogen savings ──────────────────────────────
            st.markdown("**💰 Fertilizer Savings from Previous Crop**")
            if s["season_num"] == 1:
                st.markdown(
                    "*This is your first/current crop. "
                    "Savings will be shown for next season.*"
                )
            elif s["n_saved_kg_ha"] > 0:
                total_saving_field = s["n_saved_rupees"] * field_size
                st.markdown(f"""
                <div style="background:#E8F5E9;border-radius:8px;padding:10px 12px;
                            border-left:3px solid #1D9E75;">
                  <p style="font-size:13px;color:#0A2218;margin:0">
                    ✅ Previous crop added <strong>{s['n_saved_kg_ha']} kg N/ha</strong>
                    to your soil through nitrogen fixation.<br>
                    You can skip <strong>{s['n_saved_kg_ha']} kg N/ha</strong> of Urea this season.<br>
                    <strong>Estimated saving: ₹{s['n_saved_rupees']:,}/acre
                    (₹{total_saving_field:,.0f} for {field_size} acres)</strong>
                  </p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(
                    f"ICAR recommends {s['icar_n_rec']} kg N/ha for {s['display']}. "
                    f"After rotation credit: apply only **{s['adjusted_n']} kg N/ha**."
                )
            else:
                st.markdown(
                    f"Previous crop did not add nitrogen. "
                    f"Apply full ICAR recommended dose: **{s['icar_n_rec']} kg N/ha**."
                )

        with col_right:
            # ── Fertilizer recommendation ─────────────────────
            st.markdown("**🧪 Fertilizer to Apply This Season**")

            urea_kg   = math.ceil(s["adjusted_n"] / 0.46 * field_size)
            dap_kg    = math.ceil(s["icar_p_rec"] / 0.46 * field_size)
            mop_kg    = math.ceil(s["icar_k_rec"] / 0.60 * field_size)
            urea_bags = math.ceil(urea_kg / 50)
            dap_bags  = math.ceil(dap_kg / 50)
            mop_bags  = math.ceil(mop_kg / 50)

            st.markdown(f"""
            | Fertilizer | kg/ha | Total ({field_size} acres) | Bags (50kg) |
            |-----------|-------|--------------------------|------------|
            | **Urea (46% N)** | {s['adjusted_n']} | {urea_kg} kg | {urea_bags} bags |
            | **DAP (46% P)** | {s['icar_p_rec']} | {dap_kg} kg | {dap_bags} bags |
            | **MOP (60% K)** | {s['icar_k_rec']} | {mop_kg} kg | {mop_bags} bags |
            """)

            # ── Water requirement ─────────────────────────────
            st.markdown("**💧 Water Requirement**")
            if s["season_num"] > 1 and s["water_diff"] != 0:
                water_arrow = "🔺 More" if s["water_diff"] > 0 else "🔻 Less"
                water_diff_txt = f"{water_arrow} than previous crop by {abs(s['water_diff'])}mm"
            else:
                water_diff_txt = "First season baseline"

            water_status = (
                "🟢 Low water crop" if s["water_mm"] <= 400 else
                "🟡 Medium water crop" if s["water_mm"] <= 700 else
                "🔴 High water crop"
            )

            st.markdown(f"""
            <div style="background:#F0FBF6;border-radius:8px;padding:10px 12px;">
              <p style="margin:0;font-size:13px">
                <strong>{s['water_mm']} mm/season</strong> total water need<br>
                <span style="color:#666;font-size:12px">{water_status}</span><br>
                <span style="color:#666;font-size:11px">{water_diff_txt}</span>
              </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Disease break ─────────────────────────────────
            st.markdown("**🛡️ Disease Protection from Rotation**")
            if s["season_num"] > 1:
                st.markdown(f"""
                <div style="background:#F0F0FF;border-radius:8px;padding:10px 12px;
                            border-left:3px solid #7F77DD;">
                  <p style="font-size:13px;color:#0A2218;margin:0">
                    🦠 This rotation breaks the cycle of:<br>
                    <strong>{s['breaks_disease']}</strong><br>
                    <span style="font-size:11px;color:#666">
                      Disease organisms from previous crop cannot survive
                      on {s['display']} roots — natural soil disease break.
                    </span>
                  </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f"Disease family: **{s['disease_family'].replace('_',' ').title()}**. "
                    "Next crop will be chosen to break this disease cycle."
                )

        # ── Key tip ───────────────────────────────────────────
        if s["key_tip"]:
            st.info(f"💡 **Key tip for {s['display']}:** {s['key_tip']}")


# ════════════════════════════════════════════════════════════
# ROTATION RATIONALE
# ════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🔬 Why this rotation sequence?")

for i in range(1, len(plan)):
    prev = plan[i - 1]
    curr = plan[i]

    reasons = []
    if curr["n_saved_kg_ha"] > 0:
        reasons.append(
            f"**Nitrogen savings:** Previous {prev['display']} "
            f"fixed {curr['n_saved_kg_ha']} kg N/ha — "
            f"saving ₹{curr['n_saved_rupees']:,}/acre on Urea."
        )
    if curr["water_mm"] < prev["water_mm"]:
        reasons.append(
            f"**Water efficiency:** {curr['display']} needs "
            f"{prev['water_mm'] - curr['water_mm']}mm less water than {prev['display']}."
        )
    if curr["disease_family"] != prev["disease_family"]:
        reasons.append(
            f"**Disease break:** {curr['display']} breaks the "
            f"{prev['disease_family'].replace('_',' ')} disease cycle from "
            f"{prev['display']}."
        )
    if curr["season"] != prev["season"]:
        reasons.append(
            f"**Season fit:** {curr['display']} ({curr['season']}) fits perfectly "
            f"after {prev['display']} ({prev['season']}) harvest."
        )

    st.markdown(
        f"**{prev['icon']} {prev['display']} → "
        f"{curr['icon']} {curr['display']}:**"
    )
    for r in reasons:
        st.markdown(f"  - {r}")
    st.markdown("")


# ════════════════════════════════════════════════════════════
# INCOME COMPARISON CHART
# ════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 💰 Income Comparison Across Seasons")
st.caption(
    "Based on MSP 2024-25 × ICAR average yield range. "
    "Actual income depends on market price and your crop management."
)

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    name="Minimum expected income",
    x=[s["display"] for s in plan],
    y=[s["income_min"] * field_size for s in plan],
    marker_color=[s["colour"] for s in plan],
    opacity=0.65,
    text=[f"₹{s['income_min'] * field_size:,.0f}" for s in plan],
    textposition="outside",
))

fig2.add_trace(go.Bar(
    name="Maximum expected income",
    x=[s["display"] for s in plan],
    y=[s["income_max"] * field_size for s in plan],
    marker_color=[s["colour"] for s in plan],
    opacity=0.95,
    text=[f"₹{s['income_max'] * field_size:,.0f}" for s in plan],
    textposition="outside",
))

fig2.update_layout(
    barmode="group",
    height=340,
    margin=dict(t=20, b=40, l=60, r=20),
    xaxis=dict(title="Crop"),
    yaxis=dict(
        title=f"Income (₹) for {field_size} acres",
        tickformat=",",
    ),
    legend=dict(orientation="h", y=-0.25),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════
# WATER COMPARISON CHART
# ════════════════════════════════════════════════════════════

st.markdown("### 💧 Water Requirement Comparison")
st.caption(
    "Green = low water crop · Red = high water crop. "
    "Lower water need = lower irrigation cost."
)

water_colors = [
    "#1D9E75" if s["water_mm"] <= 400 else
    "#EF9F27" if s["water_mm"] <= 700 else
    "#D85A30"
    for s in plan
]

fig3 = go.Figure(go.Bar(
    x=[s["display"] for s in plan],
    y=[s["water_mm"] for s in plan],
    marker_color=water_colors,
    opacity=0.85,
    text=[f"{s['water_mm']}mm" for s in plan],
    textposition="outside",
))
fig3.update_layout(
    height=280,
    margin=dict(t=10, b=40, l=60, r=20),
    xaxis=dict(title="Crop"),
    yaxis=dict(title="Water needed per season (mm)"),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# SOURCE PANEL
# ════════════════════════════════════════════════════════════

st.markdown("---")
with st.expander("📂 Data sources and scientific references"):
    st.markdown("""
| Information | Source |
|-------------|--------|
| Sowing windows | ICAR Crop Production Guide (state-wise) |
| Seed rates | ICAR Package of Practices — Maharashtra, Punjab, MP |
| ICAR NPK recommendations | ICAR All India Coordinated Research Projects |
| Nitrogen fixation values | ICAR Soil Fertility and Fertilizer Use in India (2022) |
| MSP 2024-25 | Ministry of Agriculture & Farmers Welfare, Govt of India |
| Yield ranges | ICAR district-level productivity data · data.gov.in |
| Disease break information | ICAR-NCIPM Integrated Pest Management guidelines |
| Water requirements | FAO Irrigation and Drainage Paper 56 (Allen et al., 1998) |
    """)

st.caption(
    "Source: ICAR Crop Production Guide · "
    "ICAR Soil Fertility Manual · "
    "FAO Irrigation Paper 56 · "
    "MSP 2024-25 — Ministry of Agriculture, Govt of India · "
    "AgriSense India — MIT CSN Nagpur"
)