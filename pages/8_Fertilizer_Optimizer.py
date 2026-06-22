"""
AgriSense India — Fertilizer Optimizer
File: pages/10_Fertilizer_Optimizer.py

Takes farmer's current soil NPK (from Soil Health Card) + recommended crop.
Calculates exact fertilizer bags to buy + cost + savings vs over-application.

No ML needed — pure ICAR agronomic science + arithmetic.
Build time: ~6 hours.

Run: streamlit run app.py → navigate to Fertilizer Optimizer
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.irrigation import calculate_irrigation
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fertilizer Optimizer — AgriSense India",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Fertilizer Optimizer")
st.caption(
    "Enter your Soil Health Card values → get exact fertilizer bags to buy "
    "and savings vs over-application"
)

# ── ICAR recommended NPK (kg/ha) per crop ────────────────────
# Source: ICAR Crop Production Guide + State Agriculture Department advisories
ICAR_NPK = {
    "Cotton":      {"N": 120, "P": 60, "K": 60},
    "Rice":        {"N": 100, "P": 50, "K": 50},
    "Wheat":       {"N": 120, "P": 60, "K": 40},
    "Maize":       {"N": 120, "P": 60, "K": 40},
    "Soybean":     {"N":  30, "P": 60, "K": 40},
    "Sugarcane":   {"N": 250, "P": 80, "K": 100},
    "Groundnut":   {"N":  20, "P": 40, "K": 40},
    "Chickpea":    {"N":  20, "P": 50, "K": 30},
    "Tur Dal":     {"N":  20, "P": 50, "K": 30},
    "Bajra":       {"N":  80, "P": 40, "K": 30},
    "Jowar":       {"N":  80, "P": 40, "K": 30},
    "Sunflower":   {"N":  90, "P": 60, "K": 60},
    "Mustard":     {"N": 100, "P": 40, "K": 40},
    "Onion":       {"N": 100, "P": 50, "K": 80},
    "Tomato":      {"N": 120, "P": 80, "K": 100},
    "Potato":      {"N": 120, "P": 100, "K": 100},
}

# ── Fertilizer conversion factors ────────────────────────────
# Urea:  46% N  → 1 kg N = 2.174 kg Urea
# DAP:   18% N + 46% P₂O₅ → 1 kg P = 2.174 kg DAP (approx)
# MOP:   60% K₂O → 1 kg K = 1.667 kg MOP
UREA_PER_KG_N  = 1 / 0.46   # = 2.174
DAP_PER_KG_P   = 1 / 0.46   # = 2.174 (DAP 46% P₂O₅, P₂O₅ mol ratio ≈1)
MOP_PER_KG_K   = 1 / 0.60   # = 1.667

# Bag sizes (kg per bag)
UREA_BAG  = 50  # kg
DAP_BAG   = 50  # kg
MOP_BAG   = 50  # kg

# Current prices per bag (₹) — update from local market
PRICE_UREA = 300
PRICE_DAP  = 1350
PRICE_MOP  = 800


def calculate_fertilizer(
    crop: str,
    field_ha: float,
    soil_n: float,
    soil_p: float,
    soil_k: float,
) -> dict:
    """
    Calculate fertilizer requirement.
    soil_n/p/k = current soil values from Soil Health Card (kg/ha)
    """
    rec = ICAR_NPK[crop]
    target_n = rec["N"]
    target_p = rec["P"]
    target_k = rec["K"]

    # Soil availability factor: assume 50% of soil N, 25% P, 75% K is plant-available
    avail_n = soil_n * 0.50
    avail_p = soil_p * 0.25
    avail_k = soil_k * 0.75

    # Required fertilizer (kg/ha) — never go below 0
    need_n = max(0, target_n - avail_n)
    need_p = max(0, target_p - avail_p)
    need_k = max(0, target_k - avail_k)

    # Convert to fertilizer quantities for total field
    urea_kg  = need_n * UREA_PER_KG_N  * field_ha
    dap_kg   = need_p * DAP_PER_KG_P   * field_ha
    mop_kg   = need_k * MOP_PER_KG_K   * field_ha

    # Bags (ceiling)
    import math
    urea_bags = math.ceil(urea_kg / UREA_BAG)
    dap_bags  = math.ceil(dap_kg  / DAP_BAG)
    mop_bags  = math.ceil(mop_kg  / MOP_BAG)

    # Cost
    cost_urea = urea_bags * PRICE_UREA
    cost_dap  = dap_bags  * PRICE_DAP
    cost_mop  = mop_bags  * PRICE_MOP
    total_cost = cost_urea + cost_dap + cost_mop

    # "Average farmer" cost — Indian farmers apply 40% more than needed
    avg_farmer_n = target_n * 1.4
    avg_farmer_urea_cost = (avg_farmer_n * UREA_PER_KG_N * field_ha / UREA_BAG) * PRICE_UREA
    avg_farmer_dap_cost  = (target_p * 1.4 * DAP_PER_KG_P * field_ha / DAP_BAG) * PRICE_DAP
    avg_farmer_mop_cost  = (target_k * 1.4 * MOP_PER_KG_K * field_ha / MOP_BAG) * PRICE_MOP
    avg_farmer_total     = avg_farmer_urea_cost + avg_farmer_dap_cost + avg_farmer_mop_cost

    savings = max(0, avg_farmer_total - total_cost)

    return {
        "crop":        crop,
        "field_ha":    field_ha,
        "target_n":    target_n,
        "target_p":    target_p,
        "target_k":    target_k,
        "avail_n":     round(avail_n, 1),
        "avail_p":     round(avail_p, 1),
        "avail_k":     round(avail_k, 1),
        "need_n":      round(need_n, 1),
        "need_p":      round(need_p, 1),
        "need_k":      round(need_k, 1),
        "urea_kg":     round(urea_kg,  1),
        "dap_kg":      round(dap_kg,   1),
        "mop_kg":      round(mop_kg,   1),
        "urea_bags":   urea_bags,
        "dap_bags":    dap_bags,
        "mop_bags":    mop_bags,
        "cost_urea":   cost_urea,
        "cost_dap":    cost_dap,
        "cost_mop":    cost_mop,
        "total_cost":  total_cost,
        "savings_vs_average": round(savings),
    }


def classify_level(current, optimal):
    """Return status label and colour based on % of optimal."""
    pct = (current / optimal * 100) if optimal > 0 else 100
    if pct < 40:   return "Very Low",   "#D85A30"
    if pct < 70:   return "Low",        "#EF9F27"
    if pct < 120:  return "Optimal",    "#1D9E75"
    return             "High",          "#7F77DD"


# ── UI ────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 1.4])

with col_in:
    st.markdown("### Step 1 — Your farm details")
    crop      = st.selectbox("Recommended crop", list(ICAR_NPK.keys()))
    field_ha  = st.slider("Field size (hectares)", 0.5, 20.0, 2.0, 0.5)

    st.markdown("### Step 2 — Your Soil Health Card values")
    st.caption("Found on the green card given by government soil testing")
    soil_n = st.number_input("Nitrogen — N (kg/ha)", 0.0, 500.0, 120.0,
                              help="Available Nitrogen value from your Soil Health Card")
    soil_p = st.number_input("Phosphorus — P (kg/ha)", 0.0, 200.0, 35.0,
                              help="Available Phosphorus value from your Soil Health Card")
    soil_k = st.number_input("Potassium — K (kg/ha)", 0.0, 600.0, 180.0,
                              help="Available Potassium value from your Soil Health Card")

    calculate = st.button("🧪 Calculate Fertilizer Need", type="primary",
                          use_container_width=True)

with col_out:
    if calculate:
        result = calculate_fertilizer(crop, field_ha, soil_n, soil_p, soil_k)
        rec    = ICAR_NPK[crop]

        # ── Savings banner ─────────────────────────────────
        savings = result["savings_vs_average"]
        if savings > 0:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0A1628,#0F2D1E);
                        border-radius:12px;padding:16px 20px;margin-bottom:16px;
                        border-left:4px solid #1D9E75;">
              <p style="color:#9BBFA0;font-size:12px;margin:0">
                You save vs average farmer
              </p>
              <p style="color:#FFFFFF;font-size:32px;font-weight:700;margin:0">
                ₹{savings:,}
              </p>
              <p style="color:#9BBFA0;font-size:12px;margin:4px 0 0">
                Indian farmers apply 40% excess fertilizer on average.
                AgriSense gives you the exact amount.
              </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Soil status cards ───────────────────────────────
        st.markdown("**Current soil status:**")
        c1, c2, c3 = st.columns(3)
        for col, nut, val, opt in [
            (c1, "N", soil_n, rec["N"]),
            (c2, "P", soil_p, rec["P"]),
            (c3, "K", soil_k, rec["K"]),
        ]:
            label, color = classify_level(val, opt)
            with col:
                st.markdown(f"""
                <div style="background:{color}22;border:1px solid {color};
                            border-radius:8px;padding:10px;text-align:center;">
                  <p style="font-size:18px;font-weight:700;color:{color};margin:0">{nut}</p>
                  <p style="font-size:22px;font-weight:500;color:#FFF;margin:0">{val:.0f}</p>
                  <p style="font-size:10px;color:{color};margin:0">{label}</p>
                  <p style="font-size:10px;color:#888;margin:0">Optimal: {opt}</p>
                </div>
                """, unsafe_allow_html=True)

        # ── Fertilizer bags to buy ──────────────────────────
        st.markdown("---")
        st.markdown(f"**Fertilizer to buy for {field_ha} ha of {crop}:**")

        fert_data = [
            ("Urea (46% N)",    result["urea_bags"],  result["urea_kg"],
             result["cost_urea"],  "#1D9E75", "Provides Nitrogen"),
            ("DAP (46% P₂O₅)", result["dap_bags"],   result["dap_kg"],
             result["cost_dap"],   "#EF9F27", "Provides Phosphorus"),
            ("MOP (60% K₂O)",  result["mop_bags"],   result["mop_kg"],
             result["cost_mop"],   "#7F77DD", "Provides Potassium"),
        ]

        for name, bags, kg, cost, color, desc in fert_data:
            st.markdown(f"""
            <div style="background:var(--secondary-bg,#F4F9F7);border-radius:8px;
                        padding:10px 14px;margin-bottom:8px;
                        border-left:3px solid {color};">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:14px;font-weight:500;color:#0A1628">{name}</span>
                <span style="font-size:12px;color:#555">{desc}</span>
              </div>
              <div style="display:flex;gap:24px;margin-top:6px;">
                <div><span style="font-size:20px;font-weight:700;color:{color}">{bags}</span>
                     <span style="font-size:11px;color:#888"> bags (50 kg)</span></div>
                <div><span style="font-size:14px;color:#555">{kg:.0f} kg total</span></div>
                <div><span style="font-size:14px;font-weight:500;color:#0A1628">₹{cost:,}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Total cost
        st.metric("Total fertilizer cost", f"₹{result['total_cost']:,}",
                  delta=f"Save ₹{savings:,} vs average farmer" if savings > 0 else None,
                  delta_color="normal")

        # ── Visual: required vs available ───────────────────
        st.markdown("---")
        fig = go.Figure()
        nutrients = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]
        avail = [result["avail_n"], result["avail_p"], result["avail_k"]]
        needed = [result["need_n"],  result["need_p"],  result["need_k"]]
        target = [rec["N"],          rec["P"],           rec["K"]]

        fig.add_trace(go.Bar(
            name="Already in soil",
            x=nutrients, y=avail,
            marker_color="#1D9E75", opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            name="Need to add",
            x=nutrients, y=needed,
            marker_color="#EF9F27", opacity=0.85,
        ))
        fig.update_layout(
            barmode="stack",
            title=f"Soil nutrients — current vs target for {crop}",
            yaxis_title="kg per hectare",
            height=280,
            margin=dict(t=40, b=20),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)
        # ==============================
# 💧 IRRIGATION WATER SECTION
# ==============================

st.markdown("---")
st.markdown("## 💧 Irrigation Water Requirement")

# You can later replace this with weather API
expected_rainfall_mm = st.number_input(
    "Expected seasonal rainfall (mm)", 
    value=600
)

pump_hp = st.selectbox("Pump Horsepower", [1, 2, 3, 5], index=2)
electricity_rate = st.number_input("Electricity Rate (₹/unit)", value=5.5)

# Calculate irrigation
irrigation = calculate_irrigation(
    crop,
    field_ha,
    expected_rainfall_mm,
    pump_hp,
    electricity_rate
)

# Metrics row
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Need", f"{irrigation['water_req_mm']} mm")
c2.metric("Rainfall", f"{irrigation['rainfall_mm']} mm")
c3.metric("Deficit", f"{irrigation['deficit_mm']} mm")
c4.metric("Pump Hours", f"{irrigation['pump_hours']} hrs")
c5.metric("Cost", f"₹{irrigation['cost']}")

# Gauge chart
fig2 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=irrigation["deficit_mm"],
    title={'text': "Water Deficit (mm)"},
    gauge={
        'axis': {'range': [0, irrigation["water_req_mm"]]},
        'bar': {'color': "red"},
        'steps': [
            {'range': [0, irrigation["water_req_mm"] * 0.4], 'color': "green"},
            {'range': [irrigation["water_req_mm"] * 0.4, irrigation["water_req_mm"] * 0.7], 'color': "orange"},
            {'range': [irrigation["water_req_mm"] * 0.7, irrigation["water_req_mm"]], 'color': "red"},
        ]
    }
))

st.plotly_chart(fig2, use_container_width=True)

# Smart message
if irrigation["deficit_mm"] == 0:
    st.success("🌧️ Rainfall is sufficient — no irrigation needed!")
else:
    st.info(
        f"To grow {crop} on {field_ha} ha, you need "
        f"{irrigation['pump_hours']} pump hours costing ₹{irrigation['cost']}."
    )

# PMKSY Scheme
if irrigation["deficit_mm"] > 200:
    st.warning(
        "⚠️ High irrigation need. You may qualify for PMKSY subsidy (up to 55% for drip irrigation)."
    )
else:
        st.markdown("""
        <div style="background:#F4F9F7;border-radius:12px;padding:40px;
                    text-align:center;border:2px dashed #1D9E75;margin-top:20px;">
          <p style="font-size:36px;margin:0 0 8px">🧪</p>
          <p style="font-size:15px;color:#0A1628;font-weight:500;margin:0 0 4px">
            Enter your Soil Health Card values
          </p>
          <p style="font-size:12px;color:#64748B;margin:0">
            AgriSense will calculate the exact fertilizer bags you need and
            show how much you save vs average application
          </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Source: ICAR Crop Production Guide · State Agriculture Department advisories · "
    "Soil nutrient availability factors from ICAR Soil Health Management guidelines. "
    "Fertilizer prices are indicative — verify with your local dealer."
)
