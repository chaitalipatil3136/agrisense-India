"""
AgriSense India — Pages 3-6 (separate files, combined here for download)
Copy each section into its own file in your pages/ folder.

pages/3_Carbon_Footprint.py
pages/4_Rotation_Planner.py
pages/5_India_Map.py
pages/6_Model_Insights.py
"""

# ════════════════════════════════════════════════════════════════════
# FILE: pages/3_Carbon_Footprint.py
# Copy everything between the markers into pages/3_Carbon_Footprint.py
# ════════════════════════════════════════════════════════════════════
CARBON_PAGE = "Carbon Footprint"

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os, sys
from utils.database import init_db, save_carbon_log
init_db()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Carbon Footprint — AgriSense", page_icon="♻️", layout="wide")
st.title("♻️ Carbon Footprint Estimator")
st.caption("Estimate CO₂ equivalent emissions per crop cycle — based on IPCC 2006 Tier 1 factors")

CROPS = ["cotton","rice","wheat","maize","soybean","sugarcane","groundnut",
         "pigeonpea","chickpea","mungbean","blackgram","lentil","sunflower","mustard"]

col1, col2 = st.columns([1, 1])
with col1:
    crop = st.selectbox("Crop", options=CROPS, format_func=str.capitalize)
    field_ha = st.slider("Field size (hectares)", 0.5, 20.0, 2.0, step=0.5)
    n_kg = st.slider("Nitrogen fertilizer (kg/ha)", 0, 200, 100)
    p_kg = st.slider("Phosphorus fertilizer (kg/ha)", 0, 100, 50)
    k_kg = st.slider("Potassium fertilizer (kg/ha)", 0, 100, 50)
    calc_btn = st.button("Calculate Carbon Footprint", type="primary", use_container_width=True)

with col2:
    if calc_btn:
        try:
            from utils.carbon import calculate_carbon, get_carbon_comparison, get_sustainability_rating
            result  = calculate_carbon(crop, field_ha, n_kg, p_kg, k_kg)
            rating, color = get_sustainability_rating(result["total_kgco2e_per_ha"])

            st.markdown(f"""
            <div style="background:#0A1628;border-radius:12px;padding:18px 22px;
                        border-left:4px solid {color};margin-bottom:14px;">
              <p style="color:#9BBFA0;font-size:12px;margin:0">Total emissions</p>
              <p style="color:#FFFFFF;font-size:32px;font-weight:700;margin:0 0 4px">
                {result["total_kgco2e"]:,.0f} <span style="font-size:16px;font-weight:400">kg CO₂e</span>
              </p>
              <p style="color:#9BBFA0;font-size:13px;margin:0 0 8px">
                {result["total_kgco2e_per_ha"]:.0f} kg CO₂e/ha · Equivalent to {result["equivalent_car_km"]:,.0f} km by car
              </p>
              <span style="background:{color};color:white;padding:3px 12px;
                           border-radius:99px;font-size:12px;font-weight:500;">
                {rating}
              </span>
            </div>
            """, unsafe_allow_html=True)

            # Breakdown donut chart
            categories = ["N fertilizer", "P fertilizer", "K fertilizer", "Land use", "Sequestration"]
            values = [
                max(0, result["n_emission"]),
                max(0, result["p_emission"]),
                max(0, result["k_emission"]),
                max(0, result["land_emission"]),
                abs(min(0, result["sequestration"])),
            ]
            colors = ["#D85A30","#EF9F27","#7F77DD","#888780","#1D9E75"]
            fig = go.Figure(data=[go.Pie(
                labels=categories, values=values,
                hole=0.45,
                marker=dict(colors=colors),
                textinfo="label+percent",
            )])
            fig.update_layout(
                title="Emission sources breakdown",
                showlegend=False,
                height=280,
                margin=dict(t=40,b=0,l=0,r=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            # ✅ SAVE CARBON LOG (correct placement)
            save_carbon_log({
            "crop": crop,
            "field_ha": field_ha,
            "n_applied": n_kg,
            "p_applied": p_kg,
            "k_applied": k_kg,
            "total_kgco2e": result["total_kgco2e"],
            "sustainability": rating
            })

            st.success("✅ Carbon footprint saved to database")

            # Comparison bar chart
            st.markdown("**How does this compare to alternatives?**")
            compare = get_carbon_comparison(crop, field_ha, n_kg, p_kg, k_kg)
            comp_df = pd.DataFrame(compare)[["crop","total_kgco2e","is_selected"]]
            comp_df = comp_df.sort_values("total_kgco2e")
            bar_colors = ["#1D9E75" if r else "#C8D5CF" for r in comp_df["is_selected"]]
            fig2 = px.bar(
                comp_df, x="total_kgco2e", y="crop",
                orientation="h",
                color_discrete_sequence=bar_colors,
                labels={"total_kgco2e":"Total CO₂e (kg)","crop":"Crop"},
                title="Carbon footprint comparison",
            )
            fig2.update_layout(height=300, margin=dict(t=40,b=0,l=0,r=0), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        except ImportError:
            st.error("utils/carbon.py not found. Run notebooks/10_carbon_footprint.py first.")
        except Exception as e:
            st.error(f"Calculation error: {e}")
    else:
        st.info("👈 Enter crop details and click Calculate to see CO₂ footprint")


st.markdown("---")
st.caption("Source: IPCC 2006 Guidelines for National GHG Inventories, Volume 4 — Agriculture. Tier 1 default emission factors.")

