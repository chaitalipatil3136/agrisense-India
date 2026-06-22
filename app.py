"""
AgriSense India — Main Streamlit App Entry Point
File: app.py

Run: streamlit run app.py
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AgriSense India",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
# LOAD CSS THEME — safe, will not crash if file missing
# ════════════════════════════════════════════════════════════

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ style.css not found at: {css_path}")

load_css()

# Google Fonts — loaded separately to guarantee it works
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1'
    '&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;'
    '0,9..40,600;1,9..40,300&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌾 AgriSense India")
    st.markdown("*Intelligent Crop Advisory System*")
    st.markdown("---")

    st.markdown("### Navigate")
    st.markdown(
        "- 🌱 **Crop Advisor** — Recommend crops + earnings\n"
        "- 🍃 **Disease Detector** — Upload leaf photo\n"
        "- ♻️ **Carbon Footprint** — CO₂ per crop\n"
        "- 🗓️ **Rotation Planner** — 3-season calendar\n"
        "- 🗺️ **India Crop Map** — Interactive choropleth\n"
        "- 📊 **Model Insights** — Accuracy + data sources"
    )

    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown(
        "- [data.gov.in](https://data.gov.in) — Crop production\n"
        "- [Agmarknet](https://agmarknet.gov.in) — Mandi prices\n"
        "- [OpenWeatherMap](https://openweathermap.org) — Live weather\n"
        "- [PlantVillage](https://kaggle.com) — Disease images\n"
        "- [IPCC 2006](https://ipcc.ch) — Carbon factors\n"
        "- [ICAR](https://icar.org.in) — Rotation guidelines"
    )

    st.markdown("---")
    st.caption("Built by · MIT CSN Nagpur · 2nd Year CS")
    st.caption("GitHub · [agrisense-india](https://github.com)")


# ════════════════════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════════════════════

st.markdown(
    '<div class="agri-hero">'
    '<h1>🌾 AgriSense India</h1>'
    '<p>Intelligent Crop Advisory System for Indian Farmers — '
    'AI-powered recommendations, disease detection, market prices, '
    'and government schemes in one place.</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════
# TOP METRICS — 4 stat cards
# ════════════════════════════════════════════════════════════

col1, col2, col3, col4 = st.columns(4)

stat_data = [
    ("ML Accuracy",        "99.3%", "Random Forest"),
    ("Disease Classes",    "38",    "PlantVillage CNN"),
    ("Live Data Sources",  "3",     "APIs connected"),
    ("Crops Covered",      "22+",   "Recommendation model"),
]

for col, (label, value, sub) in zip([col1, col2, col3, col4], stat_data):
    with col:
        st.markdown(
            f'<div class="agri-stat">'
            f'<p class="agri-stat-label">{label}</p>'
            f'<p class="agri-stat-value">{value}</p>'
            f'<p class="agri-stat-sub">{sub}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SECTION HEADER — Features
# ════════════════════════════════════════════════════════════

st.markdown(
    '<div class="agri-section">'
    '<span class="agri-section-icon">🧩</span>'
    '<div>'
    '<p class="agri-section-title">14 Features in One Platform</p>'
    '<p class="agri-section-sub">Everything an Indian farmer needs — built on real government data</p>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════
# FEATURE CARDS — 3 per row
# ════════════════════════════════════════════════════════════

features = [
    ("🌱", "Crop Recommendation",  "Random Forest + XGBoost · 99.3% accuracy · Top 3 crops ranked by confidence"),
    ("📈", "Yield + Earnings",     "Predict kg/hectare · Live mandi prices from Agmarknet · Gross earnings estimate"),
    ("🔬", "SHAP Explainability",  "Why was this crop recommended? · Waterfall chart per prediction · Feature importance"),
    ("🍃", "Disease Detector",     "Upload a leaf photo · CNN detects 38 diseases · Treatment + prevention guidance"),
    ("♻️", "Carbon Footprint",     "IPCC Tier 1 emission factors · Compare crops by CO₂ impact · Sustainability rating"),
    ("🗓️", "Rotation Planner",     "3-season crop calendar · ICAR soil science · Nitrogen balance tracking"),
    ("🌧️", "Weather Alerts",       "5-day forecast · Risk alerts · Optimal sowing window calendar"),
    ("🏛️", "Govt Schemes",         "PM-KISAN eligibility · PMFBY crop insurance · KCC and more"),
    ("🗺️", "India Crop Map",       "Interactive choropleth · State-wise dominant crops · Hover tooltips"),
    ("📈", "Price Forecaster",     "30-day mandi price prediction · Facebook Prophet · Sell now or wait recommendation"),
    ("🧪", "Fertilizer Optimizer", "Exact bags to buy · Savings vs average farmer · ICAR NPK calculator"),
    ("🛰️", "Satellite Monitor",    "Real NASA/ERA5 climate data · Crop Health Index · District stress detection"),
    ("🐛", "Pest Predictor",       "Proactive pest alerts · ICAR-NCIPM calendar · Prevention before damage"),
    ("💧", "Irrigation Calculator","Exact water budget · Pump hours · FAO Paper 56 science"),
]

# Render 3 cards per row
for i in range(0, len(features), 3):
    row_features = features[i:i+3]
    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, row_features):
        with col:
            st.markdown(
                f'<div class="agri-stat" style="margin-bottom:12px;min-height:148px;">'
                f'<p style="font-size:24px;margin:0 0 6px">{icon}</p>'
                f'<p style="font-size:14px;font-weight:600;color:#1A1A14;margin:0 0 6px">{title}</p>'
                f'<p style="font-size:12px;color:#6B6B55;margin:0;line-height:1.5">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<p style="text-align:center;font-size:13px;color:#6B6B55;">'
    'Use the sidebar to explore all features.'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align:center;font-size:11px;color:#A8A890;">'
    'Data verified against official government sources — '
    'ICAR, data.gov.in, Agmarknet, IPCC 2006.'
    '</p>',
    unsafe_allow_html=True,
)