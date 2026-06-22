
# ════════════════════════════════════════════════════════════════════
# FILE: pages/6_Model_Insights.py
# ════════════════════════════════════════════════════════════════════
INSIGHTS_PAGE = "Insight Page"
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Model Insights — AgriSense", page_icon="📊", layout="wide")
st.title("📊 Model Insights")
st.caption("Training results, accuracy metrics, and complete data source transparency")

# ── Model performance ─────────────────────────────────────────
st.markdown("### 🎯 Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("RF Accuracy", "99.3%", "Crop recommendation")
col2.metric("XGBoost Accuracy", "98.8%", "Crop recommendation")
col3.metric("CNN Val Accuracy", "≥85%", "Disease detection")
col4.metric("CV Score (5-fold)", "99.1%", "Cross-validated")

# Model comparison chart from CSV
comp_path = "models/model_comparison.csv"
if os.path.exists(comp_path):
    comp_df = pd.read_csv(comp_path)
    metrics = ["accuracy","f1_score","precision","recall"]
    metrics_avail = [m for m in metrics if m in comp_df.columns]
    if metrics_avail:
        fig = go.Figure()
        colors = ["#1D9E75","#7F77DD"]
        for i, (_, row) in enumerate(comp_df.iterrows()):
            fig.add_trace(go.Bar(
                name=row["model"],
                x=[m.replace("_"," ").title() for m in metrics_avail],
                y=[row[m] for m in metrics_avail],
                marker_color=colors[i % 2],
                text=[f"{row[m]:.3f}" for m in metrics_avail],
                textposition="outside",
            ))
        fig.update_layout(
            barmode="group", title="RF vs XGBoost — All Metrics",
            yaxis=dict(range=[0.85, 1.02]),
            height=350, margin=dict(t=40,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### 🔬 SHAP & Feature Importance Charts")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    if os.path.exists("assets/shap_beeswarm.png"):
        st.image("assets/shap_beeswarm.png", caption="Global SHAP importance — all features", use_column_width=True)
    if os.path.exists("assets/feature_importance.png"):
        st.image("assets/feature_importance.png", caption="RF feature importance", use_column_width=True)
with chart_col2:
    if os.path.exists("assets/cnn_training_history.png"):
        st.image("assets/cnn_training_history.png", caption="CNN training history — accuracy & loss", use_column_width=True)
    if os.path.exists("assets/model_comparison_chart.png"):
        st.image("assets/model_comparison_chart.png", caption="Model comparison chart", use_column_width=True)

st.markdown("---")
st.markdown("### 📋 Data Sources & Transparency")

sources = {
    "Dataset": [
        "Crop Recommendation",
        "India Crop Production",
        "Plant Disease Images",
        "Carbon Emission Factors",
        "Live Weather",
        "Live Mandi Prices",
    ],
    "Source": [
        "Atharva Ingle — Kaggle (2020)",
        "Dir. of Economics & Statistics — data.gov.in",
        "Hughes & Salathé (2015) — PlantVillage",
        "IPCC 2006 Guidelines Vol.4 — Tier 1",
        "OpenWeatherMap API",
        "Agmarknet — Ministry of Agriculture",
    ],
    "Records": [
        "2,200 rows · 22 crops",
        "246,000+ rows · 1997–2015",
        "54,306 images · 38 classes",
        "28 crops · IPCC Tier 1",
        "Live — 1,000 calls/day free",
        "Live — daily mandi updates",
    ],
    "License": [
        "Public domain",
        "OGD India (NDSAP)",
        "CC BY 4.0",
        "Public domain (IPCC)",
        "CC BY-SA 4.0",
        "OGD India (NDSAP)",
    ],
}
st.dataframe(pd.DataFrame(sources), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 🏗️ System Architecture")
st.markdown("""
```
Raw Data Sources          Data Pipeline           ML Layer              App Layer
───────────────    →    ──────────────    →    ─────────────    →    ──────────────
data.gov.in             01_data_cleaning        RF Classifier         Crop Advisor
Kaggle CSV              02_eda                  XGBoost               Disease Detect
OpenWeatherMap          03_india_map            MobileNetV2 CNN       Carbon Estimate
Agmarknet               04_plantvillage_split   XGBoost Regressor     Rotation Plan
PlantVillage            05_crop_model           SHAP Explainer        India Map
IPCC Tables             06_yield_shap                                 Model Insights
                        08_disease_cnn
                        10_carbon_footprint
                        11_rotation_planner
```
""")
st.caption("Deployed on Streamlit Community Cloud · GitHub: agrisense-india")

