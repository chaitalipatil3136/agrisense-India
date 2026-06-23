
# ════════════════════════════════════════════════════════════════════
# FILE: pages/5_India_Map.py
# ════════════════════════════════════════════════════════════════════
MAP_PAGE = "Map"
import streamlit as st
"""
AgriSense India — Interactive India Choropleth Map
File: notebooks/03_india_map.py

Builds an interactive Plotly HTML map showing dominant crop per state.
Open the output file in any browser — no internet needed.

Input:  assets/india_states.geojson
        data/processed/india_production_clean.csv (if available)
Output: assets/india_map.html  (interactive, ~2MB)

Run: python notebooks/03_india_map.py
"""

import json
import json
import plotly.express as px
import pandas as pd
import os
geojson_path = "assets/india_states.geojson"

with open(geojson_path, "r", encoding="utf-8") as f:
    india_geo = json.load(f)
st.set_page_config(page_title="India Crop Map — AgriSense", page_icon="🗺️", layout="wide")
st.title("🗺️ India Crop Map")

st.caption("Interactive map showing dominant crop by state — hover over any state for details")


st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Top Kharif States:**")
    st.markdown("- Maharashtra → Cotton\\n- Punjab → Rice\\n- West Bengal → Rice")
with col2:
    st.markdown("**Top Rabi States:**")
    st.markdown("- Punjab → Wheat\\n- UP → Wheat\\n- MP → Wheat")
with col3:
    st.markdown("**Top Annual Crops:**")
    st.markdown("- Maharashtra → Sugarcane\\n- UP → Sugarcane\\n- Karnataka → Sugarcane")

