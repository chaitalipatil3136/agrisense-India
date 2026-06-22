
# ════════════════════════════════════════════════════════════════════
# FILE: pages/5_India_Map.py
# ════════════════════════════════════════════════════════════════════
MAP_PAGE = "Map"
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="India Crop Map — AgriSense", page_icon="🗺️", layout="wide")
st.title("🗺️ India Crop Map")
st.caption("Interactive map showing dominant crop by state — hover over any state for details")

map_path = "assets/india_map.html"

if os.path.exists(map_path):
    with open(map_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=620, scrolling=False)
    st.caption("Source: District-wise Season-wise Crop Production Statistics — Directorate of Economics & Statistics, MoAFW, Govt of India (data.gov.in)")
else:
    st.error("India map not found at assets/india_map.html")
    st.info("Run: python notebooks/03_india_map.py")

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

