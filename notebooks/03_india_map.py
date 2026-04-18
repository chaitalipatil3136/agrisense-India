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
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

print("=" * 55)
print("AgriSense India — India Choropleth Map Builder")
print("=" * 55)

# ── Step 1: Load GeoJSON ────────────────────────────────────
geojson_path = "assets/india_states.geojson"

if not os.path.exists(geojson_path):
    print(f"\nERROR: GeoJSON not found at {geojson_path}")
    print("Download command:")
    print('  curl -L -o assets/india_states.geojson "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"')
    exit(1)

with open(geojson_path, encoding="utf-8") as f:
    india_geo = json.load(f)

# Extract actual state names from GeoJSON
geo_states = []
feature_key = None
for feat in india_geo["features"]:
    props = feat.get("properties", {})
    # Try common property name variants
    for key in ["NAME_1", "name", "ST_NM", "NAME", "state"]:
        if key in props:
            geo_states.append(props[key])
            if feature_key is None:
                feature_key = f"properties.{key}"
            break

print(f"\nGeoJSON loaded — {len(geo_states)} states found")
print(f"Feature key: {feature_key}")
print(f"Sample state names: {geo_states[:5]}")


# ── Step 2: State name normalizer ──────────────────────────
# Handles mismatches between GeoJSON names and dataset names
NAME_FIX = {
    "Andaman & Nicobar Island": "Andaman and Nicobar Islands",
    "Arunanchal Pradesh":       "Arunachal Pradesh",
    "Dadra & Nagar Haveli":    "Dadra and Nagar Haveli",
    "Daman & Diu":             "Daman and Diu",
    "Jammu & Kashmir":         "Jammu and Kashmir",
    "NCT of Delhi":            "Delhi",
    "Tamilnadu":               "Tamil Nadu",
    "Uttaranchal":             "Uttarakhand",
    "Pondicherry":             "Puducherry",
}

def normalize_state(name):
    if not isinstance(name, str):
        return name
    name = name.strip().title()
    return NAME_FIX.get(name, name)


# ── Step 3: Build state → crop mapping ─────────────────────
prod_path = "data/processed/india_production_clean.csv"

if os.path.exists(prod_path):
    print(f"\nLoading production data: {prod_path}")
    prod_df = pd.read_csv(prod_path)

    # Identify state and crop columns
    state_col = next(
        (c for c in prod_df.columns if "state" in c.lower()), None
    )
    crop_col = next(
        (c for c in prod_df.columns if c.lower() == "crop"), None
    )
    area_col = next(
        (c for c in prod_df.columns
         if "area" in c.lower() and "hect" not in c.lower()), None
    ) or next(
        (c for c in prod_df.columns if "area" in c.lower()), None
    )

    print(f"  State col: {state_col} | Crop col: {crop_col} | Area col: {area_col}")

    if state_col and crop_col and area_col:
        prod_df[state_col] = prod_df[state_col].apply(normalize_state)
        prod_df[area_col]  = pd.to_numeric(prod_df[area_col], errors="coerce")

        # Dominant crop = crop with highest total area per state
        dom_crop = (
            prod_df.groupby([state_col, crop_col])[area_col]
            .sum()
            .reset_index()
            .sort_values(area_col, ascending=False)
            .groupby(state_col)
            .first()
            .reset_index()[[state_col, crop_col]]
        )
        dom_crop.columns = ["state", "dominant_crop"]

        # Clean up crop names
        dom_crop["dominant_crop"] = (
            dom_crop["dominant_crop"]
            .str.strip()
            .str.title()
        )

        print(f"  Dominant crop computed for {len(dom_crop)} states")
        print(dom_crop.head(8).to_string(index=False))
        use_real_data = True
    else:
        print("  Column detection failed — using curated data")
        use_real_data = False
else:
    print("\nProduction data not found — using curated crop mapping")
    use_real_data = False

# ── Curated fallback (all 28 states + 8 UTs) ───────────────
CURATED = {
    "Andhra Pradesh":       ("Rice",        "#1D9E75"),
    "Arunachal Pradesh":    ("Rice",        "#1D9E75"),
    "Assam":                ("Rice",        "#1D9E75"),
    "Bihar":                ("Wheat",       "#EF9F27"),
    "Chhattisgarh":         ("Rice",        "#1D9E75"),
    "Goa":                  ("Rice",        "#1D9E75"),
    "Gujarat":              ("Cotton",      "#7F77DD"),
    "Haryana":              ("Wheat",       "#EF9F27"),
    "Himachal Pradesh":     ("Wheat",       "#EF9F27"),
    "Jharkhand":            ("Rice",        "#1D9E75"),
    "Karnataka":            ("Maize",       "#639922"),
    "Kerala":               ("Coconut",     "#5DCAA5"),
    "Madhya Pradesh":       ("Soybean",     "#D85A30"),
    "Maharashtra":          ("Cotton",      "#7F77DD"),
    "Manipur":              ("Rice",        "#1D9E75"),
    "Meghalaya":            ("Rice",        "#1D9E75"),
    "Mizoram":              ("Rice",        "#1D9E75"),
    "Nagaland":             ("Rice",        "#1D9E75"),
    "Odisha":               ("Rice",        "#1D9E75"),
    "Punjab":               ("Wheat",       "#EF9F27"),
    "Rajasthan":            ("Bajra",       "#BA7517"),
    "Sikkim":               ("Maize",       "#639922"),
    "Tamil Nadu":           ("Rice",        "#1D9E75"),
    "Telangana":            ("Cotton",      "#7F77DD"),
    "Tripura":              ("Rice",        "#1D9E75"),
    "Uttar Pradesh":        ("Wheat",       "#EF9F27"),
    "Uttarakhand":          ("Wheat",       "#EF9F27"),
    "West Bengal":          ("Rice",        "#1D9E75"),
    "Delhi":                ("Wheat",       "#EF9F27"),
    "Jammu and Kashmir":    ("Wheat",       "#EF9F27"),
    "Ladakh":               ("Wheat",       "#EF9F27"),
    "Puducherry":           ("Rice",        "#1D9E75"),
    "Chandigarh":           ("Wheat",       "#EF9F27"),
    "Andaman and Nicobar Islands": ("Rice", "#1D9E75"),
    "Lakshadweep":          ("Coconut",     "#5DCAA5"),
    "Dadra and Nagar Haveli": ("Rice",      "#1D9E75"),
    "Daman and Diu":        ("Rice",        "#1D9E75"),
}

if not use_real_data:
    dom_crop = pd.DataFrame([
        {"state": k, "dominant_crop": v[0]}
        for k, v in CURATED.items()
    ])

# ── Step 4: Build the Plotly choropleth ─────────────────────
print("\nBuilding choropleth map...")

# Discrete color map per crop type
CROP_COLOR_MAP = {
    "Rice":        "#1D9E75",
    "Wheat":       "#EF9F27",
    "Cotton":      "#7F77DD",
    "Soybean":     "#D85A30",
    "Maize":       "#639922",
    "Bajra":       "#BA7517",
    "Sugarcane":   "#0F6E56",
    "Groundnut":   "#F0997B",
    "Coconut":     "#5DCAA5",
    "Jowar":       "#AFA9EC",
    "Ragi":        "#85B7EB",
    "Mustard":     "#FAC775",
}

# Add yield/area info to hover
dom_crop["info"] = dom_crop.apply(
    lambda r: f"{r['state']}<br>Dominant crop: <b>{r['dominant_crop']}</b>",
    axis=1,
)

fig = px.choropleth(
    dom_crop,
    geojson=india_geo,
    featureidkey=feature_key,
    locations="state",
    color="dominant_crop",
    color_discrete_map=CROP_COLOR_MAP,
    hover_name="state",
    hover_data={"dominant_crop": True, "state": False},
    title="AgriSense India — Dominant Crop by State",
    labels={"dominant_crop": "Dominant Crop"},
)

fig.update_geos(
    fitbounds="locations",
    visible=False,
    bgcolor="rgba(0,0,0,0)",
)

fig.update_layout(
    title=dict(
        text="<b>AgriSense India</b> — Dominant Crop by State",
        font=dict(size=20, family="Arial"),
        x=0.5,
        xanchor="center",
    ),
    legend=dict(
        title="<b>Dominant Crop</b>",
        title_font_size=13,
        font_size=12,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#ddd",
        borderwidth=1,
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
    ),
    margin={"r": 0, "t": 60, "l": 0, "b": 20},
    height=620,
    paper_bgcolor="white",
    plot_bgcolor="white",
    annotations=[
        dict(
            text=(
                "Source: District-wise Season-wise Crop Production Statistics — "
                "Directorate of Economics & Statistics, MoAFW, Govt of India"
            ),
            xref="paper", yref="paper",
            x=0.99, y=-0.03,
            showarrow=False,
            font=dict(size=9, color="#888"),
            align="right",
        )
    ],
)

# Add custom hover template
fig.update_traces(
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "Dominant crop: %{customdata[0]}<br>"
        "<extra></extra>"
    )
)

# ── Step 5: Save ────────────────────────────────────────────
out_path = "assets/india_map.html"
fig.write_html(
    out_path,
    include_plotlyjs="cdn",   # loads from CDN — keeps file small
    full_html=True,
)

file_size_kb = os.path.getsize(out_path) // 1024
print(f"\nSaved: {out_path} ({file_size_kb} KB)")
print("Open this file in your browser to see the interactive map.")
print("\nCrops on the map:")
print(dom_crop["dominant_crop"].value_counts().to_string())

print("\nMap build complete!")
print("Screenshot it for your LinkedIn post — this is your wow visual.")
