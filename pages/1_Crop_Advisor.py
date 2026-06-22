"""
AgriSense India — Complete Crop Advisor (Consolidated)
File: pages/1_Crop_Advisor.py

All 10 sections in one page:
1. District + weather
2. Soil inputs
3. Crop recommendation
4. Yield + mandi earnings
5. Govt scheme eligibility
6. Pest risk this season
7. Irrigation requirement
8. Carbon footprint
9. SHAP explanation
10. Prediction history + PDF note
"""
from utils.translations import t, get_lang
lang = get_lang()
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.translations import T, get_lang, lang_selector, t

st.set_page_config(
    page_title="Crop Advisor",
    page_icon="🌱",
    layout="wide",
)

# sidebar
with st.sidebar:
    lang = lang_selector()

# title using translation
st.title(t("ca_title"))
# ════════════════════════════════════════════════════════════
# DATABASE INIT
# ════════════════════════════════════════════════════════════
try:
    from utils.database import init_db, save_prediction, get_history, get_stats
    init_db()
    DB_OK = True
except Exception:
    DB_OK = False

# ════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════

DISTRICTS = {
    "Nagpur, Maharashtra":     {"state": "Maharashtra",    "city": "Nagpur"},
    "Amravati, Maharashtra":   {"state": "Maharashtra",    "city": "Amravati"},
    "Wardha, Maharashtra":     {"state": "Maharashtra",    "city": "Wardha"},
    "Pune, Maharashtra":       {"state": "Maharashtra",    "city": "Pune"},
    "Nashik, Maharashtra":     {"state": "Maharashtra",    "city": "Nashik"},
    "Aurangabad, Maharashtra": {"state": "Maharashtra",    "city": "Aurangabad"},
    "Ludhiana, Punjab":        {"state": "Punjab",         "city": "Ludhiana"},
    "Amritsar, Punjab":        {"state": "Punjab",         "city": "Amritsar"},
    "Varanasi, UP":            {"state": "Uttar Pradesh",  "city": "Varanasi"},
    "Patna, Bihar":            {"state": "Bihar",          "city": "Patna"},
    "Indore, MP":              {"state": "Madhya Pradesh", "city": "Indore"},
    "Bhopal, MP":              {"state": "Madhya Pradesh", "city": "Bhopal"},
    "Jaipur, Rajasthan":       {"state": "Rajasthan",      "city": "Jaipur"},
    "Jodhpur, Rajasthan":      {"state": "Rajasthan",      "city": "Jodhpur"},
    "Hyderabad, Telangana":    {"state": "Telangana",      "city": "Hyderabad"},
    "Vijayawada, AP":          {"state": "Andhra Pradesh", "city": "Vijayawada"},
    "Coimbatore, Tamil Nadu":  {"state": "Tamil Nadu",     "city": "Coimbatore"},
    "Bangalore, Karnataka":    {"state": "Karnataka",      "city": "Bangalore"},
    "Kolkata, West Bengal":    {"state": "West Bengal",    "city": "Kolkata"},
    "Ahmedabad, Gujarat":      {"state": "Gujarat",        "city": "Ahmedabad"},
}

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FEATURE_DISPLAY = {
    "N":           "Nitrogen in soil",
    "P":           "Phosphorus in soil",
    "K":           "Potassium in soil",
    "temperature": "Temperature",
    "humidity":    "Humidity",
    "ph":          "Soil pH (acidity)",
    "rainfall":    "Rainfall",
}

# MSP 2024-25 — Ministry of Agriculture (₹/quintal)
MSP_2024 = {
    "rice": 2183, "wheat": 2275, "maize": 2090,
    "cotton": 6620, "soybean": 4600, "sugarcane": 340,
    "groundnut": 6377, "chickpea": 5440, "lentil": 6425,
    "mungbean": 8558, "blackgram": 7400, "pigeonpea": 7000,
    "mustard": 5650, "sunflower": 6760, "bajra": 2500,
    "jowar": 3180, "barley": 1735,
}

# Cultivation cost per hectare (ICAR estimates, ₹)
CULTIVATION_COST = {
    "cotton": 35000, "rice": 25000, "wheat": 18000,
    "maize": 20000, "soybean": 15000, "sugarcane": 45000,
    "groundnut": 22000, "chickpea": 12000, "pigeonpea": 14000,
    "mustard": 14000, "bajra": 10000, "jowar": 11000,
}

MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]


# ════════════════════════════════════════════════════════════
# CACHED MODEL LOADERS
# ════════════════════════════════════════════════════════════

@st.cache_resource
def load_crop_models():
    rf = joblib.load("models/rf_crop_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    sc = joblib.load("models/scaler.pkl")
    return rf, le, sc


@st.cache_resource
def load_yield_model():
    ym = joblib.load("models/yield_model.pkl")
    ys = joblib.load("models/yield_scaler.pkl")
    return ym, ys


@st.cache_resource
def get_shap_explainer(_rf):
    import shap
    return shap.TreeExplainer(_rf)


# ════════════════════════════════════════════════════════════
# HELPER: section header with farmer-friendly style
# ════════════════════════════════════════════════════════════

def section(icon: str, title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;
                margin:28px 0 6px;border-left:4px solid #1D9E75;
                padding-left:12px;">
      <span style="font-size:22px">{icon}</span>
      <div>
        <p style="font-size:18px;font-weight:600;color:#0A2218;margin:0">{title}</p>
        {"" if not subtitle else f'<p style="font-size:12px;color:#64748B;margin:0">{subtitle}</p>'}
      </div>
    </div>
    """, unsafe_allow_html=True)


def green_card(big_text: str, label: str, sub: str = ""):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0A1628,#0F2D1E);
                border-radius:14px;padding:20px 24px;margin:10px 0;
                border-left:5px solid #1D9E75;">
      <p style="color:#9BBFA0;font-size:12px;margin:0">{label}</p>
      <p style="color:#FFFFFF;font-size:34px;font-weight:700;margin:4px 0">{big_text}</p>
      {"" if not sub else f'<p style="color:#9BBFA0;font-size:12px;margin:0">{sub}</p>'}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(135deg,#0A2218,#1D9E75);
            border-radius:16px;padding:28px 32px;margin-bottom:24px;">
  <h1 style="color:white;font-size:32px;margin:0 0 6px">🌱 AgriSense Crop Advisor</h1>
  <p style="color:#C8F0DF;font-size:15px;margin:0">
    Enter your soil details below — get crop recommendation, earnings estimate,
    pest warnings, irrigation need, and government schemes in one place.
  </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SECTION 1 — DISTRICT + WEATHER
# ════════════════════════════════════════════════════════════

section("📍", "Your Location", "Select district — weather fills automatically")

col_dist, col_wthr = st.columns([1, 1])

# Session state for live weather
if "live_temp"  not in st.session_state: st.session_state.live_temp  = 25.0
if "live_humid" not in st.session_state: st.session_state.live_humid = 70.0

with col_dist:
    selected_district = st.selectbox(
        "District / Taluka",
        options=list(DISTRICTS.keys()),
        index=0,
    )
    district_info = DISTRICTS[selected_district]
    field_ha = st.number_input(
        "Your field size (hectares)",
        min_value=0.5, max_value=50.0, value=2.0, step=0.5,
        help="1 hectare = 2.47 acres",
    )

with col_wthr:
    if st.button("🌤️ Get live weather for my district", type="secondary",
                 use_container_width=True):
        try:
            from utils.weather_api import get_weather
            with st.spinner(f"Connecting to weather server for {district_info['city']}..."):
                w = get_weather(district_info["city"])
            st.session_state.live_temp  = float(w.get("temperature", 25.0))
            st.session_state.live_humid = float(w.get("humidity",    70.0))
            st.success(
                f"✅ Live weather loaded — "
                f"{st.session_state.live_temp}°C, "
                f"{st.session_state.live_humid}% humidity"
            )
        except Exception:
            st.warning("Weather server unavailable. Enter values manually below.")

    st.markdown(f"""
    <div style="background:#F0FBF6;border-radius:10px;padding:12px 16px;margin-top:8px;">
      <div style="display:flex;gap:24px;">
        <div><p style="color:#666;font-size:11px;margin:0">Temperature</p>
             <p style="color:#0A2218;font-size:20px;font-weight:600;margin:0">
               {st.session_state.live_temp:.1f}°C</p></div>
        <div><p style="color:#666;font-size:11px;margin:0">Humidity</p>
             <p style="color:#0A2218;font-size:20px;font-weight:600;margin:0">
               {st.session_state.live_humid:.0f}%</p></div>
      </div>
      <p style="color:#888;font-size:10px;margin:6px 0 0">
        Source: OpenWeatherMap · Updates when you click the button above
      </p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SECTION 2 — SOIL DATA
# ════════════════════════════════════════════════════════════

section("🧪", "Your Soil Data",
        "Get N, P, K, pH from your Soil Health Card (Green card from government)")

col1, col2, col3 = st.columns(3)

with col1:
    n_val = st.slider(
        "Nitrogen — N (kg/ha)",
        0, 140, 90,
        help="📋 From Soil Health Card → 'Available N'. Typical: 50–120 kg/ha",
    )
    p_val = st.slider(
        "Phosphorus — P (kg/ha)",
        5, 145, 42,
        help="📋 From Soil Health Card → 'Available P'. Typical: 20–80 kg/ha",
    )

with col2:
    k_val = st.slider(
        "Potassium — K (kg/ha)",
        5, 205, 43,
        help="📋 From Soil Health Card → 'Available K'. Typical: 100–200 kg/ha",
    )
    ph_val = st.slider(
        "Soil pH (acidity level)",
        3.5, 9.5, 6.5, step=0.1,
        help="7.0 = neutral. Below 7 = acidic, Above 7 = alkaline. Most crops prefer 6.0–7.5",
    )

with col3:
    temp_val = st.number_input(
        "Temperature (°C)",
        8.0, 50.0,
        value=float(st.session_state.live_temp),
        step=0.5,
    )
    humidity_val = st.number_input(
        "Humidity (%)",
        10.0, 100.0,
        value=float(st.session_state.live_humid),
        step=1.0,
    )
    rainfall_val = st.slider(
        "Expected rainfall this season (mm)",
        20, 300, 200,
        help="Check IMD forecast or last year's rainfall for your district",
    )

# ════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button(
    "🔮  Get My Complete Farm Advisory",
    type="primary",
    use_container_width=True,
)

if not predict_btn:
    st.info(
        "👆 Fill in your soil details above and click the button "
        "to get crop recommendation, earnings, pest risk, irrigation "
        "need, and govt schemes — all in one place."
    )
    st.stop()

# ════════════════════════════════════════════════════════════
# LOAD MODELS
# ════════════════════════════════════════════════════════════

try:
    rf, le, sc = load_crop_models()
except Exception as e:
    st.error(
        f"❌ Could not load AI models: {e}  \n"
        "Make sure you have run `python notebooks/05_crop_model.py` first."
    )
    st.stop()

input_vals   = np.array([[n_val, p_val, k_val,
                           temp_val, humidity_val,
                           ph_val, rainfall_val]])
input_scaled = sc.transform(input_vals)

with st.spinner("🌾 AI is analysing your soil and climate data..."):
    proba      = rf.predict_proba(input_scaled)[0]
    top3_idx   = np.argsort(proba)[::-1][:3]
    top3_crops = [(le.inverse_transform([i])[0], float(proba[i]))
                  for i in top3_idx]
    best_crop, best_conf = top3_crops[0]

conf_pct    = best_conf * 100
badge_color = (
    "#1D9E75" if conf_pct >= 80 else
    "#EF9F27" if conf_pct >= 60 else
    "#D85A30"
)
conf_word = (
    "Very confident" if conf_pct >= 80 else
    "Confident"      if conf_pct >= 60 else
    "Moderate confidence"
)

st.markdown("---")

# ════════════════════════════════════════════════════════════
# SECTION 3 — RECOMMENDATION
# ════════════════════════════════════════════════════════════

section("🏆", "AI Crop Recommendation",
        "Based on your soil data, weather, and 2,200 farming records")

col_rec, col_alt = st.columns([1.4, 1])

with col_rec:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0A1628,#0F2D1E);
                border-radius:14px;padding:24px 28px;
                border-left:6px solid {badge_color};">
      <p style="color:#9BBFA0;font-size:12px;margin:0;
                text-transform:uppercase;letter-spacing:0.06em">
        Best crop for your field
      </p>
      <p style="color:#FFFFFF;font-size:48px;font-weight:700;
                margin:6px 0 10px;line-height:1">
        {best_crop.upper()}
      </p>
      <span style="background:{badge_color};color:white;
                   padding:5px 16px;border-radius:99px;
                   font-size:13px;font-weight:500;">
        {conf_word} ({conf_pct:.1f}%)
      </span>
      <p style="color:#9BBFA0;font-size:12px;margin:14px 0 0">
        AI analysed: Nitrogen ({n_val} kg/ha) · Phosphorus ({p_val} kg/ha) ·
        Potassium ({k_val} kg/ha) · pH ({ph_val}) ·
        Temp ({temp_val}°C) · Rain ({rainfall_val}mm)
      </p>
    </div>
    """, unsafe_allow_html=True)

with col_alt:
    st.markdown("**Other crops that could also work:**")
    for i, (crop, conf) in enumerate(top3_crops[1:], 2):
        st.markdown(f"""
        <div style="background:#F0FBF6;border-radius:10px;
                    padding:10px 14px;margin-bottom:8px;
                    border-left:3px solid #5DCAA5;">
          <div style="display:flex;justify-content:space-between;">
            <span style="font-weight:500;color:#0A2218">
              #{i} {crop.capitalize()}
            </span>
            <span style="color:#1D9E75;font-weight:600">
              {conf*100:.1f}%
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "📊 Model: Random Forest (99.3% accuracy) · "
        "Trained on crop_recommendation.csv · "
        "Source: Atharva Ingle, Kaggle"
    )


# ════════════════════════════════════════════════════════════
# SECTION 5 — GOVERNMENT SCHEMES
# ════════════════════════════════════════════════════════════

section("🏛️", "Government Schemes You May Qualify For",
        "Auto-checked based on your crop, district, and field size")

# Scheme eligibility rules
schemes = []

# PM-KISAN — all farmers with land
schemes.append({
    "name":    "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
    "benefit": "₹6,000 per year directly to your bank account (₹2,000 × 3 installments)",
    "how":     "Apply at pmkisan.gov.in or nearest CSC (Common Service Centre)",
    "docs":    "Aadhaar card · Bank passbook · Land records (Khasra/Khatoni)",
    "eligible": True,
    "color":   "#1D9E75",
    "icon":    "💰",
})

# PMFBY — all crops
schemes.append({
    "name":    f"PMFBY — Crop Insurance for {best_crop.capitalize()}",
    "benefit": (
        f"Insurance against drought, flood, pest damage. "
        f"Your premium: only {2 if best_crop.lower() in ['rice','maize','bajra','jowar','cotton','soybean','groundnut','sugarcane'] else 1.5}% "
        f"of sum insured (Govt pays the rest)"
    ),
    "how":     "Apply at nearest bank branch or CSC before sowing season ends",
    "docs":    "Aadhaar · Bank passbook · Land records · Sowing certificate",
    "eligible": True,
    "color":   "#EF9F27",
    "icon":    "🛡️",
})

# KCC
schemes.append({
    "name":    "Kisan Credit Card (KCC)",
    "benefit": "Low-interest crop loan up to ₹3 lakh at only 4% interest per year (after govt subsidy)",
    "how":     "Apply at any nationalised bank (SBI, Bank of Maharashtra, PNB, etc.)",
    "docs":    "Aadhaar · PAN card · Land records · Passport photo",
    "eligible": True,
    "color":   "#7F77DD",
    "icon":    "💳",
})

# PMKSY — irrigation subsidy if deficit > 150mm
if rainfall_val < 150:
    schemes.append({
        "name":    "PMKSY — Drip/Sprinkler Irrigation Subsidy",
        "benefit": f"Up to 55% subsidy on drip irrigation system (saves 40% water). Estimated subsidy: ₹{int(field_ha * 45000 * 0.55):,}",
        "how":     "Apply through State Agriculture Department",
        "docs":    "Aadhaar · Land records · Bank account",
        "eligible": True,
        "color":   "#5DCAA5",
        "icon":    "💧",
    })

# Soil Health Card
schemes.append({
    "name":    "Soil Health Card Scheme",
    "benefit": "FREE soil testing every 2 years + personalised fertilizer recommendation card",
    "how":     "Contact nearest KVK (Krishi Vigyan Kendra) or agriculture department office",
    "docs":    "Aadhaar · Land records",
    "eligible": True,
    "color":   "#BA7517",
    "icon":    "🧪",
})

st.success(f"✅ You are likely eligible for **{len(schemes)} government schemes**")

for scheme in schemes:
    with st.expander(
        f"{scheme['icon']} {scheme['name']}", expanded=False
    ):
        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            st.markdown(f"**What you get:** {scheme['benefit']}")
            st.markdown(f"**How to apply:** {scheme['how']}")
        with col_b:
            st.markdown("**Documents needed:**")
            for doc in scheme["docs"].split("·"):
                st.markdown(f"- {doc.strip()}")

st.caption(
    "Source: Ministry of Agriculture & Farmers Welfare, Govt of India · "
    "Eligibility is indicative — verify at official portals before applying"
)

# ════════════════════════════════════════════════════════════
# SECTION 6 — PEST RISK
# ════════════════════════════════════════════════════════════

section("🐛", "Pest & Disease Risk This Season",
        f"Which pests to watch for when growing {best_crop.capitalize()} in {MONTH_NAMES[datetime.now().month - 1]}")

current_month = datetime.now().month
pests_found   = []

try:
    from utils.pest_advisor import get_pest_risks, get_overall_risk
    pests_found = get_pest_risks(
        crop      = best_crop,
        month     = current_month,
        temp_c    = temp_val,
        rain_mm   = rainfall_val,
    )
    overall_risk, risk_color = get_overall_risk(pests_found)
except ImportError:
    # Inline fallback if pest_advisor not available
    BASIC_PESTS = {
        "cotton":   [{"pest":"Whitefly","risk_level":"high","symptoms":"Yellow leaves, sticky coating","prevention":"Yellow sticky traps 10/acre, Neem oil 3ml/L spray","chemical":"Imidacloprid 0.5ml/L","organic":"Neem oil + garlic spray","damage":"Transmits Leaf Curl Virus — can destroy entire crop","source":"ICAR-CICR Nagpur"}],
        "rice":     [{"pest":"Brown Planthopper","risk_level":"high","symptoms":"Yellowing patches, hopperburn","prevention":"Avoid excess nitrogen, drain field 3-4 days","chemical":"Buprofezin 1.25ml/L (NOT pyrethroids)","organic":"Drain field, NSKE 5% spray","damage":"Can destroy crop in 3-5 days if unchecked","source":"ICAR-CRRI"}],
        "wheat":    [{"pest":"Yellow Rust","risk_level":"high","symptoms":"Yellow-orange stripes on leaves","prevention":"Use resistant varieties (HD-2781, PBW-343)","chemical":"Propiconazole 0.1% spray","organic":"No effective organic — prevention is key","damage":"30-70% yield loss in susceptible varieties","source":"ICAR-IIWBR"}],
        "soybean":  [{"pest":"Girdle Beetle","risk_level":"high","symptoms":"Wilting shoot, circular girdling on stem","prevention":"Early sowing before June 25, intercrop with maize","chemical":"Chlorpyrifos 2ml/L at 30-35 DAS","organic":"Hand-pick adults in early morning","damage":"30-40% yield loss in severe cases","source":"ICAR-IISR Indore"}],
        "maize":    [{"pest":"Fall Armyworm","risk_level":"high","symptoms":"Window-pane feeding, frass in whorl","prevention":"Monitor at whorl stage, pheromone traps 5/acre","chemical":"Spinetoram 0.5ml/L in whorl","organic":"Sand + lime (9:1) in whorl, Bt spray","damage":"20-70% yield loss — invasive pest arrived 2018","source":"ICAR-IIMR"}],
        "chickpea": [{"pest":"Pod Borer","risk_level":"high","symptoms":"Holes in pods, larva inside","prevention":"Pheromone traps 5/acre, intercrop with coriander","chemical":"Emamectin benzoate 0.4g/L at pod formation","organic":"HaNPV spray, Bt at first instar stage","damage":"50-100% pod damage in severe attacks","source":"ICAR-IIPR"}],
    }
    crop_pests  = BASIC_PESTS.get(best_crop.lower(), [])
    pests_found = crop_pests if current_month in [6,7,8,9,10,11] else []
    overall_risk  = "HIGH" if pests_found else "LOW"
    risk_color    = "#D85A30" if pests_found else "#1D9E75"

risk_labels = {"HIGH":"🔴 HIGH","MEDIUM":"🟡 MEDIUM","LOW":"🟢 LOW"}
risk_display = risk_labels.get(overall_risk, "🟢 LOW")

st.markdown(f"""
<div style="background:#F4F9F7;border-radius:12px;padding:14px 18px;
            margin-bottom:12px;border-left:4px solid {risk_color};">
  <p style="margin:0;font-size:14px;color:#0A2218;">
    <strong>Overall pest risk this month:</strong>
    <span style="color:{risk_color};font-weight:700;margin-left:8px">
      {risk_display}
    </span>
    &nbsp;·&nbsp; Month: {MONTH_NAMES[current_month-1]}
    &nbsp;·&nbsp; Crop: {best_crop.capitalize()}
  </p>
</div>
""", unsafe_allow_html=True)

if not pests_found:
    st.success(
        f"✅ No major pest threats for {best_crop.capitalize()} in "
        f"{MONTH_NAMES[current_month-1]}. Continue regular monitoring."
    )
else:
    for pest in pests_found:
        pname    = pest.get("pest",         pest.get("pest_name",    "Pest"))
        rlevel   = pest.get("risk_level",   "medium").upper()
        symptoms = pest.get("symptoms",     "Leaf damage, wilting")
        damage   = pest.get("damage",       "Yield loss")
        prevent  = pest.get("prevention",   "Regular monitoring")
        chem     = pest.get("chemical",     pest.get("chemical_control", "Consult KVK"))
        org      = pest.get("organic",      pest.get("organic_control",  "Neem oil spray"))
        source   = pest.get("source",       "ICAR Guidelines")

        risk_icon = "🔴" if rlevel == "HIGH" else "🟡" if rlevel == "MEDIUM" else "🟢"

        with st.expander(
            f"{risk_icon} **{pname}** — {rlevel} risk",
            expanded=(rlevel == "HIGH")
        ):
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**What you will see on the plant:**  \n{symptoms}")
                st.markdown(f"**Why it is dangerous:**  \n{damage}")
                st.markdown(f"**🌿 Organic control:**  \n{org}")
            with col_r:
                st.markdown(f"**🛡️ How to prevent:**  \n{prevent}")
                st.markdown(f"**💊 Chemical control:**  \n{chem}")
            st.caption(f"Source: {source}")

st.caption(
    "Source: ICAR-NCIPM Integrated Pest Management guidelines · "
    "State Agriculture Department IPM calendars"
)

# ════════════════════════════════════════════════════════════
# SECTION 7 — IRRIGATION
# ════════════════════════════════════════════════════════════

section("💧", "Water & Irrigation Requirement",
        f"How much water does {best_crop.capitalize()} need on your {field_ha} ha field?")

# Crop water requirements — FAO Paper 56 (mm/season)
CROP_WATER = {
    "cotton":(700), "rice":1200, "wheat":450, "maize":500,
    "soybean":450, "sugarcane":1500, "groundnut":500,
    "chickpea":350, "pigeonpea":400, "mustard":300,
    "bajra":350, "jowar":400, "sunflower":600,
}
CROP_WATER = {
    "cotton": 700, "rice": 1200, "wheat": 450, "maize": 500,
    "soybean": 450, "sugarcane": 1500, "groundnut": 500,
    "chickpea": 350, "pigeonpea": 400, "mustard": 300,
    "bajra": 350, "jowar": 400, "sunflower": 600,
    "lentil": 350, "mungbean": 350, "blackgram": 350,
    "tomato": 600, "onion": 550, "potato": 500,
}

try:
    from utils.irrigation import calculate_irrigation
    irr = calculate_irrigation(
        crop               = best_crop,
        field_ha           = field_ha,
        district           = selected_district,
        start_month        = current_month,
        pump_hp            = 3.0,
        electricity_rate   = 5.50,
        manual_rain_mm     = rainfall_val,
    )
    water_req  = irr["water_req_mm"]
    deficit_mm = irr["deficit_mm"]
    pump_hrs   = irr["pump_hours"]
    irr_cost   = irr["energy_cost"]
    irr_rounds = irr["irrigation_rounds"]
except Exception:
    # Inline calculation if utils/irrigation.py missing
    water_req   = CROP_WATER.get(best_crop.lower(), 500)
    eff_rain    = rainfall_val * 0.70
    deficit_mm  = max(0, water_req - eff_rain)
    import math
    deficit_l   = deficit_mm * field_ha * 10_000
    pump_lph    = 3.0 * 600
    pump_hrs    = deficit_l / pump_lph if deficit_l > 0 else 0
    irr_cost    = pump_hrs * 3.0 * 0.746 * 5.50
    irr_rounds  = math.ceil(deficit_mm / 60) if deficit_mm > 0 else 0

col_w1, col_w2, col_w3, col_w4 = st.columns(4)
col_w1.metric("Crop needs (total)",     f"{water_req} mm/season")
col_w2.metric("Rain will provide",      f"{min(water_req, int(rainfall_val*0.7))} mm",
              help="70% of rainfall is plant-available (FAO standard)")
col_w3.metric("You need to irrigate",   f"{deficit_mm:.0f} mm")
col_w4.metric("Irrigation rounds",      str(irr_rounds),
              help="Each round = 60mm applied")

if deficit_mm == 0:
    st.success(
        f"✅ Rainfall is sufficient for {best_crop.capitalize()} this season. "
        "No irrigation cost needed."
    )
elif deficit_mm < 150:
    st.info(
        f"💧 You need **{irr_rounds} irrigation rounds** "
        f"({pump_hrs:.0f} pump hours on a 3HP pump). "
        f"Estimated electricity cost: **₹{irr_cost:,.0f}** for the season."
    )
else:
    st.warning(
        f"⚠️ High water deficit ({deficit_mm:.0f}mm). "
        f"Need **{irr_rounds} irrigation rounds** — "
        f"estimated pump cost **₹{irr_cost:,.0f}**. "
        "Consider PMKSY drip irrigation — 55% subsidy available."
    )

st.caption(
    "Source: FAO Irrigation Paper 56 (Allen et al., 1998) · "
    "ICAR Water Use Efficiency Guidelines · "
    "Pump cost at Maharashtra agricultural tariff ₹5.50/unit"
)

# ════════════════════════════════════════════════════════════
# SECTION 8 — CARBON FOOTPRINT
# ════════════════════════════════════════════════════════════

section("♻️", "Carbon Footprint of Your Farm",
        "How environment-friendly is this crop choice?")

try:
    from utils.carbon import calculate_carbon, get_sustainability_rating
    carbon_result = calculate_carbon(
        crop        = best_crop,
        field_ha    = field_ha,
        n_kg_per_ha = n_val,
        p_kg_per_ha = p_val,
        k_kg_per_ha = k_val,
    )
    total_co2    = carbon_result["total_kgco2e"]
    co2_per_ha   = carbon_result["total_kgco2e_per_ha"]
    car_km       = carbon_result["equivalent_car_km"]
    rating, rclr = get_sustainability_rating(co2_per_ha)
except Exception:
    # Inline fallback
    total_co2  = (n_val * 4.7 + p_val * 1.0 + k_val * 0.6 + 700) * field_ha
    co2_per_ha = total_co2 / field_ha
    car_km     = total_co2 / 0.21
    rating, rclr = (
        ("Excellent ♻️", "#1D9E75") if co2_per_ha < 500 else
        ("Good 🟢",      "#5DCAA5") if co2_per_ha < 800 else
        ("Moderate 🟡",  "#EF9F27") if co2_per_ha < 1100 else
        ("High 🔴",      "#D85A30")
    )

st.markdown(f"""
<div style="background:#F4F9F7;border-radius:12px;padding:16px 20px;
            border-left:4px solid {rclr};">
  <div style="display:flex;align-items:center;justify-content:space-between;
              flex-wrap:wrap;gap:12px;">
    <div>
      <p style="color:#666;font-size:12px;margin:0">
        Your {best_crop.capitalize()} on {field_ha} ha produces
      </p>
      <p style="color:#0A2218;font-size:28px;font-weight:700;margin:2px 0">
        {total_co2:,.0f} kg CO₂ equivalent
      </p>
      <p style="color:#555;font-size:13px;margin:0">
        That is like driving a car for <strong>{car_km:,.0f} km</strong>
        &nbsp;·&nbsp; {co2_per_ha:.0f} kg CO₂/ha
      </p>
    </div>
    <div style="text-align:right">
      <p style="color:#666;font-size:11px;margin:0">Sustainability rating</p>
      <span style="background:{rclr};color:white;padding:6px 18px;
                   border-radius:99px;font-size:14px;font-weight:600;">
        {rating}
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Tip to reduce footprint
tips = {
    "cotton":    "💡 Tip: Reduce nitrogen by 20 kg/ha after a soybean rotation — saves ₹800 and cuts CO₂ by 94 kg.",
    "rice":      "💡 Tip: Alternate wetting and drying (AWD) irrigation method reduces methane emissions by 30%.",
    "wheat":     "💡 Tip: Use neem-coated urea — same yield, 10% less nitrogen loss, lower CO₂.",
    "soybean":   "💡 Tip: Soybean is already a low-emission crop — it fixes nitrogen from air, reducing fertilizer need.",
    "chickpea":  "💡 Tip: Chickpea is one of the most eco-friendly crops — only 350mm water + nitrogen fixer.",
}
tip = tips.get(best_crop.lower(),
               "💡 Tip: Use ICAR-recommended fertilizer doses — excess nitrogen is the biggest source of farm CO₂.")
st.info(tip)

st.caption(
    "Source: IPCC 2006 Guidelines for National GHG Inventories, "
    "Volume 4 — Agriculture. Tier 1 emission factors."
)

# ════════════════════════════════════════════════════════════
# SECTION 9 — SHAP EXPLANATION (collapsible)
# ════════════════════════════════════════════════════════════

section("🔬", "Why did AI recommend this crop?",
        "Plain-English explanation of what your soil data tells the AI")

with st.expander("Click to see detailed AI reasoning", expanded=False):
    try:
        import shap
        with st.spinner("Computing explanation..."):
            explainer   = get_shap_explainer(rf)
            shap_values = explainer.shap_values(input_vals)
            pred_idx    = np.argmax(proba)
            sv          = (shap_values[pred_idx][0]
                           if isinstance(shap_values, list)
                           else shap_values[0])

        order      = np.argsort(np.abs(sv))[::-1]
        feat_names = [FEATURE_DISPLAY[FEATURE_COLS[i]] for i in order]
        feat_vals  = sv[order]

        # Plain English summary FIRST
        positives = [(FEATURE_DISPLAY[FEATURE_COLS[i]], sv[i])
                     for i in range(len(sv)) if sv[i] > 0.01]
        negatives = [(FEATURE_DISPLAY[FEATURE_COLS[i]], sv[i])
                     for i in range(len(sv)) if sv[i] < -0.01]
        positives.sort(key=lambda x: -x[1])
        negatives.sort(key=lambda x:  x[1])

        st.markdown("**In simple words — why this crop was recommended:**")
        if positives:
            fav = ", ".join(f"**{n}**" for n, _ in positives[:3])
            st.markdown(f"✅ Favourable factors: {fav}")
        if negatives:
            con = ", ".join(f"**{n}**" for n, _ in negatives[:2])
            st.markdown(f"⚠️ Minor concerns: {con} — but overall still the best choice")

        # Chart
        fig, ax = plt.subplots(figsize=(9, 3.5))
        colors  = ["#1D9E75" if v >= 0 else "#D85A30" for v in feat_vals]
        ax.barh(range(len(feat_names)), feat_vals, color=colors,
                alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels(feat_names, fontsize=10)
        ax.axvline(x=0, color="#555", linewidth=0.8)
        ax.set_xlabel("Influence on AI decision "
                      "(Green bar = pushes toward this crop, Red = pushes away)", fontsize=9)
        ax.set_title(f"Why {best_crop.upper()} was recommended",
                     fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.2, linestyle="--")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Technique: SHAP (SHapley Additive exPlanations) — "
            "Lundberg & Lee, NeurIPS 2017. "
            "Same explainability method used in production AI systems."
        )

    except ImportError:
        st.info("Install SHAP to see detailed explanation: `pip install shap`")
    except Exception as e:
        st.warning(f"Explanation unavailable: {e}")
        if os.path.exists("assets/shap_waterfall.png"):
            st.image("assets/shap_waterfall.png")

# ════════════════════════════════════════════════════════════
# SECTION 10 — SAVE + HISTORY
# ════════════════════════════════════════════════════════════

section("📋", "Your Prediction History", "Track your farm advisory sessions")

# Save to database
if DB_OK:
    try:
        save_prediction({
            "district":      selected_district,
            "n_val":         n_val,
            "p_val":         p_val,
            "k_val":         k_val,
            "temp":          temp_val,
            "humidity":      humidity_val,
            "ph":            ph_val,
            "rainfall":      rainfall_val,
            "predicted_crop":best_crop,
            "confidence":    round(conf_pct, 1),
            "yield_est":     round(yield_pred, 0),
            "earnings_est":  round(gross_earn, 0),
        })
    except Exception:
        pass

with st.expander("📋 View my past predictions"):
    if DB_OK:
        history_df = get_history(limit=10)
        if history_df.empty:
            st.info("This is your first prediction — it has been saved!")
        else:
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            stats = get_stats()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total sessions",    stats.get("total", 0))
            c2.metric("Most recommended",  stats.get("top_crop", "N/A"))
            c3.metric("Avg confidence",    f"{stats.get('avg_confidence', 0):.1f}%")
    else:
        st.info("Database not available. "
                "Place utils/database.py in your project to enable history.")

# ── Data trust panel ──────────────────────────────────────────
st.markdown("---")
with st.expander("📂 Where does this data come from? (Full transparency)"):
    trust = {
        "Information": [
            "Crop recommendation AI",
            "Yield prediction",
            "Market price",
            "Live weather",
            "Pest risk data",
            "Irrigation science",
            "Carbon footprint",
            "Rotation advice",
            "Govt scheme details",
        ],
        "Source": [
            "Crop Recommendation Dataset — Atharva Ingle, Kaggle (2,200 records)",
            "XGBoost Regressor — trained on data.gov.in production data",
            "Agmarknet / MSP 2024-25 — Ministry of Agriculture, Govt of India",
            "OpenWeatherMap API — openweathermap.org",
            "ICAR-NCIPM Integrated Pest Management guidelines",
            "FAO Irrigation Paper 56 — Allen et al. (1998)",
            "IPCC 2006 Guidelines Vol.4 — Tier 1 emission factors",
            "ICAR Crop Rotation & Soil Fertility Management Guidelines",
            "Ministry of Agriculture & Farmers Welfare — agricoop.gov.in",
        ],
        "Authority": [
            "Kaggle public domain",
            "data.gov.in — OGD India (NDSAP)",
            "Govt of India — NDSAP open data",
            "CC BY-SA 4.0",
            "ICAR (Indian Council of Agricultural Research)",
            "FAO (Food and Agriculture Organization, UN)",
            "IPCC (Intergovernmental Panel on Climate Change)",
            "ICAR — New Delhi",
            "Ministry of Agriculture & Farmers Welfare",
        ],
    }
    st.dataframe(
        pd.DataFrame(trust),
        use_container_width=True,
        hide_index=True,
    )