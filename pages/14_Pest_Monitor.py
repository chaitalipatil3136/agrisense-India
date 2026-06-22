"""
AgriSense India — Pest Outbreak Risk Predictor
File: pages/14_Pest_Monitor.py
 
Proactive pest risk advisory: given crop + month + weather conditions
→ shows which pests are HIGH/MEDIUM/LOW risk this season
→ prevention steps to take BEFORE damage appears.
 
Also creates assets/pest_calendar.json on first run.
 
Science source: ICAR-NCIPM (National Centre for Integrated Pest Management)
                ICAR Crop Protection Guides
                State Agriculture Department IPM calendars
 
Run: streamlit run app.py → Pest Monitor
No new installs needed.
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
from datetime import datetime
 
st.set_page_config(
    page_title="Pest Monitor — AgriSense India",
    page_icon="🐛",
    layout="wide",
)
 
# ════════════════════════════════════════════════════════════
# PEST CALENDAR DATA
# Source: ICAR-NCIPM + State Agriculture Department IPM guides
# ════════════════════════════════════════════════════════════
 
PEST_CALENDAR = {
    "cotton": [
        {
            "pest": "Whitefly (Bemisia tabaci)",
            "type": "insect",
            "risk_months": [6, 7, 8, 9, 10],
            "risk_level": "high",
            "trigger": {"min_temp": 28, "max_temp": 42, "min_rain_mm": 0},
            "symptoms": "Yellowing leaves, sticky honeydew on leaf surface, tiny white insects under leaves",
            "damage": "Transmits Cotton Leaf Curl Virus (CLCuV). Can destroy entire crop.",
            "prevention": "Install yellow sticky traps (10/acre). Avoid planting near old cotton fields. Remove volunteer plants.",
            "organic": "Neem oil 3ml/L + garlic extract spray. Release Encarsia formosa (parasitoid).",
            "chemical": "Imidacloprid 17.8 SL @ 0.5ml/L OR Thiamethoxam 25 WG @ 0.3g/L. Spray undersides of leaves.",
            "source": "ICAR-CICR Nagpur Pest Management Advisory",
        },
        {
            "pest": "American Bollworm (Helicoverpa armigera)",
            "type": "insect",
            "risk_months": [8, 9, 10, 11],
            "risk_level": "high",
            "trigger": {"min_temp": 25, "max_temp": 38, "min_rain_mm": 30},
            "symptoms": "Bored bolls with entry holes, larva inside boll, caterpillar on squares/flowers",
            "damage": "Up to 50-80% boll damage in severe infestations. Major economic pest.",
            "prevention": "Set up pheromone traps (5/acre) to monitor adult moth population. Plant marigold border crop as trap crop.",
            "organic": "Spray HaNPV (Helicoverpa Nuclear Polyhedrosis Virus) @ 250 LE/acre. Bacillus thuringiensis (Bt) spray.",
            "chemical": "Emamectin benzoate 5 SG @ 0.4g/L OR Spinosad 45 SC @ 0.3ml/L. Rotate with Indoxacarb.",
            "source": "ICAR-CICR Bollworm IPM Protocol",
        },
        {
            "pest": "Pink Bollworm (Pectinophora gossypiella)",
            "type": "insect",
            "risk_months": [9, 10, 11, 12],
            "risk_level": "high",
            "trigger": {"min_temp": 22, "max_temp": 36, "min_rain_mm": 0},
            "symptoms": "Rosette flowers (infested squares fail to open), damaged seeds inside bolls with pink larva",
            "damage": "Attacks bolls from within. Reduces seed cotton yield and fibre quality.",
            "prevention": "Use Bt cotton varieties. Install pheromone traps for monitoring. Deep ploughing after harvest.",
            "organic": "Release sterile pink bollworm moths (SIT method). Pheromone mass trapping.",
            "chemical": "Cypermethrin 10 EC @ 0.5ml/L OR Chlorpyrifos 20 EC @ 2ml/L during boll development stage.",
            "source": "ICAR-CICR Nagpur",
        },
        {
            "pest": "Thrips (Thrips tabaci)",
            "type": "insect",
            "risk_months": [5, 6, 7, 8],
            "risk_level": "medium",
            "trigger": {"min_temp": 25, "max_temp": 45, "min_rain_mm": 0},
            "symptoms": "Silvery streaks on leaves, upward curling of leaf margins, stunted plant growth",
            "damage": "Causes 'Hopper burn' effect. Vector for Tomato Spotted Wilt Virus in some regions.",
            "prevention": "Monitor with blue sticky traps (10/acre). Avoid water stress — thrips worsen in dry conditions.",
            "organic": "Neem oil 5ml/L spray. Reflective mulches repel thrips.",
            "chemical": "Spinosad 45 SC @ 0.3ml/L OR Fipronil 5 SC @ 1.5ml/L.",
            "source": "ICAR Integrated Pest Management in Cotton",
        },
    ],
 
    "soybean": [
        {
            "pest": "Girdle Beetle (Obereopsis brevis)",
            "type": "insect",
            "risk_months": [7, 8, 9],
            "risk_level": "high",
            "trigger": {"min_temp": 25, "max_temp": 35, "min_rain_mm": 50},
            "symptoms": "Wilting of terminal shoot, circular girdling marks on stem 15-20cm from top, yellowing",
            "damage": "Causes 30-40% yield loss in severe cases. Girdles stem cutting off nutrient supply.",
            "prevention": "Collect and destroy girdled shoots. Early sowing (before June 25) reduces attack. Intercrop with maize.",
            "organic": "Hand pick and destroy adults in early morning. NSKE 5% spray.",
            "chemical": "Chlorpyrifos 20 EC @ 2ml/L OR Profenofos 50 EC @ 2ml/L at 30-35 DAS.",
            "source": "ICAR-IISR Indore Soybean Pest Guide",
        },
        {
            "pest": "Yellow Mosaic Virus (via Whitefly vector)",
            "type": "virus",
            "risk_months": [7, 8, 9, 10],
            "risk_level": "high",
            "trigger": {"min_temp": 26, "max_temp": 38, "min_rain_mm": 20},
            "symptoms": "Bright yellow patches on young leaves, mosaic yellowing pattern, stunted plants, reduced pod set",
            "damage": "No cure once infected. Can cause 80-100% yield loss in severe outbreaks.",
            "prevention": "Use resistant variety JS-335 or JS-9560. Remove and destroy infected plants immediately.",
            "organic": "Remove infected plants. Control whitefly vector with neem oil 3ml/L.",
            "chemical": "Imidacloprid 17.8 SL @ 0.5ml/L for whitefly control at 10 and 20 DAS.",
            "source": "ICAR-IISR Yellow Mosaic Disease Advisory",
        },
        {
            "pest": "Stem Fly (Melanagromyza sojae)",
            "type": "insect",
            "risk_months": [6, 7, 8],
            "risk_level": "medium",
            "trigger": {"min_temp": 22, "max_temp": 32, "min_rain_mm": 40},
            "symptoms": "Dead heart symptom in young plants, yellowing of leaves, maggot in stem on dissection",
            "damage": "Early infection (before 25 DAS) causes 10-30% plant mortality.",
            "prevention": "Seed treatment with Thiamethoxam 70 WS @ 3g/kg seed before sowing.",
            "organic": "NSKE 5% spray at 7 and 14 DAS.",
            "chemical": "Dimethoate 30 EC @ 2ml/L at 7 and 14 DAS.",
            "source": "ICAR-IISR Indore",
        },
    ],
 
    "rice": [
        {
            "pest": "Brown Planthopper (Nilaparvata lugens)",
            "type": "insect",
            "risk_months": [8, 9, 10],
            "risk_level": "high",
            "trigger": {"min_temp": 25, "max_temp": 32, "min_rain_mm": 80},
            "symptoms": "Yellowing and drying of plants in circular patches ('hopperburn'), honeydew on leaf sheaths",
            "damage": "Can cause complete crop failure ('hopperburn') within 3-5 days of outbreak.",
            "prevention": "Avoid excessive nitrogen. Do not close plant spacing below 20x20cm. Light traps to monitor.",
            "organic": "Drain water from field for 3-4 days to expose insects. NSKE 5% spray.",
            "chemical": "Buprofezin 25 SC @ 1.25ml/L OR Imidacloprid 17.8 SL @ 0.25ml/L. DO NOT use synthetic pyrethroids — they kill natural enemies and cause resurgence.",
            "source": "ICAR-CRRI Cuttack BPH Management Protocol",
        },
        {
            "pest": "Stem Borer (Scirpophaga incertulas)",
            "type": "insect",
            "risk_months": [6, 7, 8, 9, 10],
            "risk_level": "high",
            "trigger": {"min_temp": 24, "max_temp": 34, "min_rain_mm": 50},
            "symptoms": "Dead heart at vegetative stage, white ear at reproductive stage, larva in stem",
            "damage": "Dead heart (vegetative) or white ear (reproductive) causes direct yield loss.",
            "prevention": "Remove egg masses from nursery. Clip seedling tips before transplanting to remove eggs.",
            "organic": "Trichogramma japonicum release @ 50,000/acre at egg stage.",
            "chemical": "Cartap hydrochloride 4G @ 10kg/acre OR Chlorpyrifos 20 EC @ 2ml/L.",
            "source": "ICAR Rice IPM Guidelines",
        },
        {
            "pest": "Leaf Folder (Cnaphalocrocis medinalis)",
            "type": "insect",
            "risk_months": [7, 8, 9],
            "risk_level": "medium",
            "trigger": {"min_temp": 26, "max_temp": 35, "min_rain_mm": 60},
            "symptoms": "Longitudinally folded leaves with white papery appearance, caterpillar inside fold",
            "damage": "Reduces photosynthetic area. Severe attack at panicle initiation reduces yield 15-20%.",
            "prevention": "Monitor at panicle initiation stage. Control only if >25% folded leaves.",
            "organic": "Trichogramma chilonis release. NSKE 5% spray.",
            "chemical": "Monocrotophos 36 SL @ 1.5ml/L OR Chlorpyrifos 20 EC @ 2ml/L.",
            "source": "ICAR-CRRI",
        },
    ],
 
    "wheat": [
        {
            "pest": "Yellow Rust (Puccinia striiformis)",
            "type": "fungus",
            "risk_months": [12, 1, 2],
            "risk_level": "high",
            "trigger": {"min_temp": 7, "max_temp": 15, "min_rain_mm": 10},
            "symptoms": "Bright yellow-orange pustule stripes parallel to leaf veins, yellow powder on leaves",
            "damage": "Can cause 30-70% yield loss in susceptible varieties in cool moist conditions.",
            "prevention": "Use resistant varieties (HD-2781, PBW-343). Avoid late sowing.",
            "organic": "No effective organic control once infection starts. Prevention is key.",
            "chemical": "Propiconazole 25 EC @ 0.1% OR Tebuconazole 25.9 EC @ 1ml/L. Apply at first sign.",
            "source": "ICAR-IIWBR Karnal Rust Advisory",
        },
        {
            "pest": "Aphids (Sitobion avenae)",
            "type": "insect",
            "risk_months": [1, 2, 3],
            "risk_level": "medium",
            "trigger": {"min_temp": 8, "max_temp": 20, "min_rain_mm": 0},
            "symptoms": "Colonies of small green/yellow insects on leaves and spikes, yellowing and curling",
            "damage": "Direct sap sucking + transmits Barley Yellow Dwarf Virus. 10-25% yield loss.",
            "prevention": "Monitor at flag leaf stage. Conserve natural enemies (ladybirds, parasitic wasps).",
            "organic": "Strong water spray to dislodge. NSKE 5% spray.",
            "chemical": "Dimethoate 30 EC @ 1.5ml/L OR Imidacloprid 17.8 SL @ 0.5ml/L. Treat at ETL of 2-3 aphids/tiller.",
            "source": "ICAR-IIWBR",
        },
        {
            "pest": "Karnal Bunt (Tilletia indica)",
            "type": "fungus",
            "risk_months": [2, 3],
            "risk_level": "medium",
            "trigger": {"min_temp": 16, "max_temp": 22, "min_rain_mm": 20},
            "symptoms": "Black powdery spore mass partially replacing grain, fishy odour, blackened kernels",
            "damage": "Downgrades grain quality. Export-prohibited if >0.05% incidence.",
            "prevention": "Seed treatment with Carbendazim 50 WP @ 2.5g/kg. Use certified pathogen-free seed.",
            "organic": "Hot water seed treatment at 52°C for 10 minutes.",
            "chemical": "Propiconazole seed treatment @ 1ml/kg seed OR Raxil (Tebuconazole) @ 1.5g/kg.",
            "source": "ICAR-IIWBR Karnal Bunt Management",
        },
    ],
 
    "chickpea": [
        {
            "pest": "Pod Borer (Helicoverpa armigera)",
            "type": "insect",
            "risk_months": [1, 2, 3],
            "risk_level": "high",
            "trigger": {"min_temp": 15, "max_temp": 30, "min_rain_mm": 0},
            "symptoms": "Caterpillar feeding on pods, circular entry holes in pods, damaged seeds",
            "damage": "Most destructive chickpea pest. Can cause 50-100% pod damage.",
            "prevention": "Set up pheromone traps (5/acre). Intercrop with coriander or mustard to attract natural enemies.",
            "organic": "HaNPV @ 250 LE/acre at egg hatching. Bt spray at first instar stage.",
            "chemical": "Emamectin benzoate 5 SG @ 0.4g/L OR Indoxacarb 14.5 SC @ 1ml/L at pod formation.",
            "source": "ICAR-IIPR Kanpur Pod Borer Advisory",
        },
        {
            "pest": "Fusarium Wilt (Fusarium oxysporum)",
            "type": "fungus",
            "risk_months": [11, 12, 1],
            "risk_level": "medium",
            "trigger": {"min_temp": 15, "max_temp": 25, "min_rain_mm": 15},
            "symptoms": "Wilting of plant, yellowing from base upward, dark discolouration of root and stem",
            "damage": "Soil-borne. Can persist 5-7 years. 20-30% plant mortality in infected fields.",
            "prevention": "Use wilt-resistant varieties. Treat seed with Trichoderma viride @ 4g/kg.",
            "organic": "Trichoderma harzianum soil application @ 2.5 kg/acre in FYM.",
            "chemical": "Carbendazim 50 WP @ 2g/kg seed treatment OR soil drenching with Metalaxyl.",
            "source": "ICAR-IIPR",
        },
    ],
 
    "tomato": [
        {
            "pest": "Fruit Borer (Helicoverpa armigera)",
            "type": "insect",
            "risk_months": [8, 9, 10, 11, 12, 1, 2],
            "risk_level": "high",
            "trigger": {"min_temp": 20, "max_temp": 35, "min_rain_mm": 0},
            "symptoms": "Circular entry holes on fruits, caterpillar inside fruit, frass at entry hole",
            "damage": "Single larva destroys multiple fruits. 30-70% fruit loss in severe attacks.",
            "prevention": "Pheromone traps (5/acre). Remove and destroy infested fruits. Intercrop with African marigold.",
            "organic": "HaNPV @ 250 LE/acre. Bt spray at first instar. Release Trichogramma pretiosum.",
            "chemical": "Spinosad 45 SC @ 0.3ml/L OR Emamectin benzoate 5 SG @ 0.4g/L.",
            "source": "ICAR-IIVR Varanasi",
        },
        {
            "pest": "TYLCV via Whitefly (Bemisia tabaci)",
            "type": "virus",
            "risk_months": [3, 4, 5, 6, 7],
            "risk_level": "high",
            "trigger": {"min_temp": 26, "max_temp": 40, "min_rain_mm": 0},
            "symptoms": "Upward leaf curling, yellowing of leaf margins, stunted growth, no fruit set",
            "damage": "No cure. Remove infected plants immediately. Entire crop loss possible.",
            "prevention": "Use TYLCV-resistant varieties (Arka Rakshak, Pusa Rohini). Insect-proof nursery.",
            "organic": "Yellow sticky traps. Neem oil 3ml/L to control whitefly vector.",
            "chemical": "Imidacloprid 17.8 SL @ 0.5ml/L OR Acetamiprid 20 SP @ 0.2g/L for whitefly.",
            "source": "ICAR-IIVR TYLCV Management",
        },
        {
            "pest": "Early Blight (Alternaria solani)",
            "type": "fungus",
            "risk_months": [10, 11, 12, 1],
            "risk_level": "medium",
            "trigger": {"min_temp": 18, "max_temp": 28, "min_rain_mm": 20},
            "symptoms": "Dark brown circular spots with concentric rings (target board pattern) on older leaves",
            "damage": "Defoliation from bottom up reduces yield 20-40%.",
            "prevention": "Avoid overhead irrigation. Remove infected lower leaves. Crop rotation.",
            "organic": "Neem oil 5ml/L spray. Trichoderma-based biocontrol.",
            "chemical": "Mancozeb 75 WP @ 2.5g/L OR Chlorothalonil 75 WP @ 2g/L every 7-10 days.",
            "source": "ICAR-IIVR",
        },
    ],
 
    "maize": [
        {
            "pest": "Fall Armyworm (Spodoptera frugiperda)",
            "type": "insect",
            "risk_months": [6, 7, 8, 9],
            "risk_level": "high",
            "trigger": {"min_temp": 22, "max_temp": 38, "min_rain_mm": 30},
            "symptoms": "Window-pane feeding on leaves, ragged edges, frass in whorl, pinhole damage",
            "damage": "Invasive pest. Can cause 20-70% yield loss. Arrived in India 2018.",
            "prevention": "Monitor whorl stage carefully. Pheromone traps (5/acre). Sand + lime mixture in whorl.",
            "organic": "Apply sand + lime (9:1) in whorl at 2-3 weeks. Bt spray @ 2g/L in whorl.",
            "chemical": "Spinetoram 11.7 SC @ 0.5ml/L OR Emamectin benzoate 5 SG @ 0.4g/L in whorl.",
            "source": "ICAR-IIMR Hyderabad Fall Armyworm Emergency Advisory",
        },
        {
            "pest": "Northern Leaf Blight (Exserohilum turcicum)",
            "type": "fungus",
            "risk_months": [8, 9, 10],
            "risk_level": "medium",
            "trigger": {"min_temp": 18, "max_temp": 27, "min_rain_mm": 50},
            "symptoms": "Long cigar-shaped tan/brown lesions parallel to leaf veins on upper canopy",
            "damage": "Reduces photosynthetic area. Severe attack at tasseling causes 30-50% yield loss.",
            "prevention": "Plant resistant hybrids. Crop rotation with non-host crops.",
            "organic": "Remove infected leaves. Neem oil spray as preventive.",
            "chemical": "Mancozeb 75 WP @ 2.5g/L OR Propiconazole 25 EC @ 1ml/L at first sign.",
            "source": "ICAR-IIMR",
        },
    ],
 
    "groundnut": [
        {
            "pest": "Leaf Miner (Aproaerema modicella)",
            "type": "insect",
            "risk_months": [8, 9, 10],
            "risk_level": "high",
            "trigger": {"min_temp": 25, "max_temp": 38, "min_rain_mm": 30},
            "symptoms": "White serpentine mines on leaflets, leaf folding, brownish papery leaves",
            "damage": "Severe defoliation reduces pod filling. 20-30% yield loss in Kharif.",
            "prevention": "Early sowing (June-July). Remove infected leaves. Intercrop with sunflower.",
            "organic": "NSKE 5% spray. Neem oil 3ml/L.",
            "chemical": "Quinalphos 25 EC @ 2ml/L OR Chlorpyrifos 20 EC @ 2ml/L.",
            "source": "ICAR Groundnut IPM Guidelines",
        },
        {
            "pest": "Tikka Disease (Cercospora arachidicola)",
            "type": "fungus",
            "risk_months": [8, 9, 10],
            "risk_level": "medium",
            "trigger": {"min_temp": 25, "max_temp": 35, "min_rain_mm": 60},
            "symptoms": "Circular brown spots with yellow halo on leaves (early tikka) or dark spots (late tikka)",
            "damage": "Premature defoliation. Reduces kernel size and oil content. 20-30% yield loss.",
            "prevention": "Seed treatment with Thiram 75 WP @ 3g/kg. Crop rotation.",
            "organic": "Spray Bordeaux mixture 1%.",
            "chemical": "Mancozeb 75 WP @ 2.5g/L OR Tebuconazole 250 EW @ 1ml/L.",
            "source": "ICAR-DGR Junagadh",
        },
    ],
}
 
 
# ════════════════════════════════════════════════════════════
# PEST ADVISORY LOGIC
# ════════════════════════════════════════════════════════════
 
def get_pest_risks(crop: str, month: int,
                   temp_c: float, rain_mm: float) -> list:
    """
    Returns list of active pest risks for crop in given conditions.
    Sorted: high → medium → low.
    """
    crop_lower = crop.lower().strip()
    pests      = PEST_CALENDAR.get(crop_lower, [])
 
    active = []
    for p in pests:
        if month not in p["risk_months"]:
            continue
 
        trig    = p.get("trigger", {})
        min_t   = trig.get("min_temp", -99)
        max_t   = trig.get("max_temp", 99)
        min_r   = trig.get("min_rain_mm", 0)
 
        # Check trigger conditions
        temp_ok = min_t <= temp_c <= max_t
        rain_ok = rain_mm >= min_r
 
        if temp_ok and rain_ok:
            active.append(p)
 
    # Sort: high → medium → low
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(active, key=lambda x: order.get(x["risk_level"], 3))
 
 
def get_overall_risk(pest_list: list) -> tuple:
    """Returns (level_str, color_hex) for the badge."""
    if not pest_list:
        return "LOW", "#1D9E75"
    levels = [p["risk_level"] for p in pest_list]
    if "high"   in levels: return "HIGH",   "#D85A30"
    if "medium" in levels: return "MEDIUM", "#EF9F27"
    return                        "LOW",    "#1D9E75"
 
 
# ════════════════════════════════════════════════════════════
# SAVE pest_calendar.json
# ════════════════════════════════════════════════════════════
 
def ensure_pest_json():
    path = "assets/pest_calendar.json"
    os.makedirs("assets", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(PEST_CALENDAR, f, indent=2, ensure_ascii=False)
 
 
ensure_pest_json()
 
 
# ════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════
 
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
 
st.title("🐛 Pest Outbreak Risk Predictor")
st.caption(
    "Proactive pest advisory — warns you BEFORE damage appears · "
    "ICAR-NCIPM Integrated Pest Management knowledge base"
)
 
col_in, col_out = st.columns([1, 2])
 
with col_in:
    st.markdown("### Inputs")
 
    crop = st.selectbox(
        "Crop",
        ["Cotton","Soybean","Rice","Wheat","Chickpea","Tomato","Maize","Groundnut"],
        help="Select your current or planned crop"
    )
 
    current_month = datetime.now().month
    month = st.slider(
        "Month",
        min_value=1, max_value=12,
        value=current_month,
        format="%d",
        help="Check risk for a specific month",
    )
    st.caption(f"Selected: **{MONTH_NAMES[month-1]}**")
 
    temp_c   = st.number_input("Temperature (°C)", 5.0, 48.0, 32.0, 0.5)
    rain_mm  = st.number_input("Monthly Rainfall (mm)", 0.0, 400.0, 80.0, 5.0)
 
    check_btn = st.button("🔍 Check Pest Risk", type="primary",
                          use_container_width=True)
 
    st.markdown("---")
    st.markdown("**Crops covered:**")
    for c in PEST_CALENDAR.keys():
        n = len(PEST_CALENDAR[c])
        st.markdown(f"- {c.capitalize()} ({n} pests/diseases)")
 
with col_out:
    if check_btn:
        active_pests  = get_pest_risks(crop, month, temp_c, rain_mm)
        overall, oclr = get_overall_risk(active_pests)
 
        # Overall risk banner
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0A1628,#0F2D1E);
                    border-radius:12px;padding:16px 22px;margin-bottom:16px;
                    border-left:5px solid {oclr};">
          <p style="color:#9BBFA0;font-size:12px;margin:0">
            Pest Risk Assessment — {crop}, {MONTH_NAMES[month-1]}
          </p>
          <div style="display:flex;align-items:center;gap:16px;margin-top:6px;">
            <p style="color:#FFFFFF;font-size:28px;font-weight:700;margin:0">
              {overall} RISK
            </p>
            <span style="background:{oclr};color:white;padding:4px 14px;
                         border-radius:99px;font-size:13px;font-weight:500;">
              {len(active_pests)} active threat{'s' if len(active_pests)!=1 else ''}
            </span>
          </div>
          <p style="color:#9BBFA0;font-size:12px;margin:6px 0 0">
            Temp: {temp_c}°C · Monthly rain: {rain_mm:.0f}mm · {MONTH_NAMES[month-1]}
          </p>
        </div>
        """, unsafe_allow_html=True)
 
        if not active_pests:
            st.success(
                f"✅ No significant pest threats for {crop} in {MONTH_NAMES[month-1]} "
                f"under current conditions. Continue regular monitoring."
            )
        else:
            # Pest cards
            RISK_COLORS = {"high": "#D85A30", "medium": "#EF9F27", "low": "#1D9E75"}
            RISK_ICONS  = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            TYPE_ICONS  = {"insect": "🐛", "fungus": "🍄", "virus": "🦠"}
 
            for pest in active_pests:
                rc  = RISK_COLORS.get(pest["risk_level"], "#888")
                ri  = RISK_ICONS.get(pest["risk_level"],  "⚪")
                ti  = TYPE_ICONS.get(pest["type"], "⚠️")
                exp = pest["risk_level"] == "high"  # auto-expand high risk
 
                with st.expander(
                    f"{ri} **{pest['pest']}** — {pest['risk_level'].upper()} risk",
                    expanded=exp
                ):
                    col_a, col_b = st.columns(2)
 
                    with col_a:
                        st.markdown(f"**{ti} Type:** {pest['type'].capitalize()}")
                        st.markdown(f"**📅 Peak months:** "
                                    f"{', '.join(MONTH_NAMES[m-1] for m in pest['risk_months'])}")
                        st.markdown(f"**⚠️ Symptoms:**  \n{pest['symptoms']}")
                        st.markdown(f"**💥 Damage:**  \n{pest['damage']}")
 
                    with col_b:
                        st.markdown(f"**🛡️ Prevention:**  \n{pest['prevention']}")
                        st.markdown(f"**🌿 Organic control:**  \n{pest['organic']}")
                        st.markdown(f"**💊 Chemical control:**  \n{pest['chemical']}")
 
                    st.caption(f"Source: {pest['source']}")
 
        # 12-month pest calendar heatmap
        st.markdown("---")
        st.markdown("### 📅 12-Month Pest Risk Calendar")
        st.caption("All pests for this crop across all months — red = HIGH risk season")
 
        crop_pests = PEST_CALENDAR.get(crop.lower(), [])
        if crop_pests:
            pest_names = [p["pest"][:35] for p in crop_pests]
            heat_z     = []
            RISK_VAL   = {"high": 3, "medium": 2, "low": 1, "none": 0}
 
            for pest in crop_pests:
                row = []
                for mo in range(1, 13):
                    if mo in pest["risk_months"]:
                        row.append(RISK_VAL.get(pest["risk_level"], 0))
                    else:
                        row.append(0)
                heat_z.append(row)
 
            fig = go.Figure(go.Heatmap(
                z=heat_z,
                x=MONTH_NAMES,
                y=pest_names,
                colorscale=[
                    [0.0,  "#1A1A2E"],
                    [0.33, "#1D9E75"],
                    [0.66, "#EF9F27"],
                    [1.0,  "#D85A30"],
                ],
                zmin=0, zmax=3,
                colorbar=dict(
                    tickvals=[0, 1, 2, 3],
                    ticktext=["No risk", "Low", "Medium", "High"],
                    title="Risk",
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Month: %{x}<br>"
                    "Risk: %{z}<extra></extra>"
                ),
            ))
 
            # Highlight current month
            fig.add_trace(go.Scatter(
                x=[current_month, current_month],
                y=[0,10],  # adjust based on your y-axis
                mode="lines",
                line=dict(color="#888", width=1.5, dash="dot"),
                name="Current Month",
            ))
 
            fig.update_layout(
                height=max(250, len(crop_pests) * 45 + 80),
                margin=dict(t=20, b=40, l=10, r=10),
                paper_bgcolor="#0A1628",
                plot_bgcolor="#0A1628",
                font=dict(color="white"),
                xaxis=dict(side="top"),
            )
            st.plotly_chart(fig, use_container_width=True)
 
    else:
        st.markdown("""
        <div style="background:#F4F9F7;border-radius:12px;padding:50px 40px;
                    text-align:center;border:2px dashed #1D9E75;margin-top:20px;">
          <p style="font-size:40px;margin:0 0 12px">🐛</p>
          <p style="font-size:15px;color:#0A1628;font-weight:500;margin:0 0 6px">
            Select your crop and check pest risk
          </p>
          <p style="font-size:12px;color:#64748B;margin:0;line-height:1.6">
            AgriSense warns you BEFORE pests appear — not after damage is done.<br>
            Proactive advisory based on ICAR-NCIPM IPM knowledge base.
          </p>
        </div>
        """, unsafe_allow_html=True)
 
st.markdown("---")
st.caption(
    "Source: ICAR-NCIPM (National Centre for Integrated Pest Management) · "
    "ICAR-CICR, ICAR-IISR, ICAR-IIVR, ICAR-CRRI crop-specific pest guides · "
    "State Agriculture Department IPM calendars · "
    "AgriSense India — MIT CSN Nagpur"
)
 