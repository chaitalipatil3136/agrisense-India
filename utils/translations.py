"""
AgriSense India — Multilingual Translations
File: utils/translations.py

Supports: English (en) + Hindi (hi)

Usage in any page:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.translations import T, get_lang, lang_selector

    lang = get_lang()          # reads from st.session_state
    st.title(T[lang]["app_title"])

Add lang_selector() call in your sidebar to show the toggle.
"""

# ════════════════════════════════════════════════════════════
# ALL TRANSLATIONS
# Key rule: every key must exist in BOTH "en" and "hi"
# ════════════════════════════════════════════════════════════

T = {

    # ────────────────────────────────────────────────────────
    # ENGLISH
    # ────────────────────────────────────────────────────────
    "en": {

        # App-wide
        "app_title":          "AgriSense India",
        "app_tagline":        "Intelligent Crop Advisory System for Indian Farmers",
        "language_label":     "🌐 Language / भाषा",
        "built_by":           "Built by · MIT CSN Nagpur · 2nd Year CS",

        # Navigation
        "nav_crop_advisor":   "🌱 Crop Advisor",
        "nav_disease":        "🍃 Disease Detector",
        "nav_carbon":         "♻️ Carbon Footprint",
        "nav_rotation":       "🗓️ Rotation Planner",
        "nav_india_map":      "🗺️ India Crop Map",
        "nav_model":          "📊 Model Insights",
        "nav_price":          "📈 Price Forecaster",
        "nav_fertilizer":     "🧪 Fertilizer Optimizer",
        "nav_grader":         "🍎 Produce Grader",
        "nav_ndvi":           "🛰️ NDVI Monitor",
        "nav_pest":           "🐛 Pest Monitor",
        "nav_irrigation":     "💧 Irrigation Calculator",

        # Crop Advisor page
        "ca_title":           "🌱 Crop Advisor",
        "ca_subtitle":        "Enter your soil details — get crop recommendation, earnings, pest risk, irrigation and govt schemes in one place.",
        "ca_step1":           "📍 Your Location",
        "ca_step1_sub":       "Select district — weather fills automatically",
        "ca_district":        "District / Taluka",
        "ca_field_size":      "Your field size (hectares)",
        "ca_field_help":      "1 hectare = 2.47 acres",
        "ca_weather_btn":     "🌤️ Get live weather for my district",
        "ca_weather_ok":      "✅ Live weather loaded",
        "ca_weather_fail":    "Weather server unavailable. Enter values manually below.",
        "ca_temperature":     "Temperature",
        "ca_humidity":        "Humidity",
        "ca_weather_source":  "Source: OpenWeatherMap · Updates when you click button above",
        "ca_step2":           "🧪 Your Soil Data",
        "ca_step2_sub":       "Get N, P, K, pH from your Soil Health Card (green card from government)",
        "ca_nitrogen":        "Nitrogen — N (kg/ha)",
        "ca_nitrogen_help":   "📋 From Soil Health Card → 'Available N'. Typical: 50–120 kg/ha",
        "ca_phosphorus":      "Phosphorus — P (kg/ha)",
        "ca_phosphorus_help": "📋 From Soil Health Card → 'Available P'. Typical: 20–80 kg/ha",
        "ca_potassium":       "Potassium — K (kg/ha)",
        "ca_potassium_help":  "📋 From Soil Health Card → 'Available K'. Typical: 100–200 kg/ha",
        "ca_ph":              "Soil pH (acidity level)",
        "ca_ph_help":         "7.0 = neutral. Below 7 = acidic. Most crops prefer 6.0–7.5",
        "ca_temp_input":      "Temperature (°C)",
        "ca_humid_input":     "Humidity (%)",
        "ca_rainfall":        "Expected rainfall this season (mm)",
        "ca_rainfall_help":   "Check IMD forecast or last year's rainfall for your district",
        "ca_predict_btn":     "🔮  Get My Complete Farm Advisory",
        "ca_predict_info":    "👆 Fill in your soil details above and click the button to get complete farm advisory.",
        "ca_loading":         "🌾 AI is analysing your soil and climate data...",
        "ca_model_error":     "❌ Could not load AI models. Run notebooks/05_crop_model.py first.",

        # Recommendation section
        "ca_rec_title":       "🏆 AI Crop Recommendation",
        "ca_rec_sub":         "Based on your soil data, weather, and 2,200 farming records",
        "ca_best_crop":       "Best crop for your field",
        "ca_alternatives":    "Other crops that could also work:",
        "ca_conf_high":       "Very confident",
        "ca_conf_med":        "Confident",
        "ca_conf_low":        "Moderate confidence",
        "ca_model_source":    "📊 Model: Random Forest (99.3% accuracy) · Trained on crop_recommendation.csv · Source: Atharva Ingle, Kaggle",

        # Yield section
        "ca_yield_title":     "💰 Expected Yield & Earnings",
        "ca_yield_label":     "Expected yield",
        "ca_price_label":     "Market price",
        "ca_gross_label":     "Gross earnings",
        "ca_net_label":       "Estimated net profit",
        "ca_after_cost":      "After cultivation cost",
        "ca_price_source":    "Market price source:",
        "ca_low_profit":      "⚠️ At current prices, this crop may not be profitable. Consider the 2nd or 3rd recommended crop, or check PMFBY insurance.",

        # Govt schemes section
        "ca_schemes_title":   "🏛️ Government Schemes You May Qualify For",
        "ca_schemes_sub":     "Auto-checked based on your crop, district, and field size",
        "ca_schemes_found":   "✅ You are likely eligible for {n} government schemes",
        "ca_what_you_get":    "What you get:",
        "ca_how_apply":       "How to apply:",
        "ca_docs_needed":     "Documents needed:",
        "ca_schemes_note":    "Source: Ministry of Agriculture & Farmers Welfare, Govt of India · Eligibility is indicative — verify at official portals before applying",

        # Pest section
        "ca_pest_title":      "🐛 Pest & Disease Risk This Season",
        "ca_pest_sub":        "Which pests to watch for when growing {crop} in {month}",
        "ca_pest_overall":    "Overall pest risk this month:",
        "ca_pest_none":       "✅ No major pest threats this month. Continue regular monitoring.",
        "ca_pest_symptoms":   "What you will see on the plant:",
        "ca_pest_danger":     "Why it is dangerous:",
        "ca_pest_organic":    "🌿 Organic control:",
        "ca_pest_prevent":    "🛡️ How to prevent:",
        "ca_pest_chemical":   "💊 Chemical control:",
        "ca_pest_source":     "Source: ICAR-NCIPM Integrated Pest Management guidelines",

        # Irrigation section
        "ca_irr_title":       "💧 Water & Irrigation Requirement",
        "ca_irr_sub":         "How much water does {crop} need on your {ha} ha field?",
        "ca_irr_need":        "Crop needs (total)",
        "ca_irr_rain":        "Rain will provide",
        "ca_irr_deficit":     "You need to irrigate",
        "ca_irr_rounds":      "Irrigation rounds",
        "ca_irr_ok":          "✅ Rainfall is sufficient this season. No irrigation cost needed.",
        "ca_irr_moderate":    "💧 You need {rounds} irrigation rounds ({hours} pump hours). Estimated electricity cost: ₹{cost} for the season.",
        "ca_irr_high":        "⚠️ High water deficit ({mm}mm). Need {rounds} rounds — estimated cost ₹{cost}. Consider PMKSY drip irrigation — 55% subsidy available.",
        "ca_irr_source":      "Source: FAO Irrigation Paper 56 · ICAR Water Use Efficiency Guidelines",

        # Carbon section
        "ca_carbon_title":    "♻️ Carbon Footprint of Your Farm",
        "ca_carbon_sub":      "How environment-friendly is this crop choice?",
        "ca_carbon_produces": "Your {crop} on {ha} ha produces",
        "ca_carbon_equiv":    "That is like driving a car for {km} km",
        "ca_carbon_rating":   "Sustainability rating",
        "ca_carbon_source":   "Source: IPCC 2006 Guidelines Vol.4 — Agriculture. Tier 1 emission factors.",

        # SHAP section
        "ca_shap_title":      "🔬 Why did AI recommend this crop?",
        "ca_shap_sub":        "Plain-English explanation of what your soil data tells the AI",
        "ca_shap_expand":     "Click to see detailed AI reasoning",
        "ca_shap_summary":    "In simple words — why this crop was recommended:",
        "ca_shap_positive":   "✅ Favourable factors:",
        "ca_shap_negative":   "⚠️ Minor concerns:",
        "ca_shap_but":        "— but overall still the best choice",
        "ca_shap_xlabel":     "Influence on AI decision (Green = pushes toward this crop, Red = pushes away)",
        "ca_shap_source":     "Technique: SHAP (SHapley Additive exPlanations) — Lundberg & Lee, NeurIPS 2017.",

        # History section
        "ca_history_title":   "📋 Your Prediction History",
        "ca_history_sub":     "Track your farm advisory sessions",
        "ca_history_expand":  "📋 View my past predictions",
        "ca_history_first":   "This is your first prediction — it has been saved!",
        "ca_history_total":   "Total sessions",
        "ca_history_top":     "Most recommended",
        "ca_history_conf":    "Avg confidence",
        "ca_db_missing":      "Database not available. Place utils/database.py in your project.",

        # Data trust panel
        "ca_trust_title":     "📂 Where does this data come from? (Full transparency)",
        "ca_trust_info":      "Information",
        "ca_trust_source":    "Source",
        "ca_trust_authority": "Authority",

        # Disease Detector
        "dd_title":           "🍃 Plant Disease Detector",
        "dd_subtitle":        "Upload a photo of your crop leaf — AI identifies the disease and suggests treatment",
        "dd_upload":          "Upload Leaf Photo",
        "dd_upload_label":    "Choose a leaf image",
        "dd_upload_help":     "Clear photo, single leaf, plain background = best results",
        "dd_supported":       "Supported crops:",
        "dd_placeholder":     "Upload a leaf photo to detect disease",
        "dd_placeholder_sub": "Supports: Apple, Tomato, Potato, Corn, Grape, Pepper and more",
        "dd_analysing":       "Analysing leaf image with CNN...",
        "dd_result":          "🏷️ Detection Result",
        "dd_detected":        "Detected Disease",
        "dd_crop":            "Crop:",
        "dd_confidence":      "Confidence:",
        "dd_saved":           "✅ Detection saved to database.",
        "dd_gradcam":         "🔍 Show AI attention heatmap (Grad-CAM) — see WHERE on the leaf the model looked",
        "dd_gradcam_title":   "🔍 Grad-CAM — AI Attention Map",
        "dd_gradcam_legend":  "🔴 Red = model focused here most · 🔵 Blue = model ignored this area",
        "dd_original":        "Original leaf",
        "dd_heatmap":         "AI attention heatmap",
        "dd_treatment":       "### 💊 Treatment & Prevention",
        "dd_tab_organic":     "🌿 Organic Treatment",
        "dd_tab_chemical":    "💉 Chemical Treatment",
        "dd_tab_prevention":  "🛡️ Prevention",
        "dd_spread_rate":     "⚡ How fast it spreads:",
        "dd_dosage":          "💊 Exact dosage:",
        "dd_cost":            "💰 Treatment cost:",
        "dd_recovery":        "⏱️ Days to recovery:",
        "dd_no_spray":        "⚠️ Do not spray within:",
        "dd_cause":           "Cause:",
        "dd_source":          "Source:",
        "dd_no_info":         "Detailed treatment info not found in disease_info.json.",
        "dd_confidence_chart":"📊 Prediction Confidence",
        "dd_model_source":    "Model: MobileNetV2 (transfer learning) · Dataset: PlantVillage — Hughes & Salathé 2015 · 54,306 images · 38 classes",

        # NDVI Monitor
        "ndvi_title":         "🛰️ Satellite Farm Health Monitor",
        "ndvi_subtitle":      "Real satellite data · Shows whether your district has good or stressed farming conditions",
        "ndvi_district":      "Your district",
        "ndvi_status_title":  "🌾 Your Farm Health Status Right Now",
        "ndvi_chi_label":     "Crop Health Index (CHI)",
        "ndvi_vs_avg":        "vs historical average",
        "ndvi_better":        "Better than usual",
        "ndvi_worse":         "Worse than usual",
        "ndvi_rainfall":      "Rainfall this month",
        "ndvi_temp":          "Temperature",
        "ndvi_solar":         "Sunlight (solar radiation)",
        "ndvi_trend_title":   "📈 How farm health has changed over the last 12 months",
        "ndvi_trend_sub":     "When the green line is high — conditions are good for farming. When it drops — there is drought, heat, or low sunlight stress.",
        "ndvi_chi_y_label":   "Crop Health Score (0 = very bad, 1 = perfect)",
        "ndvi_stress_title":  "🔍 What is causing stress? (Last 12 months)",
        "ndvi_stress_sub":    "Taller blue bar = more drought. Taller red bar = more heat. When both are low = good farming month.",
        "ndvi_drought":       "☔ Drought / Low rainfall",
        "ndvi_heat":          "🌡️ Heat / High temperature",
        "ndvi_heatmap_title": "📅 Year-by-year farm health calendar",
        "ndvi_heatmap_sub":   "Dark green = great conditions · Dark red = bad conditions. Which months are usually good for your crops?",
        "ndvi_advisory":      "🌾 What should you do right now?",
        "ndvi_data_ok":       "✅ Live satellite climate data loaded successfully",
        "ndvi_data_fallback": "Satellite APIs temporarily unavailable. Showing estimated data based on IMD climate normals.",

        # Fertilizer Optimizer
        "fo_title":           "🧪 Fertilizer Optimizer",
        "fo_subtitle":        "Enter your Soil Health Card values → get exact fertilizer bags to buy and savings vs average farmer",
        "fo_crop":            "Crop",
        "fo_field":           "Field size (hectares)",
        "fo_nitrogen":        "Nitrogen — N (kg/ha)",
        "fo_phosphorus":      "Phosphorus — P (kg/ha)",
        "fo_potassium":       "Potassium — K (kg/ha)",
        "fo_calc_btn":        "🧪 Calculate Fertilizer Need",
        "fo_savings":         "You save vs average farmer",
        "fo_soil_status":     "Current soil status:",
        "fo_buy_title":       "Fertilizer to buy for {ha} ha of {crop}:",
        "fo_total_cost":      "Total fertilizer cost",
        "fo_source":          "Source: ICAR Crop Production Guide · State Agriculture Department",

        # Irrigation Calculator
        "ic_title":           "💧 Irrigation Water Calculator",
        "ic_subtitle":        "Calculate exact water requirement, pump hours, and irrigation cost for your crop",
        "ic_crop":            "Crop",
        "ic_field":           "Field size (hectares)",
        "ic_district":        "District",
        "ic_month":           "Sowing month",
        "ic_pump":            "Pump horsepower (HP)",
        "ic_rate":            "Electricity rate (₹/kWh)",
        "ic_calc_btn":        "💧 Calculate Irrigation Requirement",
        "ic_need":            "Crop water need",
        "ic_rain_covers":     "Rainfall covers",
        "ic_deficit":         "Irrigation deficit",
        "ic_rounds":          "Irrigation rounds",
        "ic_pump_hrs":        "Pump hours",
        "ic_cost":            "Electricity cost",
        "ic_ok":              "✅ Rainfall sufficient — no irrigation needed this season.",
        "ic_source":          "Source: FAO Irrigation Paper 56 · ICAR Water Use Efficiency Guidelines",

        # Price Forecaster
        "pf_title":           "📈 Mandi Price Forecaster",
        "pf_subtitle":        "AI-powered 30-day price prediction · Same technology Meta uses for forecasting",
        "pf_crop":            "Crop",
        "pf_district":        "District (mandi location)",
        "pf_qty":             "Your stock (quintals)",
        "pf_horizon":         "Forecast horizon (days)",
        "pf_btn":             "📈 Generate Price Forecast",
        "pf_recommendation":  "AI Recommendation",
        "pf_current":         "Current price",
        "pf_peak":            "Forecast peak",
        "pf_gain":            "Potential gain",
        "pf_sell_now":        "SELL NOW",
        "pf_wait":            "WAIT",
        "pf_extra_earn":      "💰 If you wait {days} days: earn ₹{extra} more on {qty} quintals",
        "pf_source":          "Source: Agmarknet — Ministry of Agriculture, Govt of India",

        # Pest Monitor
        "pm_title":           "🐛 Pest Outbreak Risk Predictor",
        "pm_subtitle":        "Proactive pest advisory — warns you BEFORE damage appears",
        "pm_crop":            "Crop",
        "pm_month":           "Month",
        "pm_temp":            "Temperature (°C)",
        "pm_rain":            "Monthly Rainfall (mm)",
        "pm_btn":             "🔍 Check Pest Risk",
        "pm_overall":         "Pest Risk Assessment",
        "pm_no_pest":         "✅ No significant pest threats this month. Continue regular monitoring.",
        "pm_risk_high":       "🔴 HIGH RISK",
        "pm_risk_med":        "🟡 MEDIUM RISK",
        "pm_risk_low":        "🟢 LOW RISK",
        "pm_calendar":        "📅 12-Month Pest Risk Calendar",
        "pm_source":          "Source: ICAR-NCIPM · State Agriculture Department IPM calendars",

        # SHAP feature names (for charts)
        "feat_N":             "Nitrogen in soil",
        "feat_P":             "Phosphorus in soil",
        "feat_K":             "Potassium in soil",
        "feat_temperature":   "Temperature",
        "feat_humidity":      "Humidity",
        "feat_ph":            "Soil pH (acidity)",
        "feat_rainfall":      "Rainfall",

        # Common
        "loading":            "Loading...",
        "error_model":        "❌ Model not found. Run training script first.",
        "source_label":       "Source:",
        "save_ok":            "✅ Saved to database.",
        "no_data":            "No data available.",
        "months": ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"],
    },

    # ────────────────────────────────────────────────────────
    # HINDI
    # ────────────────────────────────────────────────────────
    "hi": {

        # App-wide
        "app_title":          "AgriSense India",
        "app_tagline":        "भारतीय किसानों के लिए बुद्धिमान फसल सलाह प्रणाली",
        "language_label":     "🌐 Language / भाषा",
        "built_by":           "निर्मित: MIT CSN नागपुर · द्वितीय वर्ष CS",

        # Navigation
        "nav_crop_advisor":   "🌱 फसल सलाहकार",
        "nav_disease":        "🍃 रोग पहचानकर्ता",
        "nav_carbon":         "♻️ कार्बन फुटप्रिंट",
        "nav_rotation":       "🗓️ फसल चक्र योजना",
        "nav_india_map":      "🗺️ भारत फसल नक्शा",
        "nav_model":          "📊 मॉडल अंतर्दृष्टि",
        "nav_price":          "📈 मंडी मूल्य पूर्वानुमान",
        "nav_fertilizer":     "🧪 उर्वरक अनुकूलक",
        "nav_grader":         "🍎 उपज ग्रेडर",
        "nav_ndvi":           "🛰️ उपग्रह निगरानी",
        "nav_pest":           "🐛 कीट निगरानी",
        "nav_irrigation":     "💧 सिंचाई कैलकुलेटर",

        # Crop Advisor page
        "ca_title":           "🌱 फसल सलाहकार",
        "ca_subtitle":        "अपनी मिट्टी की जानकारी दर्ज करें — फसल की सिफारिश, कमाई, कीट खतरा, सिंचाई और सरकारी योजनाएं एक जगह पाएं।",
        "ca_step1":           "📍 आपका स्थान",
        "ca_step1_sub":       "जिला चुनें — मौसम की जानकारी अपने आप भर जाएगी",
        "ca_district":        "जिला / तालुका",
        "ca_field_size":      "आपके खेत का आकार (हेक्टेयर में)",
        "ca_field_help":      "1 हेक्टेयर = 2.47 एकड़",
        "ca_weather_btn":     "🌤️ मेरे जिले का मौसम देखें",
        "ca_weather_ok":      "✅ मौसम की जानकारी मिली",
        "ca_weather_fail":    "मौसम सर्वर उपलब्ध नहीं। नीचे मैन्युअल रूप से दर्ज करें।",
        "ca_temperature":     "तापमान",
        "ca_humidity":        "नमी",
        "ca_weather_source":  "स्रोत: OpenWeatherMap · बटन दबाने पर अपडेट होता है",
        "ca_step2":           "🧪 आपकी मिट्टी की जानकारी",
        "ca_step2_sub":       "N, P, K, pH मान अपने सॉइल हेल्थ कार्ड (सरकारी हरे कार्ड) से लें",
        "ca_nitrogen":        "नाइट्रोजन — N (kg/ha)",
        "ca_nitrogen_help":   "📋 सॉइल हेल्थ कार्ड → 'उपलब्ध N'। सामान्य: 50–120 kg/ha",
        "ca_phosphorus":      "फास्फोरस — P (kg/ha)",
        "ca_phosphorus_help": "📋 सॉइल हेल्थ कार्ड → 'उपलब्ध P'। सामान्य: 20–80 kg/ha",
        "ca_potassium":       "पोटेशियम — K (kg/ha)",
        "ca_potassium_help":  "📋 सॉइल हेल्थ कार्ड → 'उपलब्ध K'। सामान्य: 100–200 kg/ha",
        "ca_ph":              "मिट्टी का pH (अम्लता स्तर)",
        "ca_ph_help":         "7.0 = तटस्थ। 7 से कम = अम्लीय। अधिकांश फसलें 6.0–7.5 पसंद करती हैं",
        "ca_temp_input":      "तापमान (°C)",
        "ca_humid_input":     "नमी (%)",
        "ca_rainfall":        "इस मौसम की अपेक्षित वर्षा (mm)",
        "ca_rainfall_help":   "IMD पूर्वानुमान या अपने जिले की पिछले साल की बारिश देखें",
        "ca_predict_btn":     "🔮  मेरी पूरी खेती सलाह पाएं",
        "ca_predict_info":    "👆 ऊपर मिट्टी की जानकारी भरें और बटन दबाएं।",
        "ca_loading":         "🌾 AI आपकी मिट्टी और जलवायु का विश्लेषण कर रहा है...",
        "ca_model_error":     "❌ AI मॉडल लोड नहीं हो सका। पहले notebooks/05_crop_model.py चलाएं।",

        # Recommendation
        "ca_rec_title":       "🏆 AI फसल सिफारिश",
        "ca_rec_sub":         "आपकी मिट्टी, मौसम और 2,200 खेती रिकॉर्ड के आधार पर",
        "ca_best_crop":       "आपके खेत के लिए सबसे अच्छी फसल",
        "ca_alternatives":    "अन्य फसलें जो भी काम कर सकती हैं:",
        "ca_conf_high":       "बहुत आत्मविश्वासी",
        "ca_conf_med":        "आत्मविश्वासी",
        "ca_conf_low":        "मध्यम आत्मविश्वास",
        "ca_model_source":    "📊 मॉडल: Random Forest (99.3% सटीकता) · स्रोत: Atharva Ingle, Kaggle",

        # Yield
        "ca_yield_title":     "💰 अपेक्षित उपज और कमाई",
        "ca_yield_label":     "अपेक्षित उपज",
        "ca_price_label":     "बाजार मूल्य",
        "ca_gross_label":     "कुल आमदनी",
        "ca_net_label":       "अनुमानित शुद्ध लाभ",
        "ca_after_cost":      "खेती लागत के बाद",
        "ca_price_source":    "बाजार मूल्य स्रोत:",
        "ca_low_profit":      "⚠️ वर्तमान मूल्यों पर यह फसल लाभदायक नहीं हो सकती। दूसरी या तीसरी सिफारिश देखें या PMFBY बीमा जांचें।",

        # Schemes
        "ca_schemes_title":   "🏛️ सरकारी योजनाएं जिनके लिए आप पात्र हो सकते हैं",
        "ca_schemes_sub":     "आपकी फसल, जिले और खेत के आकार के आधार पर स्वचालित जांच",
        "ca_schemes_found":   "✅ आप {n} सरकारी योजनाओं के लिए संभवतः पात्र हैं",
        "ca_what_you_get":    "आपको क्या मिलेगा:",
        "ca_how_apply":       "आवेदन कैसे करें:",
        "ca_docs_needed":     "आवश्यक दस्तावेज़:",
        "ca_schemes_note":    "स्रोत: कृषि और किसान कल्याण मंत्रालय, भारत सरकार · आधिकारिक पोर्टल पर सत्यापित करें",

        # Pest
        "ca_pest_title":      "🐛 इस मौसम में कीट और रोग का खतरा",
        "ca_pest_sub":        "{month} में {crop} उगाने पर कौन से कीटों का ध्यान रखें",
        "ca_pest_overall":    "इस महीने कीट का कुल खतरा:",
        "ca_pest_none":       "✅ इस महीने कोई बड़ा कीट खतरा नहीं। नियमित निगरानी जारी रखें।",
        "ca_pest_symptoms":   "पौधे पर क्या दिखेगा:",
        "ca_pest_danger":     "यह खतरनाक क्यों है:",
        "ca_pest_organic":    "🌿 जैविक नियंत्रण:",
        "ca_pest_prevent":    "🛡️ बचाव कैसे करें:",
        "ca_pest_chemical":   "💊 रासायनिक नियंत्रण:",
        "ca_pest_source":     "स्रोत: ICAR-NCIPM एकीकृत कीट प्रबंधन दिशानिर्देश",

        # Irrigation
        "ca_irr_title":       "💧 पानी और सिंचाई की आवश्यकता",
        "ca_irr_sub":         "आपके {ha} हेक्टेयर खेत में {crop} को कितना पानी चाहिए?",
        "ca_irr_need":        "फसल को चाहिए (कुल)",
        "ca_irr_rain":        "बारिश से मिलेगा",
        "ca_irr_deficit":     "सिंचाई करनी होगी",
        "ca_irr_rounds":      "सिंचाई के चक्र",
        "ca_irr_ok":          "✅ इस मौसम में बारिश पर्याप्त है। सिंचाई की जरूरत नहीं।",
        "ca_irr_moderate":    "💧 आपको {rounds} सिंचाई चक्र चाहिए ({hours} पंप घंटे)। अनुमानित बिजली लागत: ₹{cost}",
        "ca_irr_high":        "⚠️ पानी की कमी अधिक है ({mm}mm)। {rounds} चक्र चाहिए — अनुमानित लागत ₹{cost}। PMKSY ड्रिप सिंचाई पर 55% सब्सिडी उपलब्ध।",
        "ca_irr_source":      "स्रोत: FAO सिंचाई पेपर 56 · ICAR जल उपयोग दक्षता दिशानिर्देश",

        # Carbon
        "ca_carbon_title":    "♻️ आपके खेत का कार्बन फुटप्रिंट",
        "ca_carbon_sub":      "यह फसल कितनी पर्यावरण के अनुकूल है?",
        "ca_carbon_produces": "आपका {ha} हेक्टेयर में {crop} उत्पन्न करता है",
        "ca_carbon_equiv":    "यह कार से {km} किलोमीटर चलाने के बराबर है",
        "ca_carbon_rating":   "स्थिरता रेटिंग",
        "ca_carbon_source":   "स्रोत: IPCC 2006 दिशानिर्देश खंड 4 — कृषि। Tier 1 उत्सर्जन कारक।",

        # SHAP
        "ca_shap_title":      "🔬 AI ने यह फसल क्यों सुझाई?",
        "ca_shap_sub":        "आपकी मिट्टी AI को क्या बताती है — सरल भाषा में स्पष्टीकरण",
        "ca_shap_expand":     "विस्तृत AI तर्क देखने के लिए क्लिक करें",
        "ca_shap_summary":    "सरल शब्दों में — यह फसल क्यों सुझाई गई:",
        "ca_shap_positive":   "✅ अनुकूल कारक:",
        "ca_shap_negative":   "⚠️ मामूली चिंताएं:",
        "ca_shap_but":        "— लेकिन कुल मिलाकर यही सबसे अच्छा विकल्प है",
        "ca_shap_xlabel":     "AI निर्णय पर प्रभाव (हरा = इस फसल की ओर, लाल = दूर धकेलता है)",
        "ca_shap_source":     "तकनीक: SHAP (SHapley Additive exPlanations) — Lundberg & Lee, NeurIPS 2017",

        # History
        "ca_history_title":   "📋 आपकी भविष्यवाणी का इतिहास",
        "ca_history_sub":     "अपने खेती सलाह सत्रों को ट्रैक करें",
        "ca_history_expand":  "📋 मेरी पिछली भविष्यवाणियां देखें",
        "ca_history_first":   "यह आपकी पहली भविष्यवाणी है — सहेज ली गई है!",
        "ca_history_total":   "कुल सत्र",
        "ca_history_top":     "सबसे अनुशंसित",
        "ca_history_conf":    "औसत आत्मविश्वास",
        "ca_db_missing":      "डेटाबेस उपलब्ध नहीं। utils/database.py अपने प्रोजेक्ट में रखें।",

        # Data trust
        "ca_trust_title":     "📂 यह डेटा कहां से आता है? (पूर्ण पारदर्शिता)",
        "ca_trust_info":      "जानकारी",
        "ca_trust_source":    "स्रोत",
        "ca_trust_authority": "प्राधिकरण",

        # Disease Detector
        "dd_title":           "🍃 पौधा रोग पहचानकर्ता",
        "dd_subtitle":        "अपनी फसल की पत्ती की फोटो अपलोड करें — AI रोग पहचानेगा और उपचार बताएगा",
        "dd_upload":          "पत्ती की फोटो अपलोड करें",
        "dd_upload_label":    "छवि चुनें",
        "dd_upload_help":     "साफ फोटो, एक पत्ती, सादी पृष्ठभूमि = सबसे अच्छे परिणाम",
        "dd_supported":       "समर्थित फसलें:",
        "dd_placeholder":     "रोग पहचानने के लिए पत्ती की फोटो अपलोड करें",
        "dd_placeholder_sub": "समर्थन: सेब, टमाटर, आलू, मक्का, अंगूर, मिर्च और अधिक",
        "dd_analysing":       "CNN से पत्ती की छवि का विश्लेषण हो रहा है...",
        "dd_result":          "🏷️ पहचान परिणाम",
        "dd_detected":        "पहचाना गया रोग",
        "dd_crop":            "फसल:",
        "dd_confidence":      "विश्वास:",
        "dd_saved":           "✅ पहचान डेटाबेस में सहेजी गई।",
        "dd_gradcam":         "🔍 AI ध्यान हीटमैप दिखाएं (Grad-CAM) — देखें AI ने पत्ती पर कहां देखा",
        "dd_gradcam_title":   "🔍 Grad-CAM — AI ध्यान मानचित्र",
        "dd_gradcam_legend":  "🔴 लाल = AI ने यहां सबसे अधिक ध्यान दिया · 🔵 नीला = AI ने यहां ध्यान नहीं दिया",
        "dd_original":        "मूल पत्ती",
        "dd_heatmap":         "AI ध्यान हीटमैप",
        "dd_treatment":       "### 💊 उपचार और रोकथाम",
        "dd_tab_organic":     "🌿 जैविक उपचार",
        "dd_tab_chemical":    "💉 रासायनिक उपचार",
        "dd_tab_prevention":  "🛡️ रोकथाम",
        "dd_spread_rate":     "⚡ यह कितनी जल्दी फैलता है:",
        "dd_dosage":          "💊 सटीक खुराक:",
        "dd_cost":            "💰 उपचार की लागत:",
        "dd_recovery":        "⏱️ ठीक होने में दिन:",
        "dd_no_spray":        "⚠️ फसल काटने से पहले कितने दिन न छिड़कें:",
        "dd_cause":           "कारण:",
        "dd_source":          "स्रोत:",
        "dd_no_info":         "disease_info.json में विस्तृत जानकारी नहीं मिली।",
        "dd_confidence_chart":"📊 पूर्वानुमान आत्मविश्वास",
        "dd_model_source":    "मॉडल: MobileNetV2 · डेटासेट: PlantVillage — Hughes & Salathé 2015 · 54,306 छवियां · 38 वर्ग",

        # NDVI
        "ndvi_title":         "🛰️ उपग्रह खेत स्वास्थ्य निगरानी",
        "ndvi_subtitle":      "वास्तविक उपग्रह डेटा · आपके जिले में अच्छी या तनावग्रस्त खेती स्थितियां दिखाता है",
        "ndvi_district":      "आपका जिला",
        "ndvi_status_title":  "🌾 अभी आपके खेत का स्वास्थ्य",
        "ndvi_chi_label":     "फसल स्वास्थ्य सूचकांक (CHI)",
        "ndvi_vs_avg":        "ऐतिहासिक औसत की तुलना में",
        "ndvi_better":        "सामान्य से बेहतर",
        "ndvi_worse":         "सामान्य से खराब",
        "ndvi_rainfall":      "इस महीने वर्षा",
        "ndvi_temp":          "तापमान",
        "ndvi_solar":         "सूर्यप्रकाश (सौर विकिरण)",
        "ndvi_trend_title":   "📈 पिछले 12 महीनों में खेत का स्वास्थ्य",
        "ndvi_trend_sub":     "हरी रेखा ऊपर = खेती के लिए अच्छी स्थितियां। नीचे = सूखा, गर्मी या कम धूप का तनाव।",
        "ndvi_chi_y_label":   "फसल स्वास्थ्य अंक (0 = बहुत खराब, 1 = बिल्कुल सही)",
        "ndvi_stress_title":  "🔍 तनाव का कारण क्या है? (पिछले 12 महीने)",
        "ndvi_stress_sub":    "नीला ऊंचा = अधिक सूखा। लाल ऊंचा = अधिक गर्मी। दोनों नीचे = अच्छा खेती महीना।",
        "ndvi_drought":       "☔ सूखा / कम वर्षा",
        "ndvi_heat":          "🌡️ गर्मी / उच्च तापमान",
        "ndvi_heatmap_title": "📅 साल-दर-साल खेत स्वास्थ्य कैलेंडर",
        "ndvi_heatmap_sub":   "गहरा हरा = बढ़िया स्थितियां · गहरा लाल = खराब स्थितियां। कौन से महीने आपकी फसल के लिए आम तौर पर अच्छे हैं?",
        "ndvi_advisory":      "🌾 अभी आपको क्या करना चाहिए?",
        "ndvi_data_ok":       "✅ उपग्रह डेटा सफलतापूर्वक लोड हुआ",
        "ndvi_data_fallback": "उपग्रह APIs अस्थायी रूप से अनुपलब्ध। IMD जलवायु नॉर्मल पर आधारित अनुमानित डेटा दिखाया जा रहा है।",

        # Fertilizer Optimizer
        "fo_title":           "🧪 उर्वरक अनुकूलक",
        "fo_subtitle":        "सॉइल हेल्थ कार्ड मान दर्ज करें → खरीदने के लिए सटीक उर्वरक बोरे और औसत किसान की तुलना में बचत पाएं",
        "fo_crop":            "फसल",
        "fo_field":           "खेत का आकार (हेक्टेयर)",
        "fo_nitrogen":        "नाइट्रोजन — N (kg/ha)",
        "fo_phosphorus":      "फास्फोरस — P (kg/ha)",
        "fo_potassium":       "पोटेशियम — K (kg/ha)",
        "fo_calc_btn":        "🧪 उर्वरक आवश्यकता की गणना करें",
        "fo_savings":         "औसत किसान की तुलना में आपकी बचत",
        "fo_soil_status":     "वर्तमान मिट्टी की स्थिति:",
        "fo_buy_title":       "{ha} हेक्टेयर {crop} के लिए खरीदने योग्य उर्वरक:",
        "fo_total_cost":      "कुल उर्वरक लागत",
        "fo_source":          "स्रोत: ICAR फसल उत्पादन गाइड · राज्य कृषि विभाग",

        # Irrigation Calculator
        "ic_title":           "💧 सिंचाई जल कैलकुलेटर",
        "ic_subtitle":        "अपनी फसल के लिए सटीक जल आवश्यकता, पंप घंटे और सिंचाई लागत की गणना करें",
        "ic_crop":            "फसल",
        "ic_field":           "खेत का आकार (हेक्टेयर)",
        "ic_district":        "जिला",
        "ic_month":           "बुवाई महीना",
        "ic_pump":            "पंप हॉर्सपावर (HP)",
        "ic_rate":            "बिजली दर (₹/kWh)",
        "ic_calc_btn":        "💧 सिंचाई आवश्यकता की गणना करें",
        "ic_need":            "फसल को पानी चाहिए",
        "ic_rain_covers":     "वर्षा से मिलेगा",
        "ic_deficit":         "सिंचाई की कमी",
        "ic_rounds":          "सिंचाई के चक्र",
        "ic_pump_hrs":        "पंप घंटे",
        "ic_cost":            "बिजली लागत",
        "ic_ok":              "✅ वर्षा पर्याप्त — इस मौसम में सिंचाई की जरूरत नहीं।",
        "ic_source":          "स्रोत: FAO सिंचाई पेपर 56 · ICAR जल उपयोग दक्षता दिशानिर्देश",

        # Price Forecaster
        "pf_title":           "📈 मंडी मूल्य पूर्वानुमान",
        "pf_subtitle":        "AI-आधारित 30-दिन का मूल्य पूर्वानुमान",
        "pf_crop":            "फसल",
        "pf_district":        "जिला (मंडी स्थान)",
        "pf_qty":             "आपका स्टॉक (क्विंटल)",
        "pf_horizon":         "पूर्वानुमान अवधि (दिन)",
        "pf_btn":             "📈 मूल्य पूर्वानुमान तैयार करें",
        "pf_recommendation":  "AI सिफारिश",
        "pf_current":         "वर्तमान मूल्य",
        "pf_peak":            "पूर्वानुमानित उच्चतम",
        "pf_gain":            "संभावित लाभ",
        "pf_sell_now":        "अभी बेचें",
        "pf_wait":            "रुकें",
        "pf_extra_earn":      "💰 यदि आप {days} दिन रुकते हैं: {qty} क्विंटल पर ₹{extra} अधिक कमाएं",
        "pf_source":          "स्रोत: Agmarknet — कृषि मंत्रालय, भारत सरकार",

        # Pest Monitor
        "pm_title":           "🐛 कीट प्रकोप जोखिम भविष्यवक्ता",
        "pm_subtitle":        "सक्रिय कीट सलाह — नुकसान होने से पहले चेतावनी देता है",
        "pm_crop":            "फसल",
        "pm_month":           "महीना",
        "pm_temp":            "तापमान (°C)",
        "pm_rain":            "मासिक वर्षा (mm)",
        "pm_btn":             "🔍 कीट जोखिम जांचें",
        "pm_overall":         "कीट जोखिम मूल्यांकन",
        "pm_no_pest":         "✅ इस महीने कोई बड़ा कीट खतरा नहीं। नियमित निगरानी जारी रखें।",
        "pm_risk_high":       "🔴 उच्च जोखिम",
        "pm_risk_med":        "🟡 मध्यम जोखिम",
        "pm_risk_low":        "🟢 कम जोखिम",
        "pm_calendar":        "📅 12-महीने का कीट जोखिम कैलेंडर",
        "pm_source":          "स्रोत: ICAR-NCIPM · राज्य कृषि विभाग IPM कैलेंडर",

        # SHAP feature names (for charts)
        "feat_N":             "मिट्टी में नाइट्रोजन",
        "feat_P":             "मिट्टी में फास्फोरस",
        "feat_K":             "मिट्टी में पोटेशियम",
        "feat_temperature":   "तापमान",
        "feat_humidity":      "नमी",
        "feat_ph":            "मिट्टी pH (अम्लता)",
        "feat_rainfall":      "वर्षा",

        # Common
        "loading":            "लोड हो रहा है...",
        "error_model":        "❌ मॉडल नहीं मिला। पहले ट्रेनिंग स्क्रिप्ट चलाएं।",
        "source_label":       "स्रोत:",
        "save_ok":            "✅ डेटाबेस में सहेजा गया।",
        "no_data":            "डेटा उपलब्ध नहीं।",
        "months": ["जनवरी","फरवरी","मार्च","अप्रैल","मई","जून",
                   "जुलाई","अगस्त","सितंबर","अक्टूबर","नवंबर","दिसंबर"],
    },
}


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — import these in every page
# ════════════════════════════════════════════════════════════

def get_lang() -> str:
    """
    Read current language from session state.
    Returns "en" or "hi". Defaults to "en".
    """
    import streamlit as st
    return st.session_state.get("lang", "en")


def lang_selector() -> str:
    """
    Show language toggle in sidebar.
    Call this in your sidebar block.

    Example:
        with st.sidebar:
            lang = lang_selector()

    Returns current language code ("en" or "hi").
    """
    import streamlit as st

    options    = {"English 🇬🇧": "en", "हिंदी 🇮🇳": "hi"}
    current    = st.session_state.get("lang", "en")
    current_lbl = "English 🇬🇧" if current == "en" else "हिंदी 🇮🇳"

    selected = st.radio(
        T["en"]["language_label"],
        options=list(options.keys()),
        index=list(options.keys()).index(current_lbl),
        horizontal=True,
        key="lang_radio",
    )
    st.session_state["lang"] = options[selected]
    return st.session_state["lang"]


def t(key: str, **kwargs) -> str:
    """
    Shorthand translation helper.
    Gets current lang from session_state automatically.

    Usage:
        from utils.translations import t
        st.title(t("ca_title"))
        st.info(t("ca_irr_moderate", rounds=3, hours=12, cost=660))
    """
    import streamlit as st
    lang = st.session_state.get("lang", "en")
    text = T[lang].get(key, T["en"].get(key, key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text