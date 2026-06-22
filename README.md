# 🌾 AgriSense India — Intelligent Crop Advisory System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agrisense-india.streamlit.app)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with Love](https://img.shields.io/badge/Made%20for-Indian%20Farmers-orange.svg)]()

> **AI-powered agricultural advisory platform for 600 million Indian farmers.**
> Recommends crops, predicts earnings, detects plant diseases, monitors satellite health,
> and connects farmers to government schemes — all in one deployed web app.

**[🚀 Live Demo](https://agrisense-india-gddbdagqt2cyllyjtmgvfw.streamlit.app/)** &nbsp;·&nbsp;


## 🎯 The Problem

India loses **30–33% of crop productivity annually** because farmers choose crops based on tradition, not soil and weather data. With **600 million farmers** and no unified advisory tool, wrong crop choices, late disease detection, and missed government schemes cost the agricultural sector over ₹1.5 lakh crore every year.

---

## ✅ Solution — 14 Features in One App

| # | Feature | What It Does | Technology |
|---|---------|-------------|------------|
| 1 | 🌱 **Crop Recommendation** | Top 3 crops ranked by confidence from soil + climate data | Random Forest, XGBoost |
| 2 | 💰 **Yield + Earnings** | Predicted kg/ha × live mandi price = gross earnings | XGBoost Regressor, Agmarknet API |
| 3 | 🔬 **SHAP Explainability** | Why was this crop recommended? Waterfall chart per prediction | SHAP TreeExplainer |
| 4 | 🍃 **Disease Detector** | Upload leaf photo → CNN detects 38 diseases across 14 crops | MobileNetV2, PlantVillage |
| 5 | 🛰️ **Grad-CAM Heatmap** | Shows WHERE on the leaf the AI looked to make its decision | Gradient-weighted CAM |
| 6 | ♻️ **Carbon Footprint** | CO₂e per crop cycle + comparison against alternatives | IPCC 2006 Tier 1 factors |
| 7 | 🗓️ **Rotation Planner** | 3-season calendar with N savings, sowing dates, seed rates | ICAR guidelines |
| 8 | 🌧️ **Weather Risk Alerts** | 5-day forecast + risk alerts + optimal sowing window | OpenWeatherMap API |
| 9 | 🏛️ **Govt Scheme Finder** | PM-KISAN, PMFBY, KCC eligibility auto-checked | JSON rules engine |
| 10 | 🗺️ **India Crop Map** | Interactive choropleth — dominant crop by state | Plotly, GeoJSON |
| 11 | 📈 **Price Forecaster** | 30-day mandi price prediction with sell/hold recommendation | Facebook Prophet |
| 12 | 🧪 **Fertilizer Optimizer** | Exact bags to buy + savings vs average farmer | ICAR NPK science |
| 13 | 🐛 **Pest Predictor** | Proactive pest risk alerts before damage appears | ICAR-NCIPM calendar |
| 14 | 💧 **Irrigation Calculator** | Exact water budget, pump hours, electricity cost | FAO Paper 56 |

---

## 🏗️ Architecture

```
Raw Data Sources        Data Pipeline           ML Layer              Web App
────────────────  →    ─────────────────  →   ──────────────   →   ─────────────
data.gov.in            01_data_cleaning        RF Classifier         14 Pages
Kaggle CSV             02_eda (6 charts)       XGBoost               Live APIs
OpenWeatherMap API     03_india_map            MobileNetV2 CNN       Streamlit Cloud
Agmarknet API          05_crop_model           XGBoost Regressor     GitHub CI
PlantVillage (54K)     06_yield_shap           SHAP Explainer
IPCC Tier 1            08_disease_cnn          Facebook Prophet
ICAR Guidelines        10_carbon_footprint     Grad-CAM
NASA/Open-Meteo        11_rotation_planner     Crop Health Index
```

---

## 🎓 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Random Forest (crop classifier) | Test Accuracy | **99.3%** |
| XGBoost (crop classifier) | Test Accuracy | **98.8%** |
| Both classifiers | 5-fold CV Score | **99.1%** |
| MobileNetV2 (disease CNN) | Val Accuracy | **≥ 85%** |
| XGBoost Regressor (yield) | R² Score | **0.87** |

---

## 📁 Project Structure

```
agrisense-india/
├── app.py                          ← Streamlit entry point
├── requirements.txt
├── .gitignore
├── README.md
├── diagnose_model.py               ← Run this if crop prediction is wrong
├── .streamlit/
│   └── config.toml
├── pages/
│   ├── 1_Crop_Advisor.py           ← Crop recommendation + all 10 sections
│   ├── 2_Disease_Detector.py       ← CNN disease detection + Grad-CAM
│   ├── 3_Carbon_Footprint.py       ← IPCC carbon estimator
│   ├── 4_Rotation_Planner.py       ← 3-season ICAR rotation calendar
│   ├── 5_India_Map.py              ← Plotly choropleth India map
│   ├── 6_Model_Insights.py         ← Accuracy metrics + data sources
│   ├── 11_Price_Forecaster.py      ← Facebook Prophet 30-day forecast
│   ├── 12_Produce_Grader.py        ← EfficientNetB0 Grade A/B/C
│   ├── 13_NDVI_Monitor.py          ← NASA/Open-Meteo satellite data
│   ├── 14_Pest_Monitor.py          ← ICAR-NCIPM pest calendar
│   └── 15_Irrigation_Calculator.py ← FAO Paper 56 water budget
├── utils/
│   ├── weather_api.py
│   ├── mandi_api.py
│   ├── carbon.py
│   ├── rotation.py
│   ├── database.py
│   ├── gradcam.py
│   ├── pest_advisor.py
│   ├── irrigation.py
│   ├── nasa_power.py
│   └── translations.py
├── models/
│   ├── rf_crop_model.pkl
│   ├── xgb_crop_model.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   ├── yield_model.pkl
│   └── yield_scaler.pkl
├── assets/
│   ├── style.css
│   ├── disease_info.json
│   ├── rotation_rules.json
│   ├── pest_calendar.json
│   └── india_map.html
├── data/
│   ├── raw/crop_recommendation.csv
│   └── processed/
└── notebooks/
    ├── 05_crop_model_fixed.py
    ├── 08_disease_cnn.py
    └── ...
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/chaitalipatil3136/agrisense-india.git
cd agrisense-india

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env and add your OpenWeatherMap API key:
# OPENWEATHER_API_KEY=your_key_here

# 4. Train models (if pkl files not present)
python notebooks/05_crop_model_fixed.py

# 5. Run the app
streamlit run app.py
```

### If Crop Advisor always predicts the same crop

```bash
# Run diagnosis first to find the root cause
python diagnose_model.py

# Then retrain from scratch
del models/*.pkl        # Windows
rm models/*.pkl         # Mac/Linux
python notebooks/05_crop_model_fixed.py
```

---

## 📊 Data Sources

| Dataset | Source | Records | License |
|---------|--------|---------|---------|
| Crop Recommendation | Atharva Ingle — Kaggle | 2,200 rows · 22 crops | Public domain |
| India Crop Production | Directorate of Economics & Statistics — data.gov.in | 246,000+ rows | OGD India (NDSAP) |
| Plant Disease Images | Hughes & Salathé (2015) — PlantVillage | 54,306 images · 38 classes | CC BY 4.0 |
| Carbon Emission Factors | IPCC 2006 Guidelines Vol.4 — Agriculture | 28 crops | Public domain |
| Crop Rotation Rules | ICAR Soil Fertility Management Guidelines | 15 crops | Public domain |
| Live Weather | OpenWeatherMap API | Real-time | CC BY-SA 4.0 |
| Live Mandi Prices | Agmarknet — Ministry of Agriculture | Daily updates | OGD India |
| Satellite Climate Data | Open-Meteo ERA5 Archive | 730 days | CC BY 4.0 |

---

## 🛠️ Tech Stack

**Data & ML**
`Python 3.10` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `SHAP`

**Deep Learning**
`TensorFlow 2.13` · `Keras` · `MobileNetV2` · `EfficientNetB0` · Transfer Learning · Grad-CAM

**Forecasting**
`Facebook Prophet` · Time series decomposition

**Visualization**
`matplotlib` · `seaborn` · `Plotly` · Choropleth maps

**APIs & Data**
`OpenWeatherMap` · `Agmarknet` · `Open-Meteo` · `NASA POWER` · `requests`

**App & Deploy**
`Streamlit 1.28` · `Streamlit Community Cloud` · `SQLite` · `GitHub`

---

## 📚 Scientific References

1. Atharva Ingle. *Crop Recommendation Dataset*. Kaggle, 2020.
2. Directorate of Economics & Statistics, MoAFW. *District-wise Season-wise Crop Production Statistics*. data.gov.in.
3. Hughes D.P. & Salathé M. *An open access repository of images on plant health*. arXiv:1511.08060, 2015.
4. IPCC. *2006 IPCC Guidelines for National GHG Inventories, Volume 4*. 2006.
5. ICAR. *Crop Rotation and Soil Fertility Management Guidelines*. New Delhi.
6. Howard A.G. et al. *MobileNets: Efficient CNNs for Mobile Vision Applications*. arXiv:1704.04861, 2017.
7. Lundberg S.M. & Lee S. *A Unified Approach to Interpreting Model Predictions (SHAP)*. NeurIPS 2017.
8. Selvaraju R.R. et al. *Grad-CAM: Visual Explanations from Deep Networks*. ICCV 2017.
9. Taylor S.J. & Letham B. *Forecasting at Scale (Prophet)*. The American Statistician, 2018.
10. Allen R.G. et al. *FAO Irrigation and Drainage Paper 56*. FAO, 1998.

---

## 👨‍💻 About

Built by a 2nd year Computer Science student from CSN, Maharashtra as a complete end-to-end data science portfolio project demonstrating: data collection → cleaning → EDA → classical ML → deep learning → explainable AI → time series forecasting → satellite data → deployment.

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.

---

*Built with ❤️ for Indian farmers · AgriSense India · MIT CSN CSN*