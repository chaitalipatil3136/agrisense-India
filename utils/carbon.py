"""
AgriSense India — Carbon Footprint Utility Module
File: utils/carbon.py
 
Reusable module for the Streamlit app.
Import: from utils.carbon import calculate_carbon, get_carbon_comparison
 
Science: IPCC 2006 Guidelines Vol.4 Tier 1 Emission Factors
"""
 
import pandas as pd
import os
 
_EF_PATH   = "data/processed/emission_factors.csv"
_ef_cache  = None
 
 
def _load_ef():
    global _ef_cache
    if _ef_cache is None:
        if not os.path.exists(_EF_PATH):
            raise FileNotFoundError(
                f"Emission factors CSV not found at {_EF_PATH}. "
                "Run notebooks/10_carbon_footprint.py first."
            )
        df = pd.read_csv(_EF_PATH)
        df.columns = df.columns.str.strip().str.lower()
        _ef_cache = df
    return _ef_cache
 
 
def calculate_carbon(crop: str, field_size_ha: float,
                     n_kg_per_ha: float, p_kg_per_ha: float,
                     k_kg_per_ha: float) -> dict:
    """
    Calculate CO2 equivalent emissions for one crop cycle.
 
    Parameters
    ----------
    crop          : Crop name (e.g. "cotton", "rice", "soybean")
    field_size_ha : Field area in hectares
    n_kg_per_ha   : Nitrogen fertilizer applied (kg per hectare)
    p_kg_per_ha   : Phosphorus fertilizer applied (kg per hectare)
    k_kg_per_ha   : Potassium fertilizer applied (kg per hectare)
 
    Returns
    -------
    dict with emission breakdown and totals
    """
    ef = _load_ef()
    crop_lower = str(crop).lower().strip()
    row = ef[ef["crop"] == crop_lower]
 
    if row.empty:
        n_f, p_f, k_f = 4.7, 1.0, 0.6
        land_f = float(ef["land_kgco2e_per_ha"].mean())
        seq_c  = 0.0
    else:
        r      = row.iloc[0]
        n_f    = float(r.get("n_kgco2e_per_kg", 4.7))
        p_f    = float(r.get("p_kgco2e_per_kg", 1.0))
        k_f    = float(r.get("k_kgco2e_per_kg", 0.6))
        land_f = float(r.get("land_kgco2e_per_ha", 650))
        seq_c  = float(r.get("sequestration_credit_kgco2e_per_ha", 0))
 
    n_em    = n_kg_per_ha  * n_f    * field_size_ha
    p_em    = p_kg_per_ha  * p_f    * field_size_ha
    k_em    = k_kg_per_ha  * k_f    * field_size_ha
    land_em = land_f        * field_size_ha
    seq_em  = -seq_c        * field_size_ha
    total   = n_em + p_em + k_em + land_em + seq_em
 
    return {
        "crop":               crop,
        "n_emission":         round(n_em,    1),
        "p_emission":         round(p_em,    1),
        "k_emission":         round(k_em,    1),
        "land_emission":      round(land_em, 1),
        "sequestration":      round(seq_em,  1),
        "total_kgco2e":       round(total,   1),
        "total_kgco2e_per_ha": round(total / max(field_size_ha, 0.001), 1),
        "total_tonnes_co2e":  round(total / 1000, 3),
        "equivalent_car_km":  round(total / 0.21, 0),
    }
 
 
def get_carbon_comparison(crop: str, field_ha: float,
                          n_kgha: float, p_kgha: float, k_kgha: float,
                          compare_crops: list = None) -> list:
    """
    Compare carbon footprint of selected crop against alternatives.
 
    Returns list of dicts sorted from lowest to highest emissions.
    Selected crop is marked with is_selected=True.
    """
    if compare_crops is None:
        compare_crops = ["rice", "wheat", "cotton", "soybean",
                         "maize", "groundnut", "chickpea"]
 
    results = []
    main_result = calculate_carbon(crop, field_ha, n_kgha, p_kgha, k_kgha)
    main_result["is_selected"] = True
    results.append(main_result)
 
    for c in compare_crops:
        if c.lower() != crop.lower():
            r = calculate_carbon(c, field_ha, n_kgha, p_kgha, k_kgha)
            r["is_selected"] = False
            results.append(r)
 
    return sorted(results, key=lambda x: x["total_kgco2e"])
 
 
def get_sustainability_rating(total_kgco2e_per_ha: float) -> tuple:
    """
    Returns (rating_label, color_hex) based on emissions per hectare.
    Used for the app's sustainability badge.
    """
    if total_kgco2e_per_ha < 500:
        return ("Excellent", "#1D9E75")
    elif total_kgco2e_per_ha < 800:
        return ("Good",      "#5DCAA5")
    elif total_kgco2e_per_ha < 1000:
        return ("Moderate",  "#EF9F27")
    elif total_kgco2e_per_ha < 1300:
        return ("High",      "#D85A30")
    else:
        return ("Very High", "#A32D2D")
