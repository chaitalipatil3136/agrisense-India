
"""
AgriSense India — NASA POWER Utility Module
File: utils/nasa_power.py
 
Fetches real agricultural climate data from NASA POWER API.
Free — no API key, no login, no registration required.
 
API: https://power.larc.nasa.gov/api/temporal/monthly/point
Community: AG (Agriculture) — crop-relevant parameters
Resolution: 0.5° × 0.5° grid (~55 km²)
Update frequency: Monthly averages, near-real-time
 
Parameters used:
  PRECTOTCORR    — Monthly precipitation (mm/day)
  ALLSKY_SFC_PAR_TOT — Photosynthetically Active Radiation (W/m²)
                       Direct measure of light available for crop photosynthesis
  T2M            — Temperature at 2m (°C)
  RH2M           — Relative humidity at 2m (%)
  WS2M           — Wind speed at 2m (m/s)
 
Import: from utils.nasa_power import get_crop_health_data, compute_chi
"""
 
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
 
# ── District coordinates (lat, lon) ──────────────────────────
DISTRICT_COORDS = {
    "Nagpur, Maharashtra":     (21.15,  79.09),
    "Pune, Maharashtra":       (18.52,  73.86),
    "Nashik, Maharashtra":     (20.01,  73.79),
    "Amravati, Maharashtra":   (20.93,  77.75),
    "Wardha, Maharashtra":     (20.75,  78.60),
    "Aurangabad, Maharashtra": (19.88,  75.34),
    "Latur, Maharashtra":      (18.40,  76.56),
    "Indore, MP":              (22.72,  75.86),
    "Bhopal, MP":              (23.26,  77.41),
    "Ludhiana, Punjab":        (30.90,  75.85),
    "Amritsar, Punjab":        (31.63,  74.87),
    "Jaipur, Rajasthan":       (26.91,  75.79),
    "Jodhpur, Rajasthan":      (26.29,  73.02),
    "Varanasi, UP":            (25.32,  83.01),
    "Patna, Bihar":            (25.59,  85.14),
    "Hyderabad, Telangana":    (17.38,  78.49),
    "Vijayawada, AP":          (16.51,  80.62),
    "Coimbatore, Tamil Nadu":  (11.00,  76.96),
    "Kolkata, West Bengal":    (22.57,  88.36),
    "Ahmedabad, Gujarat":      (23.03,  72.59),
}
 
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/monthly/point"
 
# NASA fill value for missing data
NASA_FILL_VALUE = -999.0
 
 
def get_nasa_power_data(
    lat: float,
    lon: float,
    start_year: int = None,
    end_year:   int = None,
    timeout:    int = 35,
) -> pd.DataFrame:
    """
    Fetch monthly agricultural climate data from NASA POWER API.
 
    Parameters
    ----------
    lat, lon    : Coordinates
    start_year  : Start year (default: 5 years ago)
    end_year    : End year (default: current year)
    timeout     : Request timeout in seconds
 
    Returns
    -------
    DataFrame with columns: date, rainfall_mm, par_wm2,
                            temperature_c, humidity_pct, wind_ms
    """
    if end_year is None:
        end_year = datetime.now().year
    if start_year is None:
        start_year = end_year - 4    # 5 years of data
 
    params = {
       "parameters": "PRECTOTCORR,ALLSKY_SFC_PAR_TOT,T2M",
                       "community":  "AG",
        "longitude":  round(lon, 4),
        "latitude":   round(lat, 4),
        "start":      str(start_year),
        "end":        str(end_year),
        "format":     "JSON",
    }
 
    try:
        resp = requests.get(NASA_POWER_BASE, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        raise TimeoutError(
            "NASA POWER API timed out. Try again in a moment — "
            "the server can be slow during peak hours."
        )
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot reach NASA POWER API. Check your internet connection.")
    except Exception as e:
        raise Exception(f"NASA API Error: {e}")
 
    # ── Parse response ────────────────────────────────────────
    try:
        props = data["properties"]["parameter"]
    except KeyError:
        raise ValueError("Unexpected NASA POWER API response format.")
 
    rain_data = props.get("PRECTOTCORR", {})
    par_data  = props.get("ALLSKY_SFC_PAR_TOT", {})
    temp_data = props.get("T2M", {})
    hum_data  = {}
    wind_data = {}
 
    records = []
    for yyyymm in sorted(rain_data.keys()):
        if len(yyyymm) != 6 or not yyyymm.isdigit():
            continue
        year  = int(yyyymm[:4])
        month = int(yyyymm[4:])
        try:
            date = pd.Timestamp(year=year, month=month, day=15)
        except ValueError:
            continue
 
        rain = float(rain_data.get(yyyymm, NASA_FILL_VALUE))
        par  = float(par_data.get(yyyymm,  NASA_FILL_VALUE))
        temp = float(temp_data.get(yyyymm, NASA_FILL_VALUE))
        hum  = np.nan
        wind = np.nan
 
        records.append({
            "date":           date,
            "year":           year,
            "month":          month,
            "rainfall_mm":    rain if rain  > NASA_FILL_VALUE else np.nan,
            "par_wm2":        par  if par   > NASA_FILL_VALUE else np.nan,
            "temperature_c":  temp if temp  > NASA_FILL_VALUE else np.nan,
            "humidity_pct":   hum  if hum   > NASA_FILL_VALUE else np.nan,
            "wind_ms":        wind if wind  > NASA_FILL_VALUE else np.nan,
        })
 
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("NASA POWER returned no usable data for this location.")
 
    # Interpolate any remaining NaN values
    numeric_cols = ["rainfall_mm", "par_wm2", "temperature_c", "humidity_pct", "wind_ms"]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
 
    # Convert daily rainfall to monthly total
    df["rainfall_monthly_mm"] = df["rainfall_mm"] * 30
 
    return df.sort_values("date").reset_index(drop=True)
 
 
def compute_chi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Crop Health Index (CHI) for each month.
 
    CHI formula:
      CHI = PAR_norm × (1 - drought_stress) × (1 - heat_stress)
 
    Where:
      PAR_norm      = PAR / PAR_max  (normalized light availability)
      drought_stress = sigmoid of rainfall deficit vs. optimal
      heat_stress   = sigmoid of temperature excess vs. optimal
 
    CHI range: 0 (severe stress) to 1 (optimal conditions)
    Scientific basis: Derived from FAO crop water stress and
    heat stress functions (FAO Irrigation and Drainage Paper 56)
    """
    df = df.copy()
 
    # ── PAR normalization ─────────────────────────────────────
    par_max = df["par_wm2"].quantile(0.95)
    if par_max <= 0:
        par_max = 200.0
    df["par_norm"] = (df["par_wm2"] / par_max).clip(0, 1)
 
    # ── Drought stress (0 = no stress, 1 = severe stress) ────
    # Optimal monthly rainfall for Indian crops: 80-150mm
    RAIN_OPT_MIN = 60.0    # mm/month
    RAIN_OPT_MAX = 180.0
    RAIN_SEVERE  = 15.0    # severe drought below this
 
    def drought_stress(rain_mm):
        if rain_mm >= RAIN_OPT_MIN:
            return 0.0     # no stress
        deficit = (RAIN_OPT_MIN - rain_mm) / RAIN_OPT_MIN
        return float(np.clip(deficit, 0, 1))
 
    df["drought_stress"] = df["rainfall_monthly_mm"].apply(drought_stress)
 
    # ── Heat stress (0 = no stress, 1 = severe stress) ───────
    # Optimal crop temperature: 18-30°C
    TEMP_OPT_MAX = 32.0    # above this → heat stress starts
    TEMP_SEVERE  = 40.0    # severe above this
 
    def heat_stress(temp_c):
        if temp_c <= TEMP_OPT_MAX:
            return 0.0
        excess = (temp_c - TEMP_OPT_MAX) / (TEMP_SEVERE - TEMP_OPT_MAX)
        return float(np.clip(excess, 0, 1))
 
    df["heat_stress"] = df["temperature_c"].apply(heat_stress)
 
    # ── CHI calculation ───────────────────────────────────────
    df["chi"] = (
        df["par_norm"] *
        (1 - df["drought_stress"]) *
        (1 - df["heat_stress"])
    ).clip(0, 1).round(3)
 
    # ── CHI label ─────────────────────────────────────────────
    def chi_label(chi):
        if chi >= 0.75: return "Excellent",  "#1D9E75"
        if chi >= 0.55: return "Good",       "#5DCAA5"
        if chi >= 0.40: return "Moderate",   "#EF9F27"
        if chi >= 0.25: return "Stressed",   "#D85A30"
        return              "Severe",        "#8B1A1A"
 
    labels  = df["chi"].apply(chi_label)
    df["chi_label"] = [l[0] for l in labels]
    df["chi_color"] = [l[1] for l in labels]
 
    return df
 
 
def compute_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CHI anomaly = current vs 5-year monthly average.
    Positive anomaly = better than average. Negative = worse.
    """
    df = df.copy()
 
    # Monthly means across all years
    monthly_means = df.groupby("month")["chi"].mean()
    df["chi_mean"]   = df["month"].map(monthly_means)
    df["chi_anomaly_pct"] = (
        (df["chi"] - df["chi_mean"]) / df["chi_mean"].replace(0, np.nan) * 100
    ).round(1)
 
    return df
 
 
def get_crop_health_data(district: str) -> dict:
    """
    Main entry point for the Streamlit page.
    Returns all data needed for the NDVI monitor.
    """
    if district not in DISTRICT_COORDS:
        raise ValueError(f"District '{district}' not in database.")
 
    lat, lon = DISTRICT_COORDS[district]
 
    # Fetch NASA data
    try:
      df = get_nasa_power_data(lat, lon)
    except Exception as e:
      print("NASA ERROR:", e)   # terminal debug
      raise e
    df = compute_chi(df)
    df = compute_anomaly(df)
 
    # Current month stats
    now          = datetime.now()
    current_mask = (df["year"] == now.year) & (df["month"] == now.month)
    recent_mask  = (df["year"] == now.year - 1) | (
        (df["year"] == now.year) & (df["month"] <= now.month)
    )
 
    if current_mask.any():
        current_row = df[current_mask].iloc[-1]
    else:
        current_row = df.iloc[-1]  # most recent
 
    # 12-month trailing data
    df_12 = df.tail(12).copy()
 
    # 5-year comparison: same months
    five_year_chi = df.groupby("month")["chi"].mean()
 
    return {
        "district":        district,
        "lat":             lat,
        "lon":             lon,
        "df_full":         df,
        "df_12":           df_12,
        "current_chi":     float(current_row["chi"]),
        "current_label":   current_row["chi_label"],
        "current_color":   current_row["chi_color"],
        "current_anomaly": float(current_row.get("chi_anomaly_pct", 0)),
        "current_rain":    float(current_row["rainfall_monthly_mm"]),
        "current_temp":    float(current_row["temperature_c"]),
        "current_par":     float(current_row["par_wm2"]),
        "five_year_chi":   five_year_chi,
        "data_start":      str(df["date"].min().date()),
        "data_end":        str(df["date"].max().date()),
    }
 
 
def get_synthetic_fallback(district: str) -> dict:
    """
    Returns realistic synthetic data when NASA API is unavailable.
    Used as a fallback so the page always shows something meaningful.
    """
    lat, lon = DISTRICT_COORDS.get(district, (21.15, 79.09))
 
    # Generate 5 years of synthetic monthly data
    dates  = pd.date_range("2020-01-01", periods=60, freq="MS")
    np.random.seed(abs(hash(district)) % 999)
 
    # Nagpur-style climate: hot, seasonal monsoon
    monthly_rain = [5, 5, 8, 12, 25, 120, 180, 160, 100, 35, 10, 5] * 5
    monthly_temp = [22, 25, 30, 35, 40, 35, 30, 28, 28, 30, 26, 22] * 5
    monthly_par  = [180, 190, 210, 220, 200, 160, 140, 145, 160, 185, 175, 165] * 5
 
    df = pd.DataFrame({
        "date":                dates[:60],
        "year":                dates[:60].year,
        "month":               dates[:60].month,
        "rainfall_monthly_mm": [r + np.random.normal(0, r*0.15) for r in monthly_rain[:60]],
        "temperature_c":       [t + np.random.normal(0, 0.8) for t in monthly_temp[:60]],
        "par_wm2":             [p + np.random.normal(0, 10) for p in monthly_par[:60]],
        "humidity_pct":        [60 + np.random.normal(0, 8) for _ in range(60)],
        "wind_ms":             [2.5 + np.random.normal(0, 0.4) for _ in range(60)],
    })
    df["rainfall_mm"] = df["rainfall_monthly_mm"] / 30
 
    df = compute_chi(df)
    df = compute_anomaly(df)
 
    current_row = df.iloc[-1]
    five_year_chi = df.groupby("month")["chi"].mean()
 
    return {
        "district":        district,
        "lat":             lat,
        "lon":             lon,
        "df_full":         df,
        "df_12":           df.tail(12),
        "current_chi":     float(current_row["chi"]),
        "current_label":   current_row["chi_label"],
        "current_color":   current_row["chi_color"],
        "current_anomaly": float(current_row.get("chi_anomaly_pct", 0)),
        "current_rain":    float(current_row["rainfall_monthly_mm"]),
        "current_temp":    float(current_row["temperature_c"]),
        "current_par":     float(current_row["par_wm2"]),
        "five_year_chi":   five_year_chi,
        "data_start":      "2020-01-15 (synthetic fallback)",
        "data_end":        str(df["date"].max().date()),
        "is_synthetic":    True,
    }