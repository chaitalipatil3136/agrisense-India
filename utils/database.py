"""
AgriSense India — SQLite Database Utility
File: utils/database.py
 
Handles all database operations for prediction history.
SQLite is part of Python standard library — zero extra install needed.
 
Tables:
  predictions — one row per crop recommendation prediction
  disease_logs — one row per disease detection
 
Import: from utils.database import init_db, save_prediction,
                                   get_history, get_stats,
                                   save_disease_log, get_disease_history
"""
 
import sqlite3
import pandas as pd
from datetime import datetime
import os
 
DB_PATH = "agrisense.db"
 
 
# ════════════════════════════════════════════════════════════
# INIT — creates tables if they don't exist
# ════════════════════════════════════════════════════════════
 
def init_db():
    """
    Create database and all tables on first run.
    Safe to call every time the app loads — uses IF NOT EXISTS.
    """
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        cursor = conn.cursor()
 
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                district        TEXT,
                n_val           REAL,
                p_val           REAL,
                k_val           REAL,
                temp            REAL,
                humidity        REAL,
                ph              REAL,
                rainfall        REAL,
                predicted_crop  TEXT,
                confidence      REAL,
                yield_est       REAL,
                earnings_est    REAL,
                model_used      TEXT    DEFAULT 'Random Forest'
            )
        """)
 
        # Disease detections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS disease_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                disease_name    TEXT,
                crop_name       TEXT,
                severity        TEXT,
                confidence      REAL,
                image_filename  TEXT
            )
        """)
 
        # Carbon footprint logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS carbon_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                crop            TEXT,
                field_ha        REAL,
                n_applied       REAL,
                p_applied       REAL,
                k_applied       REAL,
                total_kgco2e    REAL,
                sustainability  TEXT
            )
        """)
 
        conn.commit()
 
 
# ════════════════════════════════════════════════════════════
# PREDICTIONS — save + retrieve
# ════════════════════════════════════════════════════════════
 
def save_prediction(data: dict) -> bool:
    """
    Save one crop prediction to the database.
 
    Parameters
    ----------
    data : dict with keys matching table columns.
           Required: predicted_crop, confidence
           Optional: all others default to None
 
    Returns True on success, False on failure.
    """
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            conn.execute("""
                INSERT INTO predictions
                    (timestamp, district, n_val, p_val, k_val, temp,
                     humidity, ph, rainfall, predicted_crop, confidence,
                     yield_est, earnings_est, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.get("district"),
                data.get("n_val"),
                data.get("p_val"),
                data.get("k_val"),
                data.get("temp"),
                data.get("humidity"),
                data.get("ph"),
                data.get("rainfall"),
                data.get("predicted_crop"),
                data.get("confidence"),
                data.get("yield_est"),
                data.get("earnings_est"),
                data.get("model_used", "Random Forest"),
            ))
            conn.commit()
        return True
    except Exception as e:
        print(f"[DB] save_prediction failed: {e}")
        return False
 
 
def get_history(limit: int = 10) -> pd.DataFrame:
    """
    Return last N predictions as a DataFrame.
    Returns empty DataFrame if no data or table doesn't exist.
    """
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            df = pd.read_sql_query("""
                SELECT
                    timestamp        AS "Time",
                    district         AS "District",
                    predicted_crop   AS "Recommended Crop",
                    ROUND(confidence,1) AS "Confidence (%)",
                    n_val AS "N", p_val AS "P", k_val AS "K",
                    ROUND(ph,1) AS "pH",
                    ROUND(rainfall,0) AS "Rainfall (mm)",
                    ROUND(yield_est,0) AS "Yield (kg/ha)",
                    ROUND(earnings_est,0) AS "Earnings (₹/ha)"
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
            """, conn, params=(limit,))
        return df
    except Exception:
        return pd.DataFrame()
 
 
def get_stats() -> dict:
    """
    Return aggregate statistics across all predictions.
    """
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            cursor = conn.cursor()
 
            # Total count
            total = cursor.execute(
                "SELECT COUNT(*) FROM predictions"
            ).fetchone()[0]
 
            # Most recommended crop
            top_crop_row = cursor.execute("""
                SELECT predicted_crop, COUNT(*) as cnt
                FROM predictions
                WHERE predicted_crop IS NOT NULL
                GROUP BY predicted_crop
                ORDER BY cnt DESC
                LIMIT 1
            """).fetchone()
            top_crop = top_crop_row[0].capitalize() if top_crop_row else "N/A"
 
            # Average confidence
            avg_conf = cursor.execute(
                "SELECT ROUND(AVG(confidence),1) FROM predictions"
            ).fetchone()[0] or 0.0
 
            # Average earnings
            avg_earn = cursor.execute(
                "SELECT ROUND(AVG(earnings_est),0) FROM predictions WHERE earnings_est > 0"
            ).fetchone()[0] or 0
 
            # Most active district
            top_dist_row = cursor.execute("""
                SELECT district, COUNT(*) as cnt
                FROM predictions
                WHERE district IS NOT NULL
                GROUP BY district
                ORDER BY cnt DESC
                LIMIT 1
            """).fetchone()
            top_dist = top_dist_row[0] if top_dist_row else "N/A"
 
        return {
            "total":           total,
            "top_crop":        top_crop,
            "avg_confidence":  avg_conf,
            "avg_earnings":    avg_earn,
            "top_district":    top_dist,
        }
    except Exception:
        return {
            "total": 0, "top_crop": "N/A",
            "avg_confidence": 0.0, "avg_earnings": 0,
            "top_district": "N/A",
        }
 
 
# ════════════════════════════════════════════════════════════
# DISEASE LOGS — save + retrieve
# ════════════════════════════════════════════════════════════
 
def save_disease_log(data: dict) -> bool:
    """Save one disease detection result."""
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            conn.execute("""
                INSERT INTO disease_logs
                    (timestamp, disease_name, crop_name,
                     severity, confidence, image_filename)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.get("disease_name"),
                data.get("crop_name"),
                data.get("severity"),
                data.get("confidence"),
                data.get("image_filename"),
            ))
            conn.commit()
        return True
    except Exception as e:
        print(f"[DB] save_disease_log failed: {e}")
        return False
 
 
def get_disease_history(limit: int = 10) -> pd.DataFrame:
    """Return last N disease detections as a DataFrame."""
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            df = pd.read_sql_query("""
                SELECT
                    timestamp   AS "Time",
                    disease_name AS "Disease",
                    crop_name   AS "Crop",
                    severity    AS "Severity",
                    ROUND(confidence*100,1) AS "Confidence (%)"
                FROM disease_logs
                ORDER BY id DESC
                LIMIT ?
            """, conn, params=(limit,))
        return df
    except Exception:
        return pd.DataFrame()
 
 
# ════════════════════════════════════════════════════════════
# CARBON LOGS
# ════════════════════════════════════════════════════════════
 
def save_carbon_log(data: dict) -> bool:
    """Save one carbon footprint calculation."""
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            conn.execute("""
                INSERT INTO carbon_logs
                    (timestamp, crop, field_ha, n_applied, p_applied,
                     k_applied, total_kgco2e, sustainability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.get("crop"),
                data.get("field_ha"),
                data.get("n_applied"),
                data.get("p_applied"),
                data.get("k_applied"),
                data.get("total_kgco2e"),
                data.get("sustainability"),
            ))
            conn.commit()
        return True
    except Exception as e:
        print(f"[DB] save_carbon_log failed: {e}")
        return False
 
 
# ════════════════════════════════════════════════════════════
# SQL ANALYTICS — used in notebooks for SQL skill demo
# ════════════════════════════════════════════════════════════
 
def run_analytics() -> None:
    """
    Example SQL queries for notebook use.
    Run this in a Jupyter/Colab cell to demo SQL skills.
    """
    import pandas as pd
 
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
 
        print("=== Query 1: Crop recommendation frequency ===")
        df1 = pd.read_sql_query("""
            SELECT
                predicted_crop              AS crop,
                COUNT(*)                    AS times_recommended,
                ROUND(AVG(confidence), 1)   AS avg_confidence_pct,
                ROUND(AVG(yield_est), 0)    AS avg_yield_kg_ha,
                ROUND(AVG(earnings_est), 0) AS avg_earnings_rupees
            FROM predictions
            WHERE predicted_crop IS NOT NULL
            GROUP BY predicted_crop
            ORDER BY times_recommended DESC
        """, conn)
        print(df1.to_string(index=False))
 
        print("\n=== Query 2: High-confidence predictions ===")
        df2 = pd.read_sql_query("""
            SELECT district, predicted_crop, confidence, timestamp
            FROM predictions
            WHERE confidence >= 85
            ORDER BY confidence DESC
            LIMIT 15
        """, conn)
        print(df2.to_string(index=False))
 
        print("\n=== Query 3: Soil profile by district ===")
        df3 = pd.read_sql_query("""
            SELECT
                district,
                COUNT(*)            AS prediction_count,
                ROUND(AVG(n_val),1) AS avg_N,
                ROUND(AVG(p_val),1) AS avg_P,
                ROUND(AVG(k_val),1) AS avg_K,
                ROUND(AVG(ph),2)    AS avg_pH
            FROM predictions
            WHERE district IS NOT NULL
            GROUP BY district
            ORDER BY prediction_count DESC
        """, conn)
        print(df3.to_string(index=False))
 
        print("\n=== Query 4: Monthly prediction trends ===")
        df4 = pd.read_sql_query("""
            SELECT
                SUBSTR(timestamp, 1, 7)     AS month,
                COUNT(*)                     AS predictions,
                predicted_crop               AS top_crop
            FROM predictions
            GROUP BY SUBSTR(timestamp, 1, 7)
            ORDER BY month DESC
            LIMIT 12
        """, conn)
        print(df4.to_string(index=False))
 
 
if __name__ == "__main__":
    init_db()
    print(f"Database initialised at: {os.path.abspath(DB_PATH)}")
    print("Tables created: predictions, disease_logs, carbon_logs")
 
    # Insert a test prediction
    save_prediction({
        "district": "Nagpur, Maharashtra",
        "n_val": 90, "p_val": 42, "k_val": 43,
        "temp": 28.5, "humidity": 82.0, "ph": 6.5, "rainfall": 202.0,
        "predicted_crop": "cotton", "confidence": 91.2,
        "yield_est": 1840, "earnings_est": 121440,
    })
    print("\nTest prediction saved. Running analytics...\n")
    run_analytics()