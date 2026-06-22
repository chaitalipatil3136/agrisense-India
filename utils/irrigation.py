import math

def calculate_irrigation(crop, field_ha, expected_rainfall_mm, pump_hp=3, electricity_rate=5.5):
    
    # 🌾 FAO water requirement (mm per season)
    WATER_REQ = {
        "sugarcane": 1500, "rice": 1200, "banana": 1200,
        "cotton": 700, "maize": 500, "wheat": 450, "soybean": 450,
        "chickpea": 350, "mustard": 300, "bajra": 350, "jowar": 400,
        "grapes": 600, "mango": 800, "orange": 800, "pomegranate": 600
    }

    crop = crop.lower()
    water_req = WATER_REQ.get(crop, 500)

    # 💧 Deficit
    deficit_mm = max(0, water_req - expected_rainfall_mm)

    # 💧 Convert to litres
    deficit_litres = deficit_mm * field_ha * 10000

    # 🔁 Irrigation rounds (60mm per round)
    irrigation_rounds = math.ceil(deficit_mm / 60) if deficit_mm > 0 else 0

    # ⚙️ Pump calculation
    pump_lph = pump_hp * 750   # litres per hour
    pump_hours = deficit_litres / pump_lph if pump_lph > 0 else 0

    # ⚡ Electricity cost
    units_per_hour = pump_hp * 0.75
    cost = pump_hours * units_per_hour * electricity_rate

    return {
        "water_req_mm": water_req,
        "rainfall_mm": expected_rainfall_mm,
        "deficit_mm": deficit_mm,
        "deficit_litres": deficit_litres,
        "irrigation_rounds": irrigation_rounds,
        "pump_hours": round(pump_hours, 1),
        "cost": round(cost, 0)
    }