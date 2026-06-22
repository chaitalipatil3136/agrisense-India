import json
import os

# Load JSON
def load_pest_data():
    path = os.path.join("assets", "pest_calendar.json")
    with open(path, "r") as f:
        return json.load(f)


def get_pest_risks(crop, month, temperature, rainfall_mm):
    data = load_pest_data()

    # ✅ fix mismatch
    pests = data.get(crop.lower(), [])

    active_pests = []

    for pest in pests:
        if month in pest["risk_months"]:

            cond = pest["trigger_conditions"]

            # ✅ basic condition check
            if (
                temperature >= cond["min_temp"]
                and temperature <= cond["max_temp"]
                and rainfall_mm >= cond["min_rainfall_mm"]
            ):
                active_pests.append(pest)

    # Sort by risk
    priority = {"high": 3, "medium": 2, "low": 1}
    active_pests.sort(key=lambda x: priority[x["risk_level"]], reverse=True)

    return active_pests


def get_overall_risk_level(pest_list):
    if any(p["risk_level"] == "high" for p in pest_list):
        return "HIGH"
    elif any(p["risk_level"] == "medium" for p in pest_list):
        return "MEDIUM"
    else:
        return "LOW"