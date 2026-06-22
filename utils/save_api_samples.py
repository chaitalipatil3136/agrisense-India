import json
import os
from utils.weather_api import get_weather
from utils.mandi_api import get_mandi_prices

# Ensure folder exists
os.makedirs("data/raw", exist_ok=True)

# 🌦️ Save weather data
weather_data = get_weather("Nagpur")

with open("data/raw/sample_weather_nagpur.json", "w") as f:
    json.dump(weather_data, f, indent=4)

print("✅ Weather sample saved")

# 🌾 Save mandi data
mandi_data = get_mandi_prices()

with open("data/raw/sample_mandi_cotton.json", "w") as f:
    json.dump(mandi_data, f, indent=4)

print("✅ Mandi sample saved")