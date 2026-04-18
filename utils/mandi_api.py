import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AGMARKNET_API_KEY")

def get_mandi_prices():
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&filters[state]=Maharashtra&filters[commodity]=Cotton"

    try:
        response = requests.get(url)
        data = response.json()

        if "records" not in data or len(data["records"]) == 0:
            return {"live": False, "message": "No live data, using fallback MSP"}

        record = data["records"][0]

        return {
            "live": True,
            "min_price": record.get("min_price"),
            "max_price": record.get("max_price"),
            "modal_price": record.get("modal_price")
        }

    except Exception as e:
        return {"live": False, "error": str(e)}


# test run
if __name__ == "__main__":
    result = get_mandi_prices()
    print(result)