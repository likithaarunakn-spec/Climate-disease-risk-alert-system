import requests
import numpy as np
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
#configurations
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY"
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"
#fixed values 
DEFAULT_AQI = 100              # Average AQI
DEFAULT_POPULATION_DENSITY = 400
#loading the models
model = load_model("malaria_nn_model.keras")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
print("Model, scaler, and encoder loaded")
#getting location automatically
location = requests.get("https://ipapi.co/json/").json()
region = location.get("region", "Unknown")
country = location.get("country_name", "Unknown")
lat = location.get("latitude")
lon = location.get("longitude")
print(f"Location detected: {region}, {country}")
#getting real time weather dataset
weather_url = (
    f"https://api.openweathermap.org/data/2.5/weather"
    f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
)
weather = requests.get(weather_url).json()
avg_temp_c = weather["main"]["temp"]
humidity = weather["main"]["humidity"]
precipitation_mm = weather.get("rain", {}).get("1h", 0)
# UV index requires a separate API → using safe default
uv_index = 6.0
print(" Weather data fetched")
print(
    f"Temp: {avg_temp_c} °C | Rain: {precipitation_mm} mm | "
    f"Humidity: {humidity}%"
)
#preparing model input
# Model was trained on:
# [avg_temp_c, precipitation_mm, air_quality_index, uv_index, population_density]
input_features = np.array([[
    avg_temp_c,
    precipitation_mm,
    DEFAULT_AQI,
    uv_index,
    DEFAULT_POPULATION_DENSITY
]])
# Scale features
input_scaled = scaler.transform(input_features)
#make predictions
pred_probs = model.predict(input_scaled)
pred_class = np.argmax(pred_probs, axis=1)
risk_level = label_encoder.inverse_transform(pred_class)[0]
print(f"Predicted Malaria Risk: {risk_level}")
#telegram alert function
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    requests.post(url, data=payload)
#alert logic
if risk_level == "HIGH":
    alert_message = f"""
    MALARIA RISK ALERT 

    Location: {region}, {country}
    Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

    Temperature: {avg_temp_c} °C
    Rainfall (1h): {precipitation_mm} mm
    UV Index: {uv_index}
    AQI: {DEFAULT_AQI}

    Predicted Risk Level: HIGH

Recommended Actions:
• Increase mosquito control
• Public health awareness
• Preventive medical readiness
"""
    send_telegram_alert(alert_message)
    print("ALERT SENT TO TELEGRAM")

else:
    print("Risk not high — no alert sent")

print("Prediction cycle completed")
