import requests
import numpy as np
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from meteostat import Point, Daily
#configurations
OPENWEATHER_API_KEY = "OPEN_WEATHER_API_KEY"
TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "TELEGRAM_CHAT_ID"
# Fixed values
DEFAULT_AQI = 100
DEFAULT_POPULATION_DENSITY = 400
# loading malaria model
malaria_model = load_model("malaria_nn_model.keras")
malaria_scaler = joblib.load("scaler.pkl")
malaria_label_encoder = joblib.load("label_encoder.pkl")
#loading dengue model
dengue_model = load_model("dengue_nn_model.keras")
dengue_scaler = joblib.load("dengue_scaler.pkl")
dengue_label_encoder = joblib.load("dengue_label_encoder.pkl")
print("Malaria & Dengue models loaded")
#Get location using IP geolocation
try:
    location = requests.get("https://ipapi.co/json/", timeout=5).json()
    lat = location.get("latitude")
    lon = location.get("longitude")
    region = location.get("region", "Unknown")
    country = location.get("country_name", "Unknown")
except:
    lat = lon = None
# Fallback coordinates if location API fails
if lat is None or lon is None:
    print("Location API failed, using fallback coordinates")
    lat = 13.08      # Chennai coordinates
    lon = 80.27
    region = "Tamil Nadu"
    country = "India"
print(f"Location detected: {region}, {country}")
#get weather data 
def get_weather_openweather(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    response = requests.get(url, timeout=5).json()
    if "main" not in response:
        raise ValueError("OpenWeather failed")
    temp = response["main"]["temp"]
    rain = response.get("rain", {}).get("1h", 0)
    return temp, rain
def get_weather_meteostat(lat, lon):
    end = datetime.now()
    start = end - timedelta(days=7)
    point = Point(lat, lon)
    data = Daily(point, start, end).fetch()
    if data.empty:
        raise ValueError("Meteostat failed")
    temp = data["tavg"].mean()
    rain = data["prcp"].mean()
    return temp, rain
# Try OpenWeather first
try:
    avg_temp_c, precipitation_mm = get_weather_openweather(lat, lon)
    weather_source = "OpenWeather"
except:
    print("OpenWeather failed, switching to Meteostat")
    avg_temp_c, precipitation_mm = get_weather_meteostat(lat, lon)
    weather_source = "Meteostat"
# Defaults for unavailable features
uv_index = 6.0
humidity = 75
print(f"Weather fetched using {weather_source}")
print(
    f"Temperature: {avg_temp_c:.2f} °C | "
    f"Rainfall: {precipitation_mm:.2f} mm"
)
# Prepare model input
# Model trained on:
# [avg_temp_c, precipitation_mm, air_quality_index, uv_index, population_density]
input_features = np.array([[
    avg_temp_c,
    precipitation_mm,
    DEFAULT_AQI,
    uv_index,
    DEFAULT_POPULATION_DENSITY
]])
#Malaria Prediction
malaria_scaled = malaria_scaler.transform(input_features)
malaria_probs = malaria_model.predict(malaria_scaled)
malaria_class = np.argmax(malaria_probs, axis=1)
malaria_risk = malaria_label_encoder.inverse_transform(malaria_class)[0]
#Dengue Prediction
dengue_scaled = dengue_scaler.transform(input_features)
dengue_probs = dengue_model.predict(dengue_scaled)
dengue_class = np.argmax(dengue_probs, axis=1)
dengue_risk = dengue_label_encoder.inverse_transform(dengue_class)[0]
print(f"Malaria Risk: {malaria_risk}")
print(f"Dengue Risk: {dengue_risk}")
#Telegram alert function
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=payload)
#Send alert risks
alert_message = f"""
    CLIMATE-BASED DISEASE RISK UPDATE

    Location: {region}, {country}
    Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

    Weather Conditions
        • Temperature: {avg_temp_c:.2f} °C
        • Rainfall: {precipitation_mm:.2f} mm
        • AQI: {DEFAULT_AQI}
        • UV Index: {uv_index}

    Predicted Disease Risks
        • Malaria Risk: {malaria_risk}
        • Dengue Risk: {dengue_risk}

    This is an automated early-warning health alert.
"""
send_telegram_alert(alert_message)
print("Combined disease alert sent to Telegram")



