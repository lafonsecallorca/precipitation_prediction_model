from flask import Flask, render_template, request
from geopy.geocoders import GoogleV3
from daily_model import DailyModel
from hourly_model import HourlyModel
from weatherapi_class import WeatherData

app = Flask(__name__)

# Function to get latitude and longitude from ZIP code using Google Maps API
def get_coordinates_from_zip(zip_code):
    api_key = 'AIzaSyCHy3jGmG7yN0EECyKhTtfSRLgXqHZ894M'
    geolocator = GoogleV3(api_key=api_key)
    location = geolocator.geocode(zip_code)
    if location:
        return location.latitude, location.longitude
    return None, None

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
