from flask import Flask, render_template, request
import requests
from geopy.geocoders import GoogleV3

app = Flask(__name__)

# Replace 'YOUR_API_KEY' with your actual Google Maps API key
api_key = 'AIzaSyCHy3jGmG7yN0EECyKhTtfSRLgXqHZ894M'

geolocator = GoogleV3(api_key=api_key)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        zipcode = request.form['zipcode']
        
        try:
            location = geolocator.geocode(zipcode)
            if location:
                latitude = location.latitude
                longitude = location.longitude
                
                forecast_url = get_forecast_url(latitude, longitude)
                if forecast_url:
                    weather_info = get_weather_info(forecast_url)
                    return render_template('result.html', weather=weather_info)
                else:
                    return render_template('result.html', error='Failed to fetch weather data.')
            else:
                return render_template('result.html', error='Location not found.')
        except Exception as e:
            return render_template('result.html', error=f'Error: {e}')
    
    return render_template('index.html')

def get_forecast_url(latitude, longitude):
    grid_points_url = f'https://api.weather.gov/points/{latitude},{longitude}'
    response = requests.get(grid_points_url)
    
    if response.status_code == 200:
        data = response.json()
        return data['properties']['forecast']
    else:
        return None

def get_weather_info(forecast_url):
    response = requests.get(forecast_url)
    if response.status_code == 200:
        forecast_data = response.json()
        periods = forecast_data['properties']['periods']
        if periods:
            current_conditions = periods[0]
            return {
                'temperature': current_conditions['temperature'],
                'wind_speed': current_conditions['windSpeed'],
                'wind_direction': current_conditions['windDirection']
            }
        else:
            return None
    else:
        return None

if __name__ == '__main__':
    app.run(debug=True)


