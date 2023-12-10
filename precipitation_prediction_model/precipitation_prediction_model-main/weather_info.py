import requests
from geopy.geocoders import GoogleV3

# Replace 'YOUR_API_KEY' with your actual Google Maps API key
api_key = 'AIzaSyCHy3jGmG7yN0EECyKhTtfSRLgXqHZ894M'

geolocator = GoogleV3(api_key=api_key)

address = '14420'  # Replace '90210' with the ZIP code or 'City, State' format

try:
    location = geolocator.geocode(address)
    if location:
        print(f'Latitude: {location.latitude}, Longitude: {location.longitude}')
    else:
        print('Location not found.')
except Exception as e:
    print(f'Error: {e}')

latitude = location.latitude
longitude = location.longitude

 
def get_grid_points(latitude, longitude):
    grid_points_url = f'https://api.weather.gov/points/{latitude},{longitude}'
    response = requests.get(grid_points_url)
    
    if response.status_code == 200:
        data = response.json()
        forecast_url = data['properties']['forecast']
        return forecast_url
    else:
        return None
    
forecast_url = get_grid_points(latitude, longitude)

if forecast_url:
    # Fetch forecast data for the specific point
    response = requests.get(forecast_url)
    
    if forecast_url:
        response = requests.get(forecast_url)
        if response.status_code == 200:
            forecast_data = response.json()
            
            periods = forecast_data['properties']['periods']
            if periods:
                current_conditions = periods[0]
                
                temperature = current_conditions['temperature']
                wind_speed = current_conditions['windSpeed']
                wind_direction = current_conditions['windDirection']
                
                print(f'Current Temperature: {temperature}')
                print(f'Wind Speed: {wind_speed}')
                print(f'Wind Direction: {wind_direction}')
            else:
                print('No forecast periods available.')
        else:
            print('Failed to fetch weather data.')
    else:
        print('Failed to obtain grid points for coordinates')





