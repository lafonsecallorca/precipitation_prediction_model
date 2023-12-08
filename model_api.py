import requests

# Replace OpenWeatherMap API key and location coordinates
api_key = "7243506b0b349484d43cf58e1d064bac"
lat = "43.1394398"
lon = "-77.5970213"
minutely = 'minutely'
alerts = ' alerts'
units = 'imperial'

# API endpoint URL
url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}'

# Make the API request
response = requests.get(url)

# Parse the JSON response
weather_data = response.json()

print(weather_data)