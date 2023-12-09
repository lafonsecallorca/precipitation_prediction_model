import requests

class WeatherData:

    def __init__(self, api_key, lat, lon, dew_api_key):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.dew_api_key = dew_api_key
        self.units = 'imperial'

    def get_current_data(self):
        url_current = f'https://api.openweathermap.org/data/2.5/weather?lat={self.lat}&lon={self.lon}&appid={self.api_key}&units={self.units}'
        current_response = requests.get(url_current)
        current_weather_data = current_response.json()
        return current_weather_data

    def get_forecast_data(self):
        url_forecast = f'https://api.openweathermap.org/data/2.5/forecast?lat={self.lat}&lon={self.lon}&appid={self.api_key}&units={self.units}'
        forecast_response = requests.get(url_forecast)
        forecast_weather_data = forecast_response.json()
        return forecast_weather_data

    def get_dew_point_data(self):
        url_dew_point = f'https://api.tomorrow.io/v4/weather/forecast?location={self.lat},{self.lon}&apikey={self.api_key_dew}'
        dew_point_response= requests.get(url_dew_point)
        dew_point_data = dew_point_response.json()
        return dew_point_data

    def process_data(self):
        weather_data = self.get_current_data()
        # Retrieve forecast weather data
        forecast_weather_data = self.get_forecast_data()
        # Retrieve dew point data
        dew_point_data = self.get_dew_point_data()

        first_entry = forecast_weather_data.get('list', [])[0] #getting the first list from hourly to use for sea_level pressure 
        main_info = weather_data.get('main', {})
        wind_info = weather_data.get('wind', {})
        weather = weather_data.get('weather', {})
        weather_info =  weather_data.get('weather', [{}])[0].get('main', '')
        weather_description = weather_data.get('weather', [{}])[0].get('description', '')
        temp = main_info.get('temp')
        pressure = main_info.get('pressure')
        humidity = main_info.get('humidity')
        sea_level = first_entry.get('main', {}).get('sea_level', None)
        visibility = weather_data.get('visibility')
        wind_speed = wind_info.get('speed')
        wind_direction = wind_info.get('deg')
        dew_point = dew_point_data.get('timelines', {}).get('minutely', [{}])[0].get('values', {}).get('dewPoint')

        return {
            'dew_point': dew_point,
            'temp': temp,
            'weather_info': weather_info,
            'weather_description': weather_description,
            'sea_level_pressure': sea_level,
            'visibility': visibility,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction
        }
    
    #def process_data_dailyML:
        





# Replace OpenWeatherMap API key and location coordinates
api_key = "7243506b0b349484d43cf58e1d064bac"
lat = "43.1394398"
lon = "-77.5970213"
units = 'imperial'
dew_api_key = 'b8Eudmqk3mta455AVxTMVYrrtEbbvsh7'



