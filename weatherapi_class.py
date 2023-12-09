import requests

class WeatherData:

    def __init__(self, api_key, dew_api_key, lat, lon):
        self.api_key = api_key
        self.dew_api_key = dew_api_key
        self.lat = lat
        self.lon = lon
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
        url_dew_point = f'https://api.tomorrow.io/v4/weather/forecast?location={self.lat},{self.lon}&apikey={self.dew_api_key}'
        dew_point_response= requests.get(url_dew_point)
        dew_point_data = dew_point_response.json()
        return dew_point_data

    def process_current_data(self):
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
            'temp': temp,
            'dew_point': dew_point,
            'humidity': humidity,
            'wind_direction': wind_direction,
            'wind_speed': wind_speed,
            'sea_level_pressure': sea_level,
            'visibility': visibility,
            'weather_info': weather_info,
            'weather_description': weather_description, 
        }
    
    def process_data_dailyML(self):
        weather_data = self.get_dew_point_data() #using dewpoint api for daily ML snow and snow depth
        weather_info = self.get_current_data() #using current api for the rest


        snow = weather_data.get('timelines', {}).get('minutely', [{}])[0].get('values', {}).get('snowAccumulation')
        snow_depth = weather_data.get('timelines', {}).get('minutely', [{}])[0].get('values', {}).get('snowDepth')
        main_info = weather_info.get('main', {})
        wind_info = weather_info.get('wind', {})
        tavg = main_info.get('temp')
        temp_min = main_info.get('temp_min')
        temp_max = main_info.get('temp_max')
        wind_speed = wind_info.get('speed')
        wind_direction = wind_info.get('deg')
        weather_str =  weather_info.get('weather', [{}])[0].get('main', '')
        weather_description = weather_info.get('weather', [{}])[0].get('description', '')
        

        return {
            'SNOW': snow,
            'SNWD': snow_depth,
            'TAVG': tavg,
            'TMAX': temp_max,
            'TMIN': temp_min,
            'WDF5': wind_direction,
            'WSF5': wind_speed,
            'weather_info': weather_str,
            'weather_description': weather_description, 
        }





