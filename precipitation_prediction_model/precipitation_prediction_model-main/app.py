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

# Function to fetch weather information using OpenWeatherMap API
def get_weather_info(lat, lon):
    openweathermap_api_key = '7243506b0b349484d43cf58e1d064bac'
    tomorrow_io_api_key = 'b8Eudmqk3mta455AVxTMVYrrtEbbvsh7'
    weather_instance = WeatherData(openweathermap_api_key, tomorrow_io_api_key, lat, lon)
    
    # Fetch current and forecast weather data
    current_weather = weather_instance.process_current_data()
    forecast_weather = weather_instance.process_trihourly_forecast()
    
    return current_weather, forecast_weather

# Function to predict rain using DailyModel and HourlyModel
def predict_rain(current_weather):
    daily_model = DailyModel()
    hourly_model = HourlyModel()
    # Fetch data for the models (assuming '3538140.csv' and 'ROC.csv' are available)
    daily_data = daily_model.read_and_clean_data()
    hourly_data = hourly_model.read_and_clean_data()
    
    # Split and scale data for models
    X_train_daily, X_test_daily, y_train_daily, y_test_daily, scaler_daily = daily_model.split_and_scale_data(daily_data)
    X_train_hourly, X_test_hourly, y_train_hourly, y_test_hourly, scaler_hourly = hourly_model.split_and_scale_data(hourly_data)
    
    # Train models
    rf_model_daily = daily_model.train_random_forest_model(X_train_daily, y_train_daily)
    rf_model_hourly = hourly_model.train_random_forest_model(X_train_hourly, y_train_hourly)
    
    # Fetch latitude and longitude from the current_weather dictionary
    lat = current_weather.get('latitude')
    lon = current_weather.get('longitude')
    
    if lat is not None and lon is not None:
        # Make predictions using latitude and longitude
        # You might need to format lat and lon into a list or array depending on the model's input format
        daily_rain_prediction = daily_model.predict(rf_model_daily, [[lat, lon]])
        hourly_rain_prediction = hourly_model.predict(rf_model_hourly, [[lat, lon]])
        
        return daily_rain_prediction, hourly_rain_prediction
    
    # Return None if latitude or longitude is missing
    return None, None



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        zip_code = request.form['zip_code']
        lat, lon = get_coordinates_from_zip(zip_code)
        
        if lat is not None and lon is not None:
            current_weather, forecast_weather = get_weather_info(lat, lon)
            
            # Pass current_weather to predict_rain function
            daily_rain_prediction, hourly_rain_prediction = predict_rain(current_weather)
            
            return render_template('result.html',
                                   current_weather=current_weather,
                                   forecast_weather=forecast_weather,
                                   daily_rain_prediction=daily_rain_prediction,
                                   hourly_rain_prediction=hourly_rain_prediction)
        else:
            return render_template('error.html')
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
