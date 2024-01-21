from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from daily_model import DailyModel
from hourly_model import HourlyModel
from weatherapi_class import WeatherData
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()


class PredictionRequest(BaseModel):
    latitude: float
    longitude: float


@app.get("/predict_dailyML/current_weather")
async def predict_current(predictRequest: PredictionRequest):
    try:
        weather_model = DailyModel()
        df = weather_model.read_and_clean_data()
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = weather_model.split_and_scale_data(df)

        knn_model = weather_model.train_knn_model(X_train_scaled, y_train)
        knn_predict = weather_model.predict(knn_model, X_test_scaled)

        # predict precipitation with weather api
        api_key = "7243506b0b349484d43cf58e1d064bac"
        lat = predictRequest.latitude
        lon = predictRequest.longitude
        dew_api_key = 'b8Eudmqk3mta455AVxTMVYrrtEbbvsh7'

        weather_instance = WeatherData(api_key, dew_api_key, lat, lon)

        weather_data_result = weather_instance.process_current_data_dailyML()
        if weather_data_result is None:
            raise HTTPException(status_code=500, detail="API did not return valid data.")

        # logger.debug(weather_data_result)

        current_weatherdf = pd.DataFrame([weather_data_result])

        current_weatherdf.drop(columns=['weather_info', 'weather_description', 'city_name'], axis=1, inplace=True)
        current_weatherdf.fillna(0, inplace=True)

    except Exception as e:
        print(f'Error {str(e)}')
        raise HTTPException(status_code=500, detail="Internal Server Error")

    scaled_current_weatherdf = scaler.transform(current_weatherdf)

    current_precipitation_predict = weather_model.predict(knn_model,
                                                          scaled_current_weatherdf)  # daily model is using knn algorithm

    formatted_current_precipitation_predict = ['{:.2f}'.format(value) for value in current_precipitation_predict]
    formatted_current_precipitation_predict = formatted_current_precipitation_predict[0]

    current_temp = weather_data_result.get('TAVG')
    current_temp_min = weather_data_result.get('TMIN')
    current_temp_max = weather_data_result.get('TMAX')
    current_wind_direction = weather_data_result.get('WDF5')
    current_wind_speed = weather_data_result.get('WSF5')
    current_weather_info = weather_data_result.get('weather_info')
    current_weather_description = weather_data_result.get('weather_description')
    current_city_name = weather_data_result.get('city_name')

    return {
        "Prediction": f"The current weather for {current_city_name} with precipitation prediction using the daily model",
        "Current temp is": current_temp,
        "Current temp min is": current_temp_min,
        "Current temp max is": current_temp_max,
        "current wind speed is": current_wind_speed,
        "Current wind direction is": current_wind_direction,
        "Current weather info is": current_weather_info,
        "Current weather description is": current_weather_description,
        "Current predicted precipitation is": formatted_current_precipitation_predict
        }


@app.get("/predict_hourlyML/current_weather")
async def predict_current(predictRequest: PredictionRequest):
    try:
        weather_model = HourlyModel()
        df = weather_model.read_and_clean_data()

        X_train_scaled, X_test_scaled, y_train, y_test, scaler = weather_model.split_and_scale_data(df)
        rf_model = weather_model.train_random_forest_model(X_train_scaled, y_train)
        rf_predict = weather_model.predict(rf_model, X_test_scaled)

        # predict precipitation with weather api
        api_key = "7243506b0b349484d43cf58e1d064bac"
        lat = predictRequest.latitude
        lon = predictRequest.longitude
        dew_api_key = 'b8Eudmqk3mta455AVxTMVYrrtEbbvsh7'

        weather_instance = WeatherData(api_key, dew_api_key, lat, lon)

        weather_data_result = weather_instance.process_current_data()
        if weather_data_result is None:
            raise HTTPException(status_code=500, detail="API did not return valid data.")

        # logger.debug(weather_data_result)

        current_weatherdf = pd.DataFrame([weather_data_result])
        current_weatherdf.drop(columns=['weather_info', 'weather_description', 'city_name'], axis=1, inplace=True)
        current_weatherdf.fillna(0, inplace=True)

        # our training model has visibility in miles so we must convert the api's visibility
        current_visibility = current_weatherdf['visibility'].values[0]
        conversion_factor = 0.000621371
        current_visibility_miles = current_visibility * conversion_factor  # dataset was in miles for visibility
        current_visibility_miles = current_visibility_miles.round()
        current_weatherdf['visibility'] = current_visibility_miles

    except Exception as e:
        print(f'Error {str(e)}')
        raise HTTPException(status_code=500, detail="Internal Server Error")

    api_cols = ['temp', 'dew_point', 'humidity', 'wind_direction', 'wind_speed', 'sea_level_pressure', 'visibility']
    model_cols = ['Temp (F)', 'Dew Point', 'Humidity', 'Wind Direction', 'Wind Speed', 'Sea Level Pressure',
                  'Visibility']

    for api_col, model_col in zip(api_cols, model_cols):
        current_weatherdf.rename(columns={api_col: model_col}, inplace=True)

    scaled_current_weatherdf = scaler.transform(current_weatherdf)
    current_precipitation_predict = weather_model.predict(rf_model, scaled_current_weatherdf)
    formatted_current_precipitation_predict = ['{:.2f}'.format(value) for value in current_precipitation_predict]
    formatted_current_precipitation_predict = formatted_current_precipitation_predict[0]

    current_temp = weather_data_result.get('temp')
    current_dew_point = weather_data_result.get('dew_point')
    current_humidity = weather_data_result.get('humidity')
    current_wind_direction = weather_data_result.get('wind_direction')
    current_wind_speed = weather_data_result.get('wind_speed')
    current_seal_level = weather_data_result.get('sea_level_pressure')
    current_visibility = weather_data_result.get('visibility')
    current_weather_info = weather_data_result.get('weather_info')
    current_weather_description = weather_data_result.get('weather_description')
    current_city_name = weather_data_result.get('city_name')

    return {
        "city": current_city_name,
        "temp": current_temp,
        "dew_point": current_dew_point,
        "humidity": current_humidity,
        "sea_level": current_seal_level,
        "wind_speed": current_wind_speed,
        "wind_direction": current_wind_direction,
        "weather_info": current_weather_info,
        "weather_details": current_weather_description,
        "predicted_precipitation": formatted_current_precipitation_predict
        }


@app.get("/predict_hourlyML/fiveday/trihourly_forecast")
async def predict_fiveday_trihourly(predictRequest: PredictionRequest):
    weather_model = HourlyModel()
    df = weather_model.read_and_clean_data()

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = weather_model.split_and_scale_data(df)
    rf_model = weather_model.train_random_forest_model(X_train_scaled, y_train)
    rf_predict = weather_model.predict(rf_model, X_test_scaled)

    # predict precipitation with weather api
    api_key = "7243506b0b349484d43cf58e1d064bac"
    lat = predictRequest.latitude
    lon = predictRequest.longitude
    dew_api_key = 'b8Eudmqk3mta455AVxTMVYrrtEbbvsh7'

    weather_instance = WeatherData(api_key, dew_api_key, lat, lon)

    try:
        fiveday_forecast_every_three = weather_instance.process_trihourly_forecast()
        if fiveday_forecast_every_three is None:
            raise HTTPException(status_code=500, detail="API did not return valid data.")

        fiveday_df = pd.DataFrame(fiveday_forecast_every_three)

        fiveday_df_features = fiveday_df.copy()

        fiveday_df_features.drop(
            columns=['weather_info', 'weather_description', 'Date', 'feels_like', 'temp_min', 'temp_max', 'city_name'],
            axis=1, inplace=True)
        fiveday_df_features.fillna(0, inplace=True)

        for index, row in fiveday_df_features.iterrows():
            fiveday_visibility = row['Visibility']
            conversion_factor = 0.000621371
            fiveday_visibility_miles = fiveday_visibility * conversion_factor
            fiveday_visibility_miles = fiveday_visibility_miles.round()
            fiveday_df_features.at[index, 'Visibility'] = fiveday_visibility_miles

        scaled_fiveday_df = scaler.transform(fiveday_df_features)

    except Exception as e:
        print(f"Error fetching API data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")

    fiveday_precipitation_predict = weather_model.predict(rf_model, scaled_fiveday_df)

    # logger.debug(fiveday_precipitation_predict)

    formatted_fiveday_precipitation_predict = ['{:.2f}'.format(value) for value in fiveday_precipitation_predict]

    fiveday_df['Predicted_Precipitation'] = formatted_fiveday_precipitation_predict

    # logger.debug(fiveday_df)

    return {"Prediction": fiveday_df.to_dict(orient='records')}
