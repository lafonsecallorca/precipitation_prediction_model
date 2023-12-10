import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from weatherapi_class import WeatherData

def read_and_clean_data(file_path):

    df = pd.read_csv(file_path, low_memory=False)

    df['p01i'] = df['p01i'].fillna(0)

    old_cols = ['tmpf', 'valid', 'dwpf', 'relh', 'drct', 'sknt', 'alti', 'mslp', 'p01i', 'vsby']
    new_cols = ['Temp (F)', 'Date', 'Dew Point', 'Humidity', 'Wind Direction', 'Wind Speed', 'Altimeter', 'Sea Level Pressure', 'Hourly Prcp', 'Visibility']

    for old_col, new_col in zip(old_cols, new_cols):
        df.rename(columns={old_col: new_col}, inplace=True)

    df['Temp (F)'] = df['Temp (F)'].interpolate()
    df['Dew Point'] = df['Dew Point'].interpolate()
    df['Humidity'] = df['Humidity'].interpolate()
    df['Wind Direction'] = df['Wind Direction'].interpolate()
    df['Wind Speed'] = df['Wind Speed'].interpolate()
    #df['Altimeter'] = df['Altimeter'].interpolate()
    df['Sea Level Pressure'] = df['Sea Level Pressure'].interpolate()
    df['Hourly Prcp'] = df['Hourly Prcp'].interpolate()
    df['Visibility'] = df['Visibility'].interpolate()

    #preparing data set for random forest regressor 
    rfdf = df.copy()
    rfdf = rfdf.drop(['station'], axis=1)
    rfdf = rfdf.drop(['Date'], axis=1)
    rfdf = rfdf.drop(['Altimeter'], axis=1)  #could not find API that gave altimeter so had to drop it from training feature

    return rfdf

def split_and_scale_data(rfdf):

    y = rfdf['Hourly Prcp']
    X = rfdf.drop(['Hourly Prcp'],axis=1)

    scaler = StandardScaler()  #z-score normilaztion worked best for large dataset

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
            
def train_random_forest_model(X_train, y_train):
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    return rf_regressor

def train_knn_model(X_train, y_train):
    knn = KNeighborsRegressor(n_neighbors=14)
    knn.fit(X_train, y_train)
    return knn

def train_svr_model(X_train, y_train):
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    return svr

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

def main():
    file_path = 'ROC.csv'
    df = read_and_clean_data(file_path)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(df)
    
    rf_model = train_random_forest_model(X_train_scaled, y_train)
    rf_predict = rf_model.predict(X_test_scaled)

    evaluate_model(rf_model, X_test_scaled, y_test)

    knn_model = train_knn_model(X_train_scaled, y_train)
    knn_predict = knn_model.predict(X_test_scaled)

    evaluate_model(knn_model, X_test_scaled, y_test)


    results_df = pd.DataFrame({
    'Actual_PRCP': y_test,
    'Predicted_KNN': knn_predict,
    'Predicted_RF': rf_predict
    })

    # Display the DataFrame with actual and predicted values
    print(results_df.head(10))
    print(df.tail(5))



    # Replace OpenWeatherMap API key and location coordinates
    api_key = "7243506b0b349484d43cf58e1d064bac"
    lat = "43.1394398"
    lon = "-77.5970213"
    dew_api_key = 'b8Eudmqk3mta455AVxTMVYrrtEbbvsh7'

    weather_instance = WeatherData(api_key, dew_api_key, lat, lon)
    weather_data_result = weather_instance.process_current_data()
    print(weather_data_result)

    #create df from the weather api data 
    current_weatherdf = pd.DataFrame([weather_data_result])

    current_weatherdf.drop(columns=['weather_info', 'weather_description'], axis=1, inplace=True)
    current_weatherdf.fillna(0, inplace=True)

    #our training model has visibility in miles so we must convert the api's visibility
    current_visibility = current_weatherdf['visibility'].values[0]
    conversion_factor = 0.000621371
    current_visibility_miles = current_visibility * conversion_factor  #dataset was in miles for visibility 
    current_visibility_miles = current_visibility_miles.round()
    current_weatherdf['visibility'] = current_visibility_miles


    #features must be named the same as during fit time
    api_cols = ['temp','dew_point', 'humidity', 'wind_direction', 'wind_speed', 'sea_level_pressure', 'visibility']
    model_cols = ['Temp (F)', 'Dew Point', 'Humidity', 'Wind Direction', 'Wind Speed', 'Sea Level Pressure', 'Visibility']

    for api_col, model_col in zip(api_cols, model_cols):
        current_weatherdf.rename(columns={api_col: model_col}, inplace=True)


    scaled_current_weatherdf = scaler.transform(current_weatherdf)
    current_precipitation_predict = rf_model.predict(scaled_current_weatherdf)
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


    print(f'Current temp is {current_temp}')
    print(f'Current dew point is {current_dew_point}')
    print(f'Current humidity is {current_humidity}')
    print(f'Current wind speed is {current_wind_speed} with a direction of {current_wind_direction} degrees')
    print(f'Current sea level pressure is {current_seal_level}')
    print(f'Current visbility is {current_visibility}')
    print(f'The current forecast is {current_weather_info} and the description is {current_weather_description}')
    print(f'The predicted precipitation for the current weather is: {formatted_current_precipitation_predict}')

    fiveday_forecast_every_three = weather_instance.process_hourly_forecast()

    fiveday_df = pd.DataFrame(fiveday_forecast_every_three)

    fiveday_df_features = fiveday_df.copy()

    fiveday_df_features.drop(columns=['weather_info', 'weather_description', 'Date', 'feels_like', 'temp_min', 'temp_max'], axis=1, inplace=True)
    fiveday_df_features.fillna(0, inplace=True)

    for index, row in fiveday_df_features.iterrows():
        fiveday_visibility = row['Visibility']
        conversion_factor = 0.000621371
        fiveday_visibility_miles = fiveday_visibility * conversion_factor
        fiveday_visibility_miles = fiveday_visibility_miles.round()
        fiveday_df_features.at[index, 'Visibility'] = fiveday_visibility_miles

    scaled_fiveday_df = scaler.transform(fiveday_df_features)

    fiveday_precipitation_predict = rf_model.predict(scaled_fiveday_df)

    formatted_fiveday_precipitation_predict = ['{:.2f}'.format(value) for value in fiveday_precipitation_predict]

    fiveday_df['Predicted_Precipitation'] = formatted_fiveday_precipitation_predict

    print(fiveday_df)
        
if __name__ == "__main__":
    main()
