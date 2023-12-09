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

#cleaning up the data

df = pd.read_csv('ROC.csv', low_memory=False)

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

#df.info()

#preparing data set for random forest regressor 
rfdf = df.copy()
rfdf = rfdf.drop(['station'], axis=1)

#rfdf['DATE'] = pd.to_datetime(rfdf['DATE'])
#rfdf['DAY_OF_YEAR'] = rfdf['DATE'].dt.dayofyear
rfdf = rfdf.drop(['Date'], axis=1)
rfdf = rfdf.drop(['Altimeter'], axis=1)

y = rfdf['Hourly Prcp']
X = rfdf.drop(['Hourly Prcp'],axis=1)

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust the number of trees (n_estimators)
rf_regressor.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test_scaled)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Random Forest Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

knn = KNeighborsRegressor(n_neighbors=14)  # You can experiment with different values of n_neighbors
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_knn = knn.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_knn)
r2 = r2_score(y_test, y_pred_knn)

print(f'KNN Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

results_df = pd.DataFrame({
    'Actual_PRCP': y_test,
    'Predicted_KNN': y_pred_knn,
    'Predicted_RF': y_pred
})

# Display the DataFrame with actual and predicted values
print(results_df.head(10))
print(df.tail(5))

#predicting on data from current weather api

# Replace OpenWeatherMap API key and location coordinates
api_key = "7243506b0b349484d43cf58e1d064bac"
lat = "43.1394398"
lon = "-77.5970213"
dew_api_key = 'b8Eudmqk3mta455AVxTMVYrrtEbbvsh7'

weather_instance = WeatherData(api_key, dew_api_key, lat, lon)

weather_instance.process_current_data()

weather_data_result = weather_instance.process_current_data()
print(weather_data_result)

current_weatherdf = pd.DataFrame([weather_data_result])

current_weatherdf.drop(columns=['weather_info', 'weather_description'], axis=1, inplace=True)
current_weatherdf.fillna(0, inplace=True)

current_visibility = current_weatherdf['visibility'].values[0]
conversion_factor = 0.000621371
current_visibility_miles = current_visibility * conversion_factor  #dataset was in miles for visibility 
current_visibility_miles = current_visibility_miles.round()
current_weatherdf['visibility'] = current_visibility_miles

api_cols = ['temp','dew_point', 'humidity', 'wind_direction', 'wind_speed', 'sea_level_pressure', 'visibility']
model_cols = ['Temp (F)', 'Dew Point', 'Humidity', 'Wind Direction', 'Wind Speed', 'Sea Level Pressure', 'Visibility']

for api_col, model_col in zip(api_cols, model_cols):
    current_weatherdf.rename(columns={api_col: model_col}, inplace=True)

#current_weatherdf.info()

scaled_current_weatherdf = scaler.transform(current_weatherdf)

current_precipitation_predict = rf_regressor.predict(scaled_current_weatherdf)

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

fiveday_precipitation_predict = rf_regressor.predict(scaled_fiveday_df)

formatted_fiveday_precipitation_predict = ['{:.2f}'.format(value) for value in fiveday_precipitation_predict]

fiveday_df['Predicted_Precipitation'] = formatted_fiveday_precipitation_predict

print(fiveday_df)