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

df = pd.read_csv('3538140.csv', low_memory=False)

cols = ['NAME','PGTM','WDF2','WSF2','AWND']

df = df.drop(cols, axis=1)

df['WDF5'] = df['WDF5'].interpolate()
df['WSF5'] = df['WSF5'].interpolate()


#test data is all the rows where TAVG is null 
test_data = df[df['TAVG'].isnull()]

train_data = df.copy()
train_data.dropna(inplace=True)

#drop the date and station cols for the train and test data set since linear regression only uses numerical values 
train_data = train_data.drop('DATE', axis=1)
test_data = test_data.drop('DATE', axis=1)
train_data = train_data.drop('STATION', axis=1)
test_data = test_data.drop('STATION', axis=1)

y_train = train_data['TAVG']
x_train = train_data.drop(['TAVG'], axis=1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)


#Building linear regression model to predict missing values from tavg.shape

alpha = 0.01
lasso = Lasso(alpha=alpha)

lasso.fit(x_train_scaled,y_train)

x_test = test_data.drop('TAVG', axis=1)
x_test_scaled = scaler.transform(x_test)


y_pred = lasso.predict(x_test_scaled)

test_data.loc[test_data.TAVG.isnull(), 'TAVG'] = y_pred
df.loc[df['TAVG'].isnull(), 'TAVG'] = y_pred


# Display the updated DataFrame
df.info()
print(df.tail(3))


#preparing data set for random forest regressor 
rfdf = df.copy()
rfdf = rfdf.drop(['STATION'], axis=1)

#rfdf['DATE'] = pd.to_datetime(rfdf['DATE'])
#rfdf['DAY_OF_YEAR'] = rfdf['DATE'].dt.dayofyear
rfdf = rfdf.drop(['DATE'], axis=1)

y = rfdf['PRCP']
X = rfdf.drop(['PRCP'],axis=1)


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


svr = SVR(kernel='rbf') 
svr.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_svr = svr.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_svr)
r2 = r2_score(y_test, y_pred_svr)

print(f'SVR Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


results_df = pd.DataFrame({
    'Actual_PRCP': y_test,
    'Predicted_KNN': y_pred_knn,
    'Predicted_RF': y_pred,
    'Predicted_SVR': y_pred_svr
})

# Display the DataFrame with actual and predicted values
print(results_df.head(10))
print(df.head(5))


#
# Replace OpenWeatherMap API key and location coordinates
api_key = "7243506b0b349484d43cf58e1d064bac"
lat = "43.1394398"
lon = "-77.5970213"
dew_api_key = 'b8Eudmqk3mta455AVxTMVYrrtEbbvsh7'

weather_instance = WeatherData(api_key, dew_api_key, lat, lon)

weather_data_result = weather_instance.process_current_data_dailyML()
print(weather_data_result)

current_weatherdf = pd.DataFrame([weather_data_result])

current_weatherdf.drop(columns=['weather_info', 'weather_description'], axis=1, inplace=True)
current_weatherdf.fillna(0, inplace=True)

#current_weatherdf.info()

scaled_current_weatherdf = scaler.transform(current_weatherdf)

current_precipitation_predict = knn.predict(scaled_current_weatherdf)  #daily model is using knn algorithm

formatted_current_precipitation_predict = ['{:.2f}'.format(value) for value in current_precipitation_predict]
formatted_current_precipitation_predict = formatted_current_precipitation_predict[0]

current_temp = weather_data_result.get('TAVG')
current_temp_min = weather_data_result.get('TMIN')
current_temp_max = weather_data_result.get('TMAX')
current_wind_direction = weather_data_result.get('WDF5')
current_wind_speed = weather_data_result.get('WSF5')
current_weather_info = weather_data_result.get('weather_info')
current_weather_description = weather_data_result.get('weather_description')


print(f'Current temp is {current_temp}')
print(f'Current wind speed is {current_wind_speed} with a direction of {current_wind_direction} degrees')
print(f'The current forecast is {current_weather_info} and the description is {current_weather_description}')
print(f'The predicted precipitation for the current weather with our daily model is: {formatted_current_precipitation_predict}')

