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

class DailyModel:
    def __init__(self, file_path):
            self.file_path = file_path
            self.model = None

    def read_and_clean_data(self):

        df = pd.read_csv(self.file_path, low_memory=False)

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

        scaler = MinMaxScaler()
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

        return df

    def split_and_scale_data(self, df):
        scaler = MinMaxScaler()
        rfdf = df.copy()
        rfdf = rfdf.drop(['STATION'], axis=1)
        rfdf = rfdf.drop(['DATE'], axis=1)

        y = rfdf['PRCP']
        X = rfdf.drop(['PRCP'],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
                
    def train_random_forest_model(self, X_train, y_train):
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)
        return rf_regressor

    def train_knn_model(self, X_train, y_train):
        knn = KNeighborsRegressor(n_neighbors=14)
        knn.fit(X_train, y_train)
        return knn

    def train_svr_model(self, X_train, y_train):
        svr = SVR(kernel='rbf')
        svr.fit(X_train, y_train)
        return svr

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')

    def predict(self, model, data):
        return model.predict(data)

def main():
    file_path = '3538140.csv'
    weather_model = DailyModel(file_path)
    df = weather_model.read_and_clean_data()

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = weather_model.split_and_scale_data(df)
    
    rf_model = weather_model.train_random_forest_model(X_train_scaled, y_train)
    print('RF ')
    rf_predict = weather_model.predict(rf_model, X_test_scaled)
    weather_model.evaluate_model(rf_model, X_test_scaled, y_test)
    
    knn_model = weather_model.train_knn_model(X_train_scaled, y_train)
    knn_predict = weather_model.predict(knn_model, X_test_scaled)
    print('KNN ')
    weather_model.evaluate_model(knn_model, X_test_scaled, y_test)

    svr_model = weather_model.train_svr_model(X_train_scaled, y_train)
    svr_predict = svr_model.predict(X_test_scaled)
    print('SVR ')
    weather_model.evaluate_model(svr_model, X_test_scaled, y_test)

    results_df = pd.DataFrame({
    'Actual_PRCP': y_test,
    'Predicted_KNN': knn_predict,
    'Predicted_RF': rf_predict,
    'Predicted_SVR': svr_predict
    })

    # Display the DataFrame with actual and predicted values
    print(results_df.head(10))
    print(df.tail(5))

    # predict precipitation with weather api
    api_key = "7243506b0b349484d43cf58e1d064bac"
    lat = "43.1394398"
    lon = "-77.5970213"
    dew_api_key = 'b8Eudmqk3mta455AVxTMVYrrtEbbvsh7'

    weather_instance = WeatherData(api_key, dew_api_key, lat, lon)

    try:
        weather_data_result = weather_instance.process_current_data_dailyML()
        if weather_data_result is None:
            raise Exception('API did not return valid data.')
        
        print(weather_data_result)

        current_weatherdf = pd.DataFrame([weather_data_result])

        current_weatherdf.drop(columns=['weather_info', 'weather_description'], axis=1, inplace=True)
        current_weatherdf.fillna(0, inplace=True)

    except Exception as e:
        print(f'Error fetching API data: {str(e)}')
        return None

    #current_weatherdf.info()

    scaled_current_weatherdf = scaler.transform(current_weatherdf)

    current_precipitation_predict = weather_model.predict(knn_model, scaled_current_weatherdf)  #daily model is using knn algorithm

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

if __name__ == "__main__":
    main()
