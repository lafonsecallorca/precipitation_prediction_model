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

class HourlyModel:
        
    def __init__(self):
            self.file_path = 'ROC.csv'
            self.model = None

    def read_and_clean_data(self):

        df = pd.read_csv(self.file_path, low_memory=False)

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

    def split_and_scale_data(self, rfdf):

        y = rfdf['Hourly Prcp']
        X = rfdf.drop(['Hourly Prcp'],axis=1)

        scaler = StandardScaler()

        #z-score normilaztion worked best for large dataset

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
                
    def train_random_forest_model(self, X_train, y_train):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        return self.model

    def train_knn_model(self, X_train, y_train):
        self.model = KNeighborsRegressor(n_neighbors=14)
        self.model.fit(X_train, y_train)
        return self.model

    def train_svr_model(self, X_train, y_train):
        self.model = SVR(kernel='rbf')
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')

    def predict(self, model, data):
        return model.predict(data)


def main():
    weather_model = HourlyModel()
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

    results_df = pd.DataFrame({
    'Actual_PRCP': y_test,
    'Predicted_KNN': knn_predict,
    'Predicted_RF': rf_predict
    })

     # Display the DataFrame with actual and predicted values
    print(results_df.head(10))
    print(df.tail(5))

 



