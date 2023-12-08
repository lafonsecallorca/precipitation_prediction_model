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
import model_api

#cleaning up the data

df = pd.read_csv('ROC.csv', low_memory=False)

df['p01i'] = df['p01i'].fillna(0)

old_cols = ['tmpf', 'valid', 'dwpf', 'relh', 'drct', 'sknt', 'alti', 'mslp', 'p01i', 'vsby']
new_cols = ['Temp (F)', 'Date', 'Dew Point', 'Humidity', 'Wind Direction', 'Wind Speed', 'Altimeter', 'Sea Level Pressure', 'Hourly Prcp', 'Visbility']

for old_col, new_col in zip(old_cols, new_cols):
    df.rename(columns={old_col: new_col}, inplace=True)

df['Temp (F)'] = df['Temp (F)'].interpolate()
df['Dew Point'] = df['Dew Point'].interpolate()
df['Humidity'] = df['Humidity'].interpolate()
df['Wind Direction'] = df['Wind Direction'].interpolate()
df['Wind Speed'] = df['Wind Speed'].interpolate()
df['Altimeter'] = df['Altimeter'].interpolate()
df['Sea Level Pressure'] = df['Sea Level Pressure'].interpolate()
df['Hourly Prcp'] = df['Hourly Prcp'].interpolate()
df['Visbility'] = df['Visbility'].interpolate()


df.info()

#preparing data set for random forest regressor 
rfdf = df.copy()
rfdf = rfdf.drop(['station'], axis=1)

#rfdf['DATE'] = pd.to_datetime(rfdf['DATE'])
#rfdf['DAY_OF_YEAR'] = rfdf['DATE'].dt.dayofyear
rfdf = rfdf.drop(['Date'], axis=1)

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

#svr = SVR(kernel='linear') 
#svr.fit(X_train_scaled, y_train)

# Make predictions on the test set
#y_pred_svr = svr.predict(X_test_scaled)

# Evaluate the model
#mse = mean_squared_error(y_test, y_pred_svr)
#r2 = r2_score(y_test, y_pred_svr)

#print(f'SVR Mean Squared Error: {mse}')
#print(f'R-squared: {r2}')


results_df = pd.DataFrame({
    'Actual_PRCP': y_test,
    'Predicted_KNN': y_pred_knn,
    'Predicted_RF': y_pred
})

# Display the DataFrame with actual and predicted values
print(results_df.head(10))
print(df.head(5))



