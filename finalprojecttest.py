import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

#cleaning up the data

df = pd.read_csv('3526767.csv', low_memory=False)

cols = ['STATION','NAME','PGTM', 'WESD', 'WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT09', 'WT11', 'WT13', 'WT14', 'WT16', 'WT18', 'WT22']


df = df.drop(cols, axis=1)

df['WDF5'] = df['WDF5'].interpolate()
df['WSF5'] = df['WSF5'].interpolate()

df['DATE'] = pd.to_datetime(df['DATE'])

df['DAY_OF_YEAR'] = df['DATE'].dt.dayofyear

# Drop the original 'DATE' column
df = df.drop('DATE', axis=1)
 

test_data = df[df['TAVG'].isnull()]


df.dropna(inplace=True)

print(df.head())

df.info()

y_train = df['TAVG']
x_train = df.drop(['TAVG'], axis=1)

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)


#Building linear regression model to predict missing values from tavg.shape

lr = LinearRegression()

lr.fit(x_train_scaled,y_train)

x_test = test_data.drop('TAVG', axis=1)
x_test_scaled = scaler.transform(x_test)


y_pred = lr.predict(x_test_scaled)

test_data.loc[test_data.TAVG.isnull(), 'TAVG'] = y_pred

# Update the missing values in 'df' with the imputed values from 'test_data'
df.update(test_data, overwrite=True)


#this is not correct, start looking into mean squared_error and r2gir
mse_all = mean_squared_error(df['TAVG'], y_pred)
r2_all = r2_score(df['TAVG'], y_pred)

print(f'Mean Squared Error (Entire Dataset): {mse_all}')
print(f'R-squared (Entire Dataset): {r2_all}')

# Display the updated DataFrame
df.info





