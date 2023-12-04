import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

#cleaning up the data

df = pd.read_csv('3526767.csv', low_memory=False)

cols = ['STATION','NAME','PGTM', 'WESD', 'WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT09', 'WT11', 'WT13', 'WT14', 'WT16', 'WT18', 'WT22']

df = df.drop(cols, axis=1)

df['WDF5'] = df['WDF5'].interpolate()
df['WSF5'] = df['WSF5'].interpolate()

#test data is all the rows where TAVG is null 
test_data = df[df['TAVG'].isnull()]

train_data = df.copy()
train_data.dropna(inplace=True)

train_data = train_data.drop('DATE', axis=1)
test_data = test_data.drop('DATE', axis=1)


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


# Display the updated DataFrame
df.info()
print(df.head(3))

df['target'] = df.shift(-1)['PRCP']
print(df.head(5))






