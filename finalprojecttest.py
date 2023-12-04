import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
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

#test data is all the rows where TAVG is null 
test_data = df[df['TAVG'].isnull()]

train_data = df.copy()
train_data.dropna(inplace=True)


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


num_rows_to_select = 212
y_true_imputed = y_train.iloc[:num_rows_to_select]

# Calculate MSE between predicted and true values
mse_y_pred = mean_squared_error(y_true_imputed, y_pred)

print(f'Mean Squared Error (Predicted Values vs True Values): {mse_y_pred}')


test_data.loc[test_data.TAVG.isnull(), 'TAVG'] = y_pred
df.loc[df['TAVG'].isnull(), 'TAVG'] = y_pred



# Display the updated DataFrame
df.info()
print(df.head(3))
print(df.iloc[50:100])
print(df.iloc[200:230])






