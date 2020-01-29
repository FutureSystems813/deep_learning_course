import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from caliHousingData import *
 
# Dataset
data = CALIHOUSING()
x_train, y_train = data.x_train, data.y_train
x_test, y_test = data.x_test, data.y_test

#scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

regr = LinearRegression()
regr.fit(x_train, y_train)
score = regr.score(x_test, y_test)
y_pred = regr.predict(x_test)
mse = mean_squared_error(y_pred, y_test)

print("Score: [", mse, ", ", score, "]")