import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import mean_squared_error

from bostonData import *
 
# Dataset
data = BOSTON()
x_train, y_train = data.x_train, data.y_train
x_test, y_test = data.x_test, data.y_test

from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(x_train, y_train)
score = regr.score(x_test, y_test)
y_pred = regr.predict(x_test)
mse = mean_squared_error(y_pred, y_test)

print("Score: [", mse, ", ", score, "]")