import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#from sklearn.inspection.tests.test_partial_dependence import regression_data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

np.random.seed(42)

def f(x):
    return 2.0 * x + 5.0

def classification_data():
    x1 = np.random.multivariate_normal(mean=[5.0, 0.0], cov=[[5, 0], [0, 1]], size=15)
    y1 = np.array([0 for i in range(15)])
    x2 = np.random.multivariate_normal(mean=[0.0, 0.0], cov=[[1, 0], [0, 5]], size=15)
    y2 = np.array([1 for i in range(15)])
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    return x, y

def regression_data():
    x = np.random.uniform(low=-10.0, high=10.0, size=100)
    y = f(x) + np.random.normal(scale=2.0, size=100)
    return x, y

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


x, y = regression_data()
x = x.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

regression = LinearRegression()
regression.fit(x_train, y_train)
y_pred = regression.predict(x_test)

print("R2: ", x_test, y_test)
print("MAE: ", mae(y_test, y_pred), mean_absolute_error(y_test, y_pred))
print("MSE: ", mse(y_test, y_pred), mean_squared_error(y_test, y_pred))

plt.scatter(x, y)
plt.plot(x_test, y_pred)
plt.show()
