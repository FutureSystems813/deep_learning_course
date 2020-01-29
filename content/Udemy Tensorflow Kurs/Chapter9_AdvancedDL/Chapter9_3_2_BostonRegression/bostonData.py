import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.datasets import boston_housing

class BOSTON:
    x_train, y_train, x_test, y_test = None, None, None, None
    train_size, test_size = 0, 0

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = boston_housing.load_data()
        # Reshape
        self.y_train = self.y_train.reshape(self.y_train.shape[0], 1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0], 1)
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Data rescaling
        #scaler = StandardScaler()
        scaler = MinMaxScaler()
        scaler.fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)