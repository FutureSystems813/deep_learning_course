import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CALIHOUSING:
    x_train, y_train, x_test, y_test = None, None, None, None
    train_size, test_size = 0, 0

    def __init__(self):
        self.dataset = fetch_california_housing()
        self.x = self.dataset.data
        self.y = self.dataset.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3)
        self.feature_names = self.dataset.feature_names
        self.DESCR = self.dataset.DESCR
        # Reshape
        self.y_train = self.y_train.reshape(self.y_train.shape[0], 1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0], 1)
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]

if __name__ == "__main__":
    data = CALIHOUSING()
    df = pd.DataFrame(data=data.x, columns=data.feature_names)
    df["y"] = data.y

    print(data.DESCR)
    print(df.head(n=20))

    for col in df[:-2]:
        plt.scatter(df[col], df["y"])
        plt.xlabel(col)
        plt.ylabel("y")
        plt.show()