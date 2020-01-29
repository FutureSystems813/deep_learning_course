import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ROUTING:
    x_train, y_train, x_test, y_test = None, None, None, None
    train_size, test_size = 0, 0

    def __init__(self, path):
        self.df = pd.read_excel(open(
            path, "rb"))
        self.df = pd.DataFrame(self.df,
            columns=["Uhrzeit","Straße Start","Nr Start",
            "Stadt Start","Lat Start","Lon Start",
            "Straße Ziel","Nr Ziel","Stadt Ziel",
            "Lat Ziel","Lon Ziel","OSRM Dauer","OSRM Distanz", "y"])

        self.x = self.df.loc[:, ["Uhrzeit","Lat Start","Lon Start",
            "Lat Ziel","Lon Ziel","OSRM Dauer","OSRM Distanz"]]
        self.y = self.df["y"]

        self.x = self.x.to_numpy()
        self.y = self.y.to_numpy()

        self.x[:, 0] = [float(val[:2])*60 + float(val[3:5]) for val in self.x[:, 0]]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3)
        
if __name__ == "__main__":
    path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/Udemy Tensorflow Kurs/data/taxiDataset.xlsx")
    data = ROUTING(path=path)