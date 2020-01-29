import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper import *

x, y = classification_data()

m = -4
b = 7
border = [m * xi + b for xi in x]

colors = np.array(["red", "blue"])
plt.scatter(x[:,0], x[:,1], color=colors[y[:]])
plt.plot(x, border)
plt.show()