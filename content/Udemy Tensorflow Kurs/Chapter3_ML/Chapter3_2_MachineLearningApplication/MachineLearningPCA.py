import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from helper import *

x, y = classification_data()

pca = PCA(n_components=1)
pca.fit(x)
x_transformed = pca.transform(x)

colors = np.array(["red", "blue"])
plt.scatter(x[:,0], x[:,1], color=colors[y[:]])
plt.show()

colors = np.array(["red", "blue"])
plt.scatter(x_transformed, [0 for i in range(len(x_transformed))], color=colors[y[:]])
plt.show()