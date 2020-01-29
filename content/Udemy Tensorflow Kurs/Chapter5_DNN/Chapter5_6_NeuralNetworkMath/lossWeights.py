import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def f(x):
    return x**2 + x + 10

x = np.linspace(start=-10.0, stop=10.0, num=1000).reshape(-1, 1)
y = f(x)

model = Sequential()
model.add(Dense(12)) # Input zu Hidden
model.add(Activation("relu")) # ReLU vom Hidden
model.add(Dense(1)) # Vom Hidden zum Output
model.compile(optimizer=RMSprop(lr=1e-2), loss="mse") # 1e-3 <=> 0.001
model.fit(x, y, epochs=20)
y_pred = model.predict(x).flatten()
W, b = model.layers[0].get_weights()
print("Weights: ", W[0][0])

w = np.linspace(start=-5, stop=5, num=200)
losses = []
for wi in w:
    W, b = model.layers[0].get_weights()
    W[0][0] = wi
    model.layers[0].set_weights((W, b))
    new_pred = model.predict(x).flatten()
    loss = mean_squared_error(y, new_pred)
    losses.append(loss)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
ax1.grid(True)
ax2.grid(True)
ax1.plot(x, y, color="blue")
ax1.plot(x, y_pred, color="red")
ax2.plot(w, losses, color="orange")
plt.show()