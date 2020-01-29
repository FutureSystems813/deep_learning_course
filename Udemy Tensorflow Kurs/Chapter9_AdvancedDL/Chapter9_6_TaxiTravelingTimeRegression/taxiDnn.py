import os

import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *

from taxiRoutingData import *

def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped

# Dataset
path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/Udemy Tensorflow Kurs/data/taxiDataset.xlsx")
data = ROUTING(path=path)
x_train, y_train = data.x_train, data.y_train
x_test, y_test = data.x_test, data.y_test
num_features = x_train.shape[1]

# df = data.df

# print(df.head(n=20))
# print(df.info())
# print(df.describe())

# df.hist(bins=20, figsize=(12,12))
# plt.show()

#scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# df = pd.DataFrame(x_train, columns=[
#     "Uhrzeit","Lat Start",
#     "Lon Start", "Lat Ziel",
#     "Lon Ziel","OSRM Dauer",
#     "OSRM Distanz"])
# df["y"] = y_train

# print(df.head(n=20))
# print(df.info())
# print(df.describe())

# df.hist(bins=20, figsize=(12,12))
# plt.show()

# Define the DNN
def create_model():
    input_route = Input(shape=(num_features,))

    x = Dense(units=500)(input_route)
    x = Activation("relu")(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(units=500)(x)
    x = Activation("relu")(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(units=500)(x)
    x = Activation("relu")(x)
    x = Dropout(rate=0.1)(x)

    output_pred = Dense(units=1)(x)

    optimizer = Adam()
    model = Model(
        inputs=input_route, 
        outputs=output_pred)
    model.compile(
        loss="mse", 
        optimizer=optimizer, 
        metrics=[r_squared])
    model.summary()
    return model

model = create_model()

model.fit(
    x=x_train,
    y=y_train,
    verbose=1,
    batch_size=512,
    epochs=100,
    validation_data=[x_test, y_test])

# Test the DNN
score = model.evaluate(
    x_test, 
    y_test,
    verbose=0)
print("Score: ", score)

y_pred = model.predict(x_test)

sns.residplot(y_test, y_pred, scatter_kws={"s": 2, "alpha": 0.5})
plt.show()