import os

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *

from caliHousingData import *

def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped

data = CALIHOUSING()
x_train, y_train = data.x_train, data.y_train
x_test, y_test = data.x_test, data.y_test

df = pd.DataFrame(data=data.x, columns=data.feature_names)
df["y"] = data.y

# print(df.head(n=20))
# print(df.info())
# print(df.describe())

df.hist(bins=30, figsize=(20,15))
plt.show()

df.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.4,
    figsize=(10,7), c="y", cmap=plt.get_cmap("jet"), colorbar=True)
plt.show()

# # Define the DNN
# def create_model():

#     optimizer = Adam()
#     model = Model(
#         inputs=input_img, 
#         outputs=output_pred)
#     model.compile(
#         loss="categorical_crossentropy", 
#         optimizer=optimizer, 
#         metrics=["accuracy"])
#     model.summary()
#     return model

# model = create_model()

# model.fit(
#     x=x_train,
#     y=y_train,
#     verbose=1,
#     batch_size=128,
#     epochs=2000,
#     validation_data=[x_test, y_test])

# # Test the DNN
# score = model.evaluate(
#     x_test, 
#     y_test)
# print("Score: ", score)