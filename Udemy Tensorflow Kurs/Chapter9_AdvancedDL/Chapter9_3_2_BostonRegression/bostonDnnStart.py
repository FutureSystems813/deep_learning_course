import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
 
from plotting import *

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

num_features = 13
train_size, test_size = x_train.shape[0], x_test.shape[0]

init_w = RandomUniform(minval=-1.0, maxval=1.0)
init_b = Constant(value=0.0)

# Define the DNN
model = Sequential()
model.add(Dense(16, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,)))
model.add(Activation("relu"))
model.add(Dense(1, kernel_initializer=init_w, bias_initializer=init_b))
model.summary()

def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped

# Train the DNN
lr = 0.005
optimizer = Adam(lr=lr)

model.compile(
    loss="mse",
    optimizer=optimizer,
    metrics=[r_squared])

model.fit(
    x=x_train,
    y=y_train,
    verbose=1,
    batch_size=128,
    epochs=2000,
    validation_data=[x_test, y_test])

# Test the DNN
score = model.evaluate(x_test, y_test, verbose=0)
print("Score: ", score)