import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
 
from plotting import *
from mnistDataAugmentation import *

# Save Path
dir_path = os.path.abspath("C:/Users/jan/Dropbox/_Programmieren/Udemy Tensorflow Kurs")
log_dir = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/Udemy Tensorflow Kurs/logs/mnist")

data = MNIST()
data.data_augmentation(augment_size=5000)
x_train, y_train = data.x_train, data.y_train
x_test, y_test = data.x_test, data.y_test
width = 28
height = 28
depth = 1
num_classes = 10
epochs = 1

# Define the DNN
input_img = Input(shape=(width, height, depth))

x = Conv2D(filters=16, kernel_size=3, padding='same')(input_img)
x = Activation("relu")(x)
x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
x = Activation("relu")(x)
x = MaxPool2D()(x)

x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
x = Activation("relu")(x)
x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
x = Activation("relu")(x)
x = MaxPool2D()(x)

x = Flatten()(x)

x = Dense(64)(x)
x = Activation("relu")(x)
x = Dense(num_classes)(x)
output_pred = Activation("softmax")(x)

model = Model(inputs=[input_img], outputs=[output_pred])

model.summary()

# Train the DNN
lr = 5e-4
optimizer = RMSprop(lr=lr)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])

tb = TensorBoard(log_dir=log_dir, 
    histogram_freq=0)

model.fit(
    x=x_train,
    y=y_train,
    verbose=1,
    batch_size=64,
    epochs=epochs,
    validation_data=[x_test, y_test],
    callbacks=[tb])

# Test the DNN
score = model.evaluate(x_test, y_test, verbose=0)
print("Score: ", score)