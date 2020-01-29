# Imports
import tensorflow as tf
import numpy as np
import os
from os import makedirs
from os.path import exists, join
from tensorflow.keras.datasets import mnist
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import *

from plotting import *
from cifar10Data import *

log_dir = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/Udemy Tensorflow Kurs/logs/cifar10/")

# Load MNIST dataset
data = CIFAR10()
x_train, y_train = data.x_train, data.y_train
x_test, y_test = data.x_test, data.y_test
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

lr = 1e-3
optimizer = Adam
batch_size = 128
epochs = 1
train_size, width, height, depth = x_train.shape
test_size, num_classes = y_test.shape

tb = TensorBoard(
    log_dir=log_dir, 
    histogram_freq=0)

# Define the DNN
def create_model(optimizer, lr):
    

    optimizer = optimizer(lr=lr)
    model = Model(inputs=input_img, outputs=output_pred)
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizer, 
        metrics=["accuracy"])
    model.summary()
    return model

model = create_model(optimizer, lr)

model.fit(
    x=x_train, 
    y=y_train, 
    verbose=1, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_data=(x_valid, y_valid), 
    callbacks=[tb])

# Test the DNN
score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
print("Test performance: ", score)