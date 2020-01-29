import os
import time

import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

# Define the DNN
def create_model():
    optimizer = Adam(
        lr=1e-3)
    model = Model(
        inputs=input_img, 
        outputs=output_pred)
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizer, 
        metrics=["accuracy"])
    model.summary()
    return model

model = create_model()
log_dir = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/Udemy Tensorflow Kurs/logs/dogscats")

tb = TensorBoard(
    log_dir=log_dir, 
    histogram_freq=0)

model.fit(
    x=x_train, 
    y=y_train, 
    verbose=1, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_data=(x_test, y_test), 
    callbacks=[tb])

# Test the DNN
score = model.evaluate(
    x=x_test, 
    y=y_test, 
    batch_size=batch_size,
    verbose=0)
print("Test performance: ", score)