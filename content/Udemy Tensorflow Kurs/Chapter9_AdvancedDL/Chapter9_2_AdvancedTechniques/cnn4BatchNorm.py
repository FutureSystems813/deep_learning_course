import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
 
from plotting import *
from dogsCatsData import *

file_dir = os.path.abspath("C:/Users/Jan/Documents/DogsAndCats")

data = DOGSCATS(file_dir=file_dir)
data.data_augmentation(augment_size=10000)
x_train, y_train = data.x_train, data.y_train
x_test, y_test = data.x_test, data.y_test

batch_size = 256
epochs = 10
train_size, width, height, depth = x_train.shape
test_size, num_classes = y_test.shape
width, height, depth = x_train.shape[1:]

# Define the DNN
def create_model():
    input_img = Input(shape=(width, height, depth))

    x = Conv2D(filters=16, kernel_size=3, padding="same")(input_img)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=16, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x) # 32x32x16
    #x = Dropout(rate=0.2)(x)

    x = Conv2D(filters=32, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x) # 16x16x32
    #x = Dropout(rate=0.2)(x)

    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x) # 8x8x64
    #x = Dropout(rate=0.2)(x)

    x = Flatten()(x)

    x = Dense(units=128)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #x = Dropout(rate=0.2)(x)

    x = Dense(units=num_classes)(x)
    output_pred = Activation("softmax")(x)

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
    name = "small"
    return model, name

model, name = create_model()
log_dir = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/Udemy Tensorflow Kurs/logs/dogscats/bn")

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
score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
print("Test performance: ", score)