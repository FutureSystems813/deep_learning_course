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
 
from imdbData2 import *

num_words = 10000
skip_top = 10
maxlen = 100
embedding_dim = 15
data = IMDB(num_words=num_words, skip_top=skip_top, maxlen=maxlen)

epochs = 5
batch_size = 128

num_classes = data.y_train.shape[1]
x_train, y_train = data.x_train, data.y_train
x_test, y_test = data.x_test, data.y_test

# Define the DNN
def create_model():
    input_review = Input(shape=(maxlen,))

    x = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen)(input_review)
    x = LSTM(units=100)(x)

    x = Dense(num_classes)(x)
    output_pred = Activation("softmax")(x)

    optimizer = Adam(
        lr=1e-3)
    model = Model(
        inputs=input_review, 
        outputs=output_pred)
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizer, 
        metrics=["accuracy"])
    model.summary()
    return model

model = create_model()
log_dir = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/Udemy Tensorflow Kurs/logs/imdb")

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