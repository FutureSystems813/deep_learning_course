import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
 
from imdbData3 import *
from helper import *

log_dir = os.path.join(os.path.abspath('C:/Users/Jan/Dropbox/_Programmieren/Udemy Tensorflow Kurs/logs/imdb/'), str(time.time()))

# Load MNIST dataset
maxlen = 80
embedding_dim = 100
num_words = 1000
data = IMDB(num_words, maxlen)
vocab_size = len(data.word_index.values())
x_train, y_train = data.x_train, data.y_train
x_test, y_test = data.x_test, data.y_test
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

lr = 1e-3
optimizer = Adam
batch_size = 128
epochs = 50
train_size, words = x_train.shape
test_size, num_classes = y_test.shape

print(x_train.shape)
print(y_train.shape)

tb = TensorBoard(
    log_dir=log_dir, 
    histogram_freq=0)

callbacks = [tb]

embedding_matrix = load_glove_embeddings(vocab_size, embedding_dim, data)
print(type(embedding_matrix))
print(embedding_matrix[:5])

# Define the DNN
def create_model(optimizer, lr):
    input_text = Input(shape=(words,))

    x = Embedding(input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1], 
        input_length=maxlen,
        weights=[embedding_matrix], 
        trainable=False)(input_text)
    x = LSTM(100)(x)
    #x = Flatten()(x)
    x = Dense(num_classes)(x)

    output_pred = Activation("softmax")(x)

    optimizer = optimizer(
        lr=lr)
    model = Model(
        inputs=input_text, 
        outputs=output_pred)
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
    callbacks=callbacks)

# Test the DNN
score = model.evaluate(
    x=x_test, 
    y=y_test, 
    batch_size=batch_size,
    verbose=0)
print("Test performance: ", score)