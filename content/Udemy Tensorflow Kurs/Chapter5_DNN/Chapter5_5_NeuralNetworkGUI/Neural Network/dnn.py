import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import *
 
from plotting import *

# Save Path
dir_path = os.path.abspath("C:/Users/jan/Dropbox/_Programmieren/Udemy Tensorflow Kurs")
log_dir = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/Udemy Tensorflow Kurs/logs/mnist")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_features = 784
num_classes = 10
train_size, test_size = x_train.shape[0], x_test.shape[0]
epochs = 20

x_train = x_train.reshape(train_size, num_features)
x_test = x_test.reshape(test_size, num_features)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

init_w = TruncatedNormal(mean=0.0, stddev=0.05)
init_b = Constant(value=0.05)

# Define the DNN
model = Sequential()

model.add(Dense(500, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,)))
model.add(Activation("relu"))

model.add(Dense(300, kernel_initializer=init_w, bias_initializer=init_b))
model.add(Activation("relu"))

model.add(Dense(100, kernel_initializer=init_w, bias_initializer=init_b))
model.add(Activation("relu"))

model.add(Dense(num_classes, kernel_initializer=init_w, bias_initializer=init_b, name="output"))
model.add(Activation("softmax"))

model.summary()

# Train the DNN
lr = 5e-4
optimizer = RMSprop(lr=lr)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])

# model.fit(
#     x=x_train,
#     y=y_train,
#     verbose=1,
#     batch_size=64,
#     epochs=epochs,
#     validation_data=[x_test, y_test])

# # Test the DNN
# score = model.evaluate(x_test, y_test, verbose=0)
# print("Score: ", score)

# model.save_weights(dir_path+"/Data/dnn_mnist.h5")

def nn_predict(image):
    if image is not None:
        model.load_weights(dir_path+"/Data/dnn_mnist.h5")
        pred = model.predict(image.reshape(1, 784))[0]
        pred = np.argmax(pred, axis=0)
        return pred
    else:
        return -1