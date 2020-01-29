import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class MNIST:
    def __init__(self):
        self.width = 28
        self.height = 28
        self.depth = 1
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.train_size, self.test_size = self.x_train.shape[0], self.x_test.shape[0]
        # Reshape x data
        self.x_train = self.x_train.reshape(self.train_size, self.width, self.height, self.depth)
        self.x_test = self.x_test.reshape(self.test_size, self.width, self.height, self.depth)
        # Create one hot arrays
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)
        # Change dtype from int to float
        self.x_train = self.x_train.astype("float32")
        self.x_test = self.x_test.astype("float32")
        # z-score Standardisierung
        mean = np.mean(self.x_train)
        stddev = np.std(self.x_train)
        self.x_train -= mean
        self.x_train /= stddev
        self.x_test -= mean
        self.x_test /= stddev

if __name__ == "__main__":
    pass