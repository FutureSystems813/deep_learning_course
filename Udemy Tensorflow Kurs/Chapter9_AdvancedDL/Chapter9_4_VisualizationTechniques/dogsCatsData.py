from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
from skimage import transform
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

file_dir = os.path.abspath("C:/Users/Jan/Documents/DogsAndCats")
img_size = 64
img_depth = 3
classes = 2

# Load Cats (0) and Dogs (1) from images to NumpyArray
def extract_cats_vs_dogs(file_dir):
    cats_dir = file_dir + "/cat/"
    dogs_dir = file_dir + "/dog/"

    print("Delete no jpg images!")
    for f in os.listdir(cats_dir):
        if f.split(".")[-1] != "jpg":
            print("Removing file: ", f)
            os.remove(cats_dir + f)

    print("Delete no jpg images!")
    for f in os.listdir(dogs_dir):
        if f.split(".")[-1] != "jpg":
            print("Removing file: ", f)
            os.remove(dogs_dir + f)

    num_cats = len([name for name in os.listdir(cats_dir)])
    num_dogs = len([name for name in os.listdir(dogs_dir)])
    
    x = np.empty(
        shape=(num_cats+num_dogs, img_size, img_size, img_depth),
        dtype=np.float32)
    y = np.zeros(
        shape=(num_cats+num_dogs),
        dtype=np.int8)

    cnt = 0

    print("start reading cat images!")
    for f in os.listdir(cats_dir):
        try:
            img = cv2.imread(cats_dir + f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x[cnt] = transform.resize(img, (img_size, img_size, img_depth))
            y[cnt] = 0
            cnt += 1
        except:
            pass

    print("start reading dog images!")
    for f in os.listdir(dogs_dir):
        try:
            img = cv2.imread(dogs_dir + f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x[cnt] = transform.resize(img, (img_size, img_size, img_depth))
            y[cnt] = 1
            cnt += 1
        except:
            pass

    np.save(file_dir + "/x.npy", x)
    np.save(file_dir + "/y.npy", y)

def load_cats_vs_dogs(file_dir):
    x = np.load(file_dir + "/x.npy")
    y = np.load(file_dir + "/y.npy")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return (x_train, y_train), (x_test, y_test)

class DOGSCATS:
    def __init__(self, file_dir):
        self.width = 64
        self.height = 64
        self.depth = 3
        self.num_classes = 2
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cats_vs_dogs(file_dir)
        self.train_size, self.test_size = self.x_train.shape[0], self.x_test.shape[0]
        # Reshape x data
        self.x_train = self.x_train.reshape(self.train_size, self.width, self.height, self.depth)
        self.x_test = self.x_test.reshape(self.test_size, self.width, self.height, self.depth)
        # Create one hot arrays
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)
        # Change dtype from int to float
        self.x_train = self.x_train.astype("float32")
        self.x_test = self.x_test.astype("float32")
        self.x_images = self.x_test.copy()
        # z-score Standardisierung
        mean = np.mean(self.x_train)
        stddev = np.std(self.x_train)
        self.x_train -= mean
        self.x_train /= stddev
        self.x_test -= mean
        self.x_test /= stddev

    def data_augmentation(self, augment_size=5000):
        image_generator = ImageDataGenerator(
            rotation_range=15.0,
            zoom_range=0.10,
            width_shift_range=0.10,
            height_shift_range=0.10)
        # fit data for augmenting
        image_generator.fit(self.x_train, augment=True)
        # get transformed images
        randidx = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[randidx].copy()
        x_copy = x_augmented.copy()
        y_augmented = self.y_train[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
            batch_size=augment_size, shuffle=False).next()[0]
        # append augmented data to trainset
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]

if __name__ == "__main__":
    pass