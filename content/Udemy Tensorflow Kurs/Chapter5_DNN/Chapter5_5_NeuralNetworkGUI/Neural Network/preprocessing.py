# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from scipy.ndimage import center_of_mass

def load(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return image

def resize(image):
    image = cv2.resize(image, (28,28))
    return image

def normalize(image):
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return image

def center(image):
    cy, cx = center_of_mass(image)

    rows, cols = image.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    M = np.float32([[1,0,shiftx], [0,1,shifty]])
    image = cv2.warpAffine(image, M, (cols, rows))

    return image

def get_image(DrawingFrame):
    pixmap = DrawingFrame.grab()
    pixmap.save("image", "jpg")
    image = load("image").astype(np.float32)
    image = resize(image)
    image = normalize(image)
    plt.imshow(image.reshape(28,28), cmap="gray")
    plt.show()
    image = center(image)
    return image