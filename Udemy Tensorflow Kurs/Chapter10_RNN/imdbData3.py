import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class IMDB:
    x_train, y_train, x_test, y_test = None, None, None, None
    train_size, test_size = 0, 0
 
    def __init__(self, num_words, maxlen):
        self.num_words = num_words
        self.maxlen = maxlen
        # Word index: Word -> Index
        self.word_index = imdb.get_word_index()
        self.word_index = {k:(v+3) for k,v in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2
        # Index -> Word
        self.index_to_word = {val: key for key, val in self.word_index.items()}
        # Load dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(num_words=self.num_words)
        # Save texts
        self.x_train_text = np.array([[self.index_to_word[i] for i in x] for x in self.x_train])
        self.x_test_text = np.array([[self.index_to_word[i] for i in x] for x in self.x_test])
        # Pad Sequences
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.maxlen)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.maxlen)
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one hot array
        self.y_train = to_categorical(self.y_train, 2)
        self.y_test = to_categorical(self.y_test, 2)

if __name__ == "__main__":
    pass