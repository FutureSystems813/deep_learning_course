import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class IMDB:
    def __init__(self, num_words, skip_top, maxlen):
        self.num_classes = 2
        self.num_words = num_words
        self.skip_top = skip_top
        self.maxlen = maxlen
        # Word index: Word -> Index
        self.word_index = imdb.get_word_index()
        self.word_index = {k: (v+3) for k,v in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2
        # Index -> Word
        self.index_to_word = {val: key for key, val in self.word_index.items()}
        # Load Dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(
            num_words=self.num_words,
            skip_top=self.skip_top)
        # Save text
        self.x_train_text = np.array([[self.index_to_word[i] for i in x] for x in self.x_train])
        self.x_test_text = np.array([[self.index_to_word[i] for i in x] for x in self.x_test])
        # Pad Sequences
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.maxlen)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.maxlen)
        # Dataset Parameters
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one-hot vectors for classes
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)

if __name__ == "__main__":
    pass