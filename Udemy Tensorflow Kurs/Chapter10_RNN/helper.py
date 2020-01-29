import numpy as np

def load_glove_embeddings(vocab_size, embedding_dim, data):
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('C:/Users/Jan/Documents/GLOVE/glove.6B.100d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, 100))

    for i, word in data.index_to_word.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix