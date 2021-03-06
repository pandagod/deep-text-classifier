import numpy as np


def batch(inputs):
  batch_size = len(inputs)

  document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
  document_size = document_sizes.max()

  sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
  max_sentence_size = max(map(max, sentence_sizes_))

  b = np.zeros(shape=[batch_size, document_size, max_sentence_size], dtype=np.int32) # == PAD

  sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
  for i, document in enumerate(inputs):
    for j, sentence in enumerate(document):
      sentence_sizes[i, j] = sentence_sizes_[i][j]
      for k, word in enumerate(sentence):
        b[i, j, k] = word

  return b, document_sizes, sentence_sizes

def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors