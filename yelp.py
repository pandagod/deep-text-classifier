import os
import pickle
import numpy as np

train_dir = os.path.join(os.path.curdir, './emailtask')
data_dir = os.path.join(train_dir, 'data')

for dir in [train_dir, data_dir]:
  if not os.path.exists(dir):
    os.makedirs(dir)

trainset_fn = os.path.join(data_dir, 'train.dataset')
#devset_fn = os.path.join(data_dir, 'dev.dataset')
#testset_fn = os.path.join(data_dir, 'test.dataset')
devset1_fn = os.path.join(data_dir, 'dev1.dataset')
devset2_fn = os.path.join(data_dir, 'dev2.dataset')
vocab_fn = os.path.join(data_dir, 'vocab.pickle')
vocab_length_fn = os.path.join(data_dir, 'vocab_length.pickle')
target_fn = os.path.join(data_dir, 'y_target.pickle')

reserved_tokens = 5
unknown_id = 2



def vocab_size():
  vocab_len_fn = open(vocab_length_fn, 'rb')
  vocab_size = pickle.load(vocab_len_fn)
  vocab_size = vocab_size + 3
  return vocab_size


def _read_dataset(fn, review_max_sentences=30, sentence_max_length=30, epochs=1):
  f = open(target_fn, 'rb')
  lb = pickle.load(f)
  c = 0
  while 1:
    c += 1
    if epochs > 0 and c > epochs:
      return
    print('epoch %s' % c)
    with open(fn, 'rb') as f:
      try:
        while 1:
          id, x, y = pickle.load(f)
          # clip review to specified max lengths
          x = x[:review_max_sentences]
          x = [sent[:sentence_max_length] for sent in x]
          #y -= 1
          #assert y >= 0 and y <= 4

          yield x, lb.transform([y])[0]
      except EOFError:
        continue

def read_trainset(epochs=1):
  return _read_dataset(trainset_fn, epochs=epochs)

def read_devset1(epochs=1):
  return _read_dataset(devset1_fn, epochs=epochs)

def read_devset2(epochs=1):
  return _read_dataset(devset2_fn, epochs=epochs)

def read_vocab():
  with open(vocab_fn, 'rb') as f:
    return pickle.load(f)

def read_labels():
  f = open(target_fn, 'rb')
  lb = pickle.load(f)
  return lb
