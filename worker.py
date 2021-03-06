#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='yelp', choices=['yelp'])
parser.add_argument('--mode', default='train', choices=['train', 'eval'])
parser.add_argument('--checkpoint-frequency', type=int, default=500)
parser.add_argument('--eval-frequency', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument("--device", default="/cpu:0")
parser.add_argument("--max-grad-norm", type=float, default=5.0)
parser.add_argument("--lr", type=float, default=0.0001)
args = parser.parse_args()

import importlib
import os
import errno
import pickle
import random
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import yaml
from tqdm import tqdm
import sys

import ujson
from data_util import batch

task_name = args.task

task = importlib.import_module(task_name)

checkpoint_dir = os.path.join(task.train_dir, 'checkpoint')
tflog_dir = os.path.join(task.train_dir, 'tflog')
checkpoint_name = task_name + '-model'
checkpoint_dir = os.path.join(task.train_dir, 'checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

# @TODO: move calculation into `task file`
trainset = task.read_trainset(epochs=1)

class_weights = pd.Series(Counter([l for _, l in trainset]))
class_weights = 1/(class_weights/class_weights.mean())
class_weights = class_weights.to_dict()

vocab = task.read_vocab()
labels = task.read_labels()

#classes = max(labels.values())+1
classes = len(labels.classes_)

vocab_size = task.vocab_size()
print "vocab size length"
print vocab_size

#labels_rev = {v: k for k, v in labels.tolist()}
vocab_rev = {int(v): k for k, v in vocab.items()}

#print labels_rev


def HAN_model_1(session, restore_only=False):
  """Hierarhical Attention Network"""
  import tensorflow as tf
  try:
    from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, DropoutWrapper
  except ImportError:
    MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
    GRUCell = tf.nn.rnn_cell.GRUCell
  from bn_lstm import BNLSTMCell
  from HAN_model import HANClassifierModel

  is_training = tf.placeholder(dtype=tf.bool, name='is_training')

  cell = BNLSTMCell(80, is_training) # h-h batchnorm LSTMCell
  # cell = GRUCell(30)
  cell = MultiRNNCell([cell]*2)

  model = HANClassifierModel(
      vocab_size=vocab_size,
      embedding_size=200,
      classes=classes,
      word_cell=cell,
      sentence_cell=cell,
      word_output_size=100,
      sentence_output_size=100,
      max_grad_norm=args.max_grad_norm,
      is_training =is_training,
      learning_rate=args.lr,
      device=args.device,
  )

  saver = tf.train.Saver(tf.global_variables())
  checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
  if checkpoint:
    sys.stdout.write("Reading model parameters from %s \n" % checkpoint.model_checkpoint_path)
    saver.restore(session, checkpoint.model_checkpoint_path)
  elif restore_only:
    print("Cannot restore model")
  else:
    print("Created model with fresh parameters")
    session.run(tf.global_variables_initializer())
  # tf.get_default_graph().finalize()
  return model, saver

model_fn = HAN_model_1

def decode(ex):
  print('text: ' + '\n'.join([' '.join([vocab_rev.get(wid, '<?>') for wid in sent]) for sent in ex[0]]))
  print('label: ', labels.transform([ex[1]]))

sys.stdout.write('data loaded \n')

def batch_iterator(dataset, batch_size, max_epochs):
  for i in range(max_epochs):
    xb = []
    yb = []
    for ex in dataset:
      x, y = ex
      xb.append(x)
      yb.append(y)
      if len(xb) == batch_size:
        yield xb, yb
        xb, yb = [], []

def dev_iterator(dataset):
  xb = []
  yb = []
  for ex in dataset:
    x, y = ex
    xb.append(x)
    yb.append(y)
  return (xb,yb)

def ev(session, model, dataset):
  predictions = []
  labels = []
  examples = []
  for x, y in tqdm(batch_iterator(dataset, 64, 1)):
    examples.extend(x)
    labels.extend(y)
    predictions.extend(session.run(model.prediction, model.get_feed_data(x, dropout_keep_proba=1)))

  df = pd.DataFrame({'predictions': predictions, 'labels': labels, 'examples': examples})
  return df

def evaluate(dataset):
  tf.reset_default_graph()
  config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Session(config=config) as s:
    model, _ = model_fn(s, restore_only=True)
    df = ev(s, model, dataset)
    print((df['predictions'] == df['labels']).mean())
  #import IPython
  #IPython.embed()

def train():
  tf.reset_default_graph()

  config = tf.ConfigProto(allow_soft_placement=True)

  with tf.Session(config=config) as s:
    model, saver = model_fn(s)
    train_summary_dir = os.path.join(tflog_dir,"train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph=tf.get_default_graph())

    dev1_summary_dir = os.path.join(tflog_dir, "dev1")
    dev1_summary_writer = tf.summary.FileWriter(dev1_summary_dir, graph=tf.get_default_graph())

    dev2_summary_dir = os.path.join(tflog_dir, "dev2")
    dev2_summary_writer = tf.summary.FileWriter(dev2_summary_dir, graph=tf.get_default_graph())

    global_step = model.global_step

    def train_step(x, y):

      fd = model.get_feed_data(x, y, class_weights=class_weights,dropout_keep_proba=0.5)
      t0 = time.clock()
      step, summaries, loss, accuracy, _ = s.run([
        model.global_step,
        model.summary_op,
        model.loss,
        model.accuracy,
        model.train_op,
      ], fd)

      td = time.clock() - t0
      sys.stdout.write('step %s, loss=%s, accuracy=%s, t=%s, inputs=%s \n' % (step, loss, accuracy, round(td, 5), fd[model.inputs].shape))
      train_summary_writer.add_summary(summaries, global_step=step)

    def dev_step(x,y,name):
      fd = model.get_feed_data(x, y,dropout_keep_proba=1)
      step, summaries, loss, accuracy = s.run([
        model.global_step,
        model.summary_op,
        model.loss,
        model.accuracy
      ], fd)

      sys.stdout.write('evaluation at step %s \n' % step)
      sys.stdout.write('dev accuracy: %.5f \n' % accuracy)
      if name =='dev1':
        dev1_summary_writer.add_summary(summaries, global_step=step)
      elif name=='dev2':
        dev2_summary_writer.add_summary(summaries, global_step=step)

    devset1 = task.read_devset1(epochs=1)
    devset2 = task.read_devset2(epochs=1)
    dev1_x, dev1_y = dev_iterator(devset1)
    dev2_x, dev2_y = dev_iterator(devset2)
    for i, (x, y) in enumerate(batch_iterator(task.read_trainset(epochs=90), args.batch_size, 300)):
      train_step(x,y)
      current_step =tf.train.global_step(s,global_step)
      if current_step != 0 and current_step % args.checkpoint_frequency == 0:
        sys.stdout.write('checkpoint & graph meta \n')
        saver.save(s, checkpoint_path, global_step=current_step)
        sys.stdout.write('checkpoint done \n')
      if current_step != 0 and current_step % args.eval_frequency == 0:
        dev_step(dev1_x[0:1024], dev1_y[0:1024], "dev1")
        dev_step(dev2_x[0:1024], dev2_y[0:1024], "dev2")

def main():
  if args.mode == 'train':
    train()
  elif args.mode == 'eval':
    evaluate(task.read_testset(epochs=1))

if __name__ == '__main__':
  main()
