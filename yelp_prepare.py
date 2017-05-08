import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("review_path")
#args = parser.parse_args()

import os
import ujson as json
import spacy
import pickle
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from yelp import *
import MySQLdb
from sklearn import preprocessing
import sys
import re
from bs4 import BeautifulSoup

en = spacy.load('en')
en.pipeline = [en.tagger, en.parser]

#def read_reviews():
#  with open(args.review_path, 'rb') as f:
#    for line in f:
#      yield json.loads(line)

def punct_space(token):
    return token.is_punct or token.is_stop or token.is_space

def stop_word(token):
    return token.is_stop

def preparation(corpus):
    return [token.lemma_ for token in en(BeautifulSoup(corpus,"html.parser").get_text())]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()
    return string



def load_data_from_db():
  db = MySQLdb.connect("10.249.71.213", "root", "root", "ai")
  cursor = db.cursor()
  sql = "SELECT DISTINCT(sr_number),t1_final,t2_final ,subject,body FROM nice_text_source_data WHERE t2_final in ('VeRO - CCR','High Risk','Defect Appeal'," \
        "'Buyer Protection Case Qs','Paying for Items','Bidding/Buying Items','Account Restriction','Report a Member/Listing','Selling Limits - CCR','Seller Risk Management'," \
        "'Logistics - CCR','eBay Account Information - CCR','Buyer Protect High ASP Claim','Cancel Transaction','Account Suspension','Buyer Protection Appeal SNAD'," \
        "'Selling Performance','Buyer Protection Escalate INR','Listing Queries - CCR','Site Features - CCR','Buyer Protection Appeal INR','Shipping - CCR') ORDER BY RAND()"

  try:
    cursor.execute(sql)
    results = cursor.fetchall()

  except:
    sys.stdout.write("Error: unable to fecth data" + '\n')

  db.close()
  return results

def build_word_frequency_distribution():
  path = os.path.join(data_dir, 'word_freq.pickle')

  try:
    with open(path, 'rb') as freq_dist_f:
      freq_dist_f = pickle.load(freq_dist_f)
      print('frequency distribution loaded')
      return freq_dist_f
  except IOError:
    pass

  print('building frequency distribution')
  freq = defaultdict(int)
  for i, review in enumerate(load_data_from_db()):
    doc = en.tokenizer((clean_str(review[3]+". "+review[4])).decode('utf8', 'ignore'))
    print "view doc"
    print doc
    for token in doc:
      freq[token.orth_] += 1
    if i % 10000 == 0:
      with open(path, 'wb') as freq_dist_f:
        pickle.dump(freq, freq_dist_f)
      print('dump at {}'.format(i))
  return freq

def build_vocabulary(lower=3, n=50000):
  try:
    with open(vocab_fn, 'rb') as vocab_file:
      vocab = pickle.load(vocab_file)
      print('vocabulary loaded')
      return vocab
  except IOError:
    print('building vocabulary')
  freq = build_word_frequency_distribution()
  top_words = list(sorted(freq.items(), key=lambda x: -x[1]))[:n-lower+1]
  vocab = {}
  i = lower
  for w, freq in top_words:
    vocab[w] = i
    i += 1
  with open(vocab_fn, 'wb') as vocab_file:
    pickle.dump(vocab, vocab_file)
  return vocab

UNKNOWN = 2

def make_data(split_points=(0.8, 0.94)):
  train_ratio, dev_ratio = split_points
  vocab = build_vocabulary()
  train_f = open(trainset_fn, 'wb')
  dev_f = open(devset_fn, 'wb')
  test_f = open(testset_fn, 'wb')
  previous_y = set()

  try:
    source = list(load_data_from_db())
    random.shuffle(source)
    for review in tqdm(source):
      x = []
      for sent in en((clean_str(review[3]+'. '+review[4])).decode('utf8', 'ignore')).sents:
        x.append([vocab.get(tok.orth_, UNKNOWN) for tok in sent])

      y = review[1]+'|'+review[2]
      previous_y.add(y)

      r = random.random()
      if r < train_ratio:
        f = train_f
      elif r < dev_ratio:
        f = dev_f
      else:
        f = test_f
      pickle.dump((review[0],x, y), f)
  except KeyboardInterrupt:
    pass

  lb = preprocessing.LabelEncoder()
  lb.fit_transform(list(previous_y))

  f = open('./y_target.pickle', 'wb')
  pickle.dump(lb, f)
  f.close()

  train_f.close()
  dev_f.close()
  test_f.close()

if __name__ == '__main__':
  make_data()