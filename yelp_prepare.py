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

en = spacy.load('en')
en.pipeline = [en.tagger, en.parser]

#def read_reviews():
#  with open(args.review_path, 'rb') as f:
#    for line in f:
#      yield json.loads(line)

def load_data_from_db():
  db = MySQLdb.connect("10.249.71.213", "root", "root", "ai")
  cursor = db.cursor()
  sql = "SELECT distinct(sr_number),t1_final,t2_final,subject,body FROM text_source_data WHERE site in ('EBAY_AU','EBAY_MAIN','EBAY_CA','EBAY_UK') " \
        "and channel='Email' and body !='eBP Automation Request' and body !='' and t2_final in ('VeRO - CCR','High Risk','Site Features - CCR','Selling Limits - CCR'," \
        "'Report a Member/Listing','Shipping - CCR','Paying for Items','Advanced Applications','Cancel Transaction','Defect Appeal','Request a Credit'," \
        "'Account Suspension','Returns','Buyer Protection Case Qs','Account Restriction','eBay Account Information - CCR','Logistics - CCR','eBay Fees - CCR'," \
        "'Bidding/Buying Items','Selling Performance','Listing Queries - CCR','Seller Risk Management','Completing a Sale - CCR','Buyer Protection Refunds'," \
        "'Buyer Protect High ASP Claim','Contact Trading Partner - CCR','Buyer Protection Program Qs','Buyer Loyalty Programs','Specialty Selling Approvals') limit 100"

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
    doc = en.tokenizer((review[3]+" "+review[4]).decode('utf8', 'ignore'))
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
      for sent in en((review[3]+'. '+review[4]).decode('utf8', 'ignore')).sents:
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