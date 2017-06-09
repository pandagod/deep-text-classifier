#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
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
from langdetect import detect

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
    string = re.sub(r"[^A-Za-z0-9();.,!#?’'`¥$€@//\s]", "", string)
    string = re.sub(r"\s{1,}", " ", string)
    string = re.sub(r"\?{1,}", "?", string)
    string = re.sub(r"\.{1,}", ".", string)
    string = string.strip().lower()

    return string

def load_data_from_db():
  db = MySQLdb.connect("10.249.71.213", "root", "root", "ai")
  cursor = db.cursor()
  sql = "SELECT DISTINCT(sr_number),t1_final,t2_final ,subject,body FROM nice_text_source_data WHERE t2_final in ('Defect Appeal', \
          'High Risk','Site Features - CCR','Selling Performance','VeRO - CCR','Bidding/Buying Items','Report a Member/Listing','Account Restriction', \
          'Cancel Transaction','Logistics - CCR','Selling Limits - CCR','Listing Queries - CCR','Paying for Items','Seller Risk Management',\
          'eBay Account Information - CCR','Shipping - CCR','Account Suspension','Buyer Protection Case Qs','Buyer Protect High ASP Claim',\
          'Buyer Protection Appeal INR','eBay Fees - CCR','Completing a Sale - CCR') ORDER BY RAND()"

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
      if review[3] !='':
        raw = clean_str(review[3] + ". " + review[4])
      else:
        raw = clean_str(review[4])
      try:
          if detect(raw.decode('utf-8','ignore')) == 'en':
            raw =raw.decode('utf-8','ignore')
            for sent in en(raw).sents:
                for token in sent:
                    freq[token.lemma_] += 1
            if i % 10000 == 0:
              with open(path, 'wb') as freq_dist_f:
                pickle.dump(freq, freq_dist_f)
              print('dump at {}'.format(i))
      except:
          print review[0]

  return freq

def build_vocabulary(lower=3, n=200000):
  try:
    with open(vocab_fn, 'rb') as vocab_file:
      vocab = pickle.load(vocab_file)
      print('vocabulary loaded')
      return vocab
  except IOError:
    print('building vocabulary')
  freq = build_word_frequency_distribution()

  print('vocabulary length')
  print (len(freq.items()))
  top_words = list(sorted(freq.items(), key=lambda x: -x[1]))[:n-lower-1]
  print top_words
  vocab = {}
  i = lower
  for w, freq in top_words:
    vocab[w] = i
    i += 1

  with open(vocab_fn, 'wb') as vocab_file:
    pickle.dump(vocab, vocab_file)
  return vocab

UNKNOWN = 2

def make_data(split_points=(0.9, 0.95)):
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
        if review[3] != '':
            text = clean_str(review[3] + ". " + review[4])
        else:
            text = clean_str(review[4])
        try:
            if detect(text.decode('utf-8','ignore'))=='en':
                x = []
                text = text.decode('utf-8','ignore')
                for sent in en(text).sents:
                    x.append([vocab.get(tok.lemma_, UNKNOWN) for tok in sent])
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
        except:
            print review[0]
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