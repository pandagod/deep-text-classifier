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


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #string = re.sub(r"\'s", " \'s", string)
    #string = re.sub(r"\'ve", " \'ve", string)
    #string = re.sub(r"n\'t", " n\'t", string)
    #string = re.sub(r"\'re", " \'re", string)
    #string = re.sub(r"\'d", " \'d", string)
    #string = re.sub(r"\'ll", " \'ll", string)


    string = re.sub(r"\s{2,}", " ", string)

    string = re.sub(r"\<", "", string)
    string = re.sub(r"\>","",string)
    string = string.strip().lower()

    return string

def test():
    db = MySQLdb.connect("10.249.71.213", "root", "root", "ai")
    cursor = db.cursor()
    sql = "SELECT DISTINCT(sr_number),t1_final,t2_final ,subject,body FROM nice_text_source_data WHERE t2_final in ('Defect Appeal' \
              'High Risk','Site Features - CCR','Selling Performance','VeRO - CCR','Bidding/Buying Items','Report a Member/Listing','Account Restriction' \
              'Cancel Transaction','Logistics - CCR','Selling Limits - CCR','Listing Queries - CCR','Paying for Items','Seller Risk Management'," \
          "'eBay Account Information - CCR','Shipping - CCR','Account Suspension','Buyer Protection Case Qs','Buyer Protect High ASP Claim'" \
          ",'Buyer Protection Appeal INR','eBay Fees - CCR','Completing a Sale - CCR') and sr_number in('1-65330829105','1-103607488805','1-85311223454','1-55106417202'" \
          ") ORDER BY RAND()"

    try:
        cursor.execute(sql)
        results = cursor.fetchall()

    except:
        sys.stdout.write("Error: unable to fecth data" + '\n')

    db.close()

    freq = defaultdict(int)
    for i, review in enumerate(results):
        raw = review[3]+'. '+review[4]
        try:
            if review != '' and detect(raw.decode('utf8')) == 'en':
                print('raw language is %s' % detect(raw.decode('utf8')))
                raw = clean_str(raw.decode('utf8'))
                sents = en(clean_str(raw)).sents

                print ("************document length is")
                print len(list(sents))
                for sent in en(raw).sents:
                    for token in en(sent.text):
                        if token.orth_=='item':
                            print "*** get item"
                        freq[token.orth_] += 1

        except:

            print review[0]
    n=50000
    lower = 3

    top_words = list(sorted(freq.items(), key=lambda x: -x[1]))[:n - lower -1]
    print top_words

    vocab = {}
    i = lower
    for w, freq in top_words:
        vocab[w] = i
        i += 1
    x = []
    UNKNOWN = 2

    for i, review in enumerate(results):
        raw = review[3]+'. '+review[4]
        try:
            if review != '' and detect(raw.decode('utf8')) == 'en':
                raw = clean_str(raw.decode('utf8'))
                print raw
                for sent in en(raw).sents:
                    x.append([vocab.get(tok.orth_, UNKNOWN) for tok in en((sent.text))])
                    print sent.text
                    print [vocab.get(tok.orth_, UNKNOWN) for tok in en((sent.text))]
        except:
            print review[0]


if __name__ == '__main__':
  test()