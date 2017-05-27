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

    string = string.strip().lower()

    return string

def test():
    example="Regarding: BBB Customer Review for eBay Inc.Dear Complaints BBB,A consumer has submitted a BBB Customer Review for your business, " \
            "which will be posted on your BBB Business Review.These reviews may be listed as positive, neutral or negative. Please note that negative " \
            "reviews are not formal complaints, and reviews do not affect your BBB Rating. If the customer has filed a formal complaint regarding the same " \
            "experience, the review will not be posted. What you can do:If the review meets BBB's guidelines, it will be posted on your BBB Business Review " \
            "after 5 days or after you have submitted a comment. You are welcome to submit a response that will appear alongside the review. To read the review " \
            "and submit a comment, please click on the following link:http://www.bbb.org/losangelessiliconvalley/customer-reviews/ac/10673/576a9178fde2444d .How it " \
            "works:*   BBB's Customer Reviews deliver a more accurate approach to customer feedback."\
            "*   Reviews cannot be submitted anonymously. They are authenticated by confirming the email address of the reviewer.*   Reviews are sent"
    freq = defaultdict(int)
    if example != '' and detect(example) == 'en':
        for sent in en(example.decode('utf8', 'ignore')).sents:
            for token in en(clean_str(sent.text)):
                freq[token.orth_] += 1
    n=50000
    lower = 3

    top_words = list(sorted(freq.items(), key=lambda x: -x[1]))[:n - lower + 1]

    vocab = {}
    i = lower
    for w, freq in top_words:
        vocab[w] = i
        i += 1
    x = []
    UNKNOWN = 2
    for sent in en(example.decode('utf8', 'ignore')).sents:
        print sent
        x.append([vocab.get(tok.orth_, UNKNOWN) for tok in en(clean_str(sent.text))])
        print [vocab.get(tok.orth_, UNKNOWN) for tok in en(clean_str(sent.text))]


if __name__ == '__main__':
  test()