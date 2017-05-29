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
    example="Sent:	2016-10-05 18:10:16.0 Subject:	RE: Regarding the 6tb Sata Iii 6gb, item Enter your response here:	Hi I am getting nowhere with the Global Shipping team and after contacting PayPal (with eBay Rep on the phone) they confirmed my account is and always has been in good standing with absolutely no issues.  At this point I would like your escalation protocols on how to speak directly to a manger or contact information to your legal dept if a manager is not available.Without Prejudice,Richard 4168430842  Previous Message Follows-------------------------   .header{background-color:#f9f9f9;font-family:Helvetica Neue,Helvetica, Arial; Light;font-size:11px;}body{margin:16px;color:#333333;background-color:#f9f9f9;font-family:Helvetica Neue,Helvetica, Arial; Reg;font-size:14px;font-weight:normal;font-style:normal;line-height: 1.2;}#email p{margin: 14px 0;}#email a{color:#0563C1;}.subject{ font-size: 18px; margin: 16px 0 30px 0;}.thanks{margin: 36px 0 90px 0 !important;}.footer{background-color:#FFFFFF;font-family:Helvetica Neue,Helvetica, Arial; Light;font-size:11px;}.info{color:#767676;font-size:12px;font-family:font-family:Helvetica Neue,Helvetica, Arial; Light; }RE: Regarding the 6tb Sata Iii 6gb, item # 331928832470  SR# 1-85213632668Dear Richard, Thank you for reaching back out to us about the refund for the shipping on the 6TB hard drive, item 331928832470. I understand you havent seen in your account. I understand the distress of not s"
    freq = defaultdict(int)
    if example != '' and detect(example.decode('utf8', 'ignore')) == 'en':
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