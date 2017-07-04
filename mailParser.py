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

import talon
# don't forget to init the library first
# it loads machine learning classifiers
talon.init()

from talon import signature

from email_reply_parser import EmailReplyParser
from spacy.symbols import ORTH, LEMMA, POS

from talon.signature.bruteforce import extract_signature

en = spacy.load('en')

def replace_entity(span,replacement):
    i = 1
    for token in span:
        if i == span.__len__():
            token.lemma_ = replacement
        else:
            token.lemma_ = u''
        i += 1

def customize_rule(doc):
    for ent in doc.ents:
        if ent.label_ == u'PERSON':
            replace_entity(ent,u'PERSON')
        if ent.label_ == u'DATE':
            replace_entity(ent,u'DATE')
        if ent.label_ == u'TIME':
            replace_entity(ent,u'TIME')
        #if ent.label_ == u'ORG':
        #    replace_entity(ent,u'ORG')

    for token in doc:
        if token.like_url:
            token.lemma_ = u'URL'
        if token.like_email:
            token.lemma_ = u'EMAIL'
        if token.is_digit and token.lemma_ not in [u'DATE',u'TIME',u'']:
            token.lemma_ = u'NUM'
        if token.lemma_ == u'-PRON-':
            token.lemma_ = token.text

en.pipeline = [en.tagger,en.entity,en.parser,customize_rule]

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9;.,!?’'`:/¥$€@\s]", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\w/{1}\w", "", string)
    string = re.sub(r"\?{1,}", "?", string)
    string = re.sub(r"\.{1,}", ".", string)

    return string



def spacy_test():
    db = MySQLdb.connect("10.249.71.213", "root", "root", "ai")
    cursor = db.cursor()
    sql = "SELECT DISTINCT(sr_number),t1_final,t2_final ,subject,body FROM nice_text_source_data WHERE t2_final in ('Defect Appeal' \
                  'High Risk','Site Features - CCR','Selling Performance','VeRO - CCR','Bidding/Buying Items','Report a Member/Listing','Account Restriction' \
                  'Cancel Transaction','Logistics - CCR','Selling Limits - CCR','Listing Queries - CCR','Paying for Items','Seller Risk Management'," \
          "'eBay Account Information - CCR','Shipping - CCR','Account Suspension','Buyer Protection Case Qs','Buyer Protect High ASP Claim'" \
          ",'Buyer Protection Appeal INR','eBay Fees - CCR','Completing a Sale - CCR') and sr_number='1-107364398873'"

    try:
        cursor.execute(sql)
        results = cursor.fetchall()

    except:
        sys.stdout.write("Error: unable to fecth data" + '\n')

    db.close()


    for i, review in enumerate(results):
        #result=EmailReplyParser.read(review[4])
        text, asignature = signature.extract(review[4],'Mike Ogryzlo')
        print text
        print '~~~~~~~~~~~~~~~~~~~~~~~~~'
        print asignature

def test():
    db = MySQLdb.connect("10.249.71.213", "root", "root", "ai")
    cursor = db.cursor()
    sql = "SELECT DISTINCT(sr_number),t1_final,t2_final ,subject,body FROM nice_text_source_data WHERE t2_final in ('Defect Appeal' \
              'High Risk','Site Features - CCR','Selling Performance','VeRO - CCR','Bidding/Buying Items','Report a Member/Listing','Account Restriction' \
              'Cancel Transaction','Logistics - CCR','Selling Limits - CCR','Listing Queries - CCR','Paying for Items','Seller Risk Management'," \
          "'eBay Account Information - CCR','Shipping - CCR','Account Suspension','Buyer Protection Case Qs','Buyer Protect High ASP Claim'" \
          ",'Buyer Protection Appeal INR','eBay Fees - CCR','Completing a Sale - CCR') limit 3"

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
            if review != '' and detect(raw.decode('utf8','ignore')) == 'en':
                raw = clean_str(raw.decode('utf8'))
                sents = en(clean_str(raw)).sents

                print ("************document length is")
                print len(list(sents))
                for sent in en(raw).sents:
                    for token in en(sent.text):
                        freq[token.lemma_] += 1

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
            if review != '' and detect(raw.decode('utf8','ignore')) == 'en':
                raw = clean_str(raw.decode('utf8'))
                print raw
                for sent in en(raw).sents:
                    x.append([vocab.get(tok.lemma_, UNKNOWN) for tok in en((sent.text))])
                    print sent.text
                    print [vocab.get(tok.lemma_, UNKNOWN) for tok in en((sent.text))]
        except:
            print review[0]


if __name__ == '__main__':
    spacy_test()
    print "over"