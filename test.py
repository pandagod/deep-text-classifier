#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
from tqdm import tqdm
from collections import defaultdict
from yelp import *
import MySQLdb
from sklearn import preprocessing
import sys
import csv
import re
from bs4 import BeautifulSoup

from langdetect import detect

import en_core_web_sm

en = en_core_web_sm.load()

environment = 'prod'


def replace_entity(span, replacement):
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
            replace_entity(ent, u'PERSON')
        if ent.label_ == u'DATE':
            replace_entity(ent, u'DATE')
        if ent.label_ == u'TIME':
            replace_entity(ent, u'TIME')
            # if ent.label_ == u'ORG':
            #    replace_entity(ent,u'-ORG-')

    for token in doc:
        if token.like_url:
            token.lemma_ = u'URL'
        if token.like_email:
            token.lemma_ = u'EMAIL'
        if token.is_digit and token.lemma_ not in [u'-DATE-', u'-TIME-', u'']:
            token.lemma_ = u'NUM'
        if token.lemma_ == u'-PRON-':
            token.lemma_ = token.text


en.pipeline = [en.tagger, en.entity, en.parser, customize_rule]


# def read_reviews():
#  with open(args.review_path, 'rb') as f:
#    for line in f:
#      yield json.loads(line)

def clean_str(string):
    string = BeautifulSoup(string, "html.parser").get_text()

    string = re.sub(r"[^A-Za-z0-9();.,!#?’'`¥$€@//\s]", "", string)
    string = re.sub(r"\s{1,}", " ", string)
    string = re.sub(r"\?{1,}", "?", string)
    string = re.sub(r"\.{1,}", ".", string)
    string = string.strip().lower()

    return string



def load_data_from_db():
    db = MySQLdb.connect("10.249.71.213", "root", "root", "ai")
    cursor = db.cursor()
    sql = "SELECT sr_num,l1_topic,l2_topic ,email_body FROM 0101test"

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

        raw = clean_str(review[3])

        try:
            if detect(raw.decode('utf-8', 'ignore')) == 'en':
                raw = raw.decode('utf-8', 'ignore')
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


def build_vocabulary(lower=3):
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
    top_words = list(sorted(filter(lambda x: x[1] > 4, freq.items()), key=lambda x: -x[1]))
    print top_words
    print('top words length')
    print (len(top_words))
    vocab = {}
    i = lower
    for w, freq in top_words:
        vocab[w] = i
        i += 1

    with open(vocab_fn, 'wb') as vocab_file:
        pickle.dump(vocab, vocab_file)
    return vocab


UNKNOWN = 2


def topicMapToGroup(topic):
    if topic in ['Bidding/Buying Items','Account Safety - CCR','Unknown','Buyer Loyalty Programs','eBay Account Information - CCR','Listing Ended/Removed - Buyer','eBay Partner Sites - CCR',
        'Paying for Items','Forgot User ID or Password',
        'Search - Buying','Payment Service Account Setup',
        'Seller Suspended - Buyer',	'Registering an Account',
	    'Site Features - CCR']:
        return 'US Buying and General'
    elif topic in ['Advanced Applications']:
        return 'US Advanced Apps'
    elif topic in ['Account Closure - CCR',	'Billing Account on Hold',
        'Business Development - CCR',	'Billing Invoice',
        'Completing a Sale - CCR',	'Billing Refunds - CCR',
        'Listing Queries - CCR',	'Collections - CCR',
        'Managing Bidders/Buyers - CCR',	'eBay Fees - CCR',
        'Marketing Promotions - CCR',	'Non-Payment Suspension - CCR',
        'Search - Selling'	,'Paying eBay',
        'Selling Tools - CCR',	'Payment Service Account Funds',
        'Shipping - CCR'	,'Payment Service Fees',
        'Stores/Shops - CCR',	'Request a Credit','Logistics - CCR','Specialty Selling Approvals']:
        return 'US Selling'
    elif topic in ['Buyer Protection Case Qs',
        'Buyer Protection Program Qs',
        'Buyer Protection Refunds',
        'Cancel Transaction',
        'Contact Trading Partner - CCR',
        'Payment Service Dispute',
        'Seller Protection Policy',
        'Unpaid Item - Seller']:
        return 'US M2M Mediation'
    elif topic in ['Buyer Protection Escalate INR',
        'Buyer Protection Escalate SNAD',
        'Returns',
        'UPI Appeal - CCR']:
        return 'US M2M Escalation'
    elif topic in ['Buyer Protection Appeal INR',
        'Buyer Protection Appeal SNAD',
        'Defect Appeal',
        'Defect Basic Process']:
        return 'US M2M Appeals'
    elif topic in ['Buyer Protect High ASP Claim']:
        return 'US M2M High ASP Claim'
    elif topic in ['Account Restriction',
        'Account Suspension',
        'Buying - Rules & Policies',
        'Funds Availability - CCR',
        'High Risk',
        'INV Policies',
        'Known Good',
        'Law Enforcement - CCR',
        'Off Site Transaction - CCR',
        'Report a Member/Listing',
        'Spoof Email']:
        return 'US e2M Account'
    elif topic in ['Account Takeover']:
        return 'ATO Global'
    elif topic in ['Buying Limits - CCR'
        'Seller Vetting Restriction',
        'Selling Limits - CCR',
        'Selling Performance','Seller Risk Management']:
        return 'US e2M Limits'
    elif topic in ['CIT - Counterfeit',
        'Infringement - CCR',
        'List Practices',
        'Listing Removed - CCR',
        'Prohibited & Restricted Item']:
        return 'US e2M Listing'
    elif topic in ['VeRO - CCR']:
        return 'US e2M VeRO'
    else:
        return topic

def make_data():
    vocab = build_vocabulary()

    vocab_len_fn = open(vocab_length_fn, 'wb')
    pickle.dump(len(vocab), vocab_len_fn)
    vocab_len_fn.close()

    test_f = open(testset_fn, 'wb')

    try:
        source = list(load_data_from_db())
        random.shuffle(source)
        for review in tqdm(source):
            text = clean_str(review[3])
            try:
                if detect(text.decode('utf-8', 'ignore')) == 'en':
                    x = []
                    text = text.decode('utf-8', 'ignore')
                    for sent in en(text).sents:
                        x.append([vocab.get(tok.lemma_, UNKNOWN) for tok in sent])
                    y = topicMapToGroup(review[2])

                    f = test_f
                    pickle.dump((review[0], x, y), f)
            except:
                print review[0]
    except KeyboardInterrupt:
        pass


    test_f.close()


if __name__ == '__main__':
    make_data()
