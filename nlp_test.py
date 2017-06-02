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
    example=" Здравствуйте,Спасибо что наконец ответили,но у меня ещё остались вопросы и предложения к вам.Я прочитал всё что вы мне " \
            "прислали,но там я не нашёл решения своей проблемы.Поэтому я хочу предложить вам следующие : Разрешите мне продать мой первый " \
            "товар за сумму 7500$ и вы убедитесь что я честный и порядочный продавец,и что наше дальнейшее сотрудничество будет без плохих сюрпризов." \
            "Надеюсь на ваше понимание.С уважением Акимов Сергей.Sun, 26 Oct 2014 10:03:21 -0700 (MST) от customerhelp@ebay.com:eBay отправил это сообщение Akimov, Sergey (m8se2000)." \
            "Мы добавили ваше имя, указанное при регистрации, чтобы подтвердить, что сообщение действительно поступило от eBay.  Подробнее о том, как узнать, что это сообщение было действительно отправлено eBay. " \
            "RE: RE родажа — выставление товаров на продажу SR# 1-35775134222 SR# 1-37461525747 Здравствуйте, Сергей!Спасибо за Ваше обращение в службу поддержки клиентов eBay. Меня зовут " \
            "Агата, и я буду рада Вам помочь.Мы ценим каждого клиента и стремимся сделать платформу eBay самой удобной и безопасной для бизнеса. Пожалуйста, " \
            "извините за долгий ответ. На данный момент в нашу службу поддержки поступает большое количество писем, которые обрабатываются в порядке" \
            " очереди. льно изучив Вашу учетную запись, я вижу, что адрес Вашей регистрации Россия, поэтому позвольте мне продолжить разговор на русском.Я внимательн"

    freq = defaultdict(int)
    print detect(example.decode('utf8', 'ignore'))
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