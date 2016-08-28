# coding: utf-8

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


porter = PorterStemmer()
nltk.download('stopwords')
stop = stopwords.words('english')


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def remove_stopwords(text_ary):
    return [w for w in text_ary if w not in stop]


def get_stopwords():
    return stop
