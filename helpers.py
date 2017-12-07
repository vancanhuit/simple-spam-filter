import re
import collections
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def process_data(file_name):
    """ This function pre-processing data from file """
    data = ''
    with open(file_name, 'r') as f:
        data = f.read()

    data = data.lower()
    stop_words = set(stopwords.words('english'))
    non_words = re.compile(r'[^A-Za-z]+')

    # Remove all non-words characters in string
    data = re.sub(non_words, ' ', data)

    # Used for stemming words in string
    porter_stemmer = PorterStemmer()

    # Stemming each word in string which is not a stop word
    data = ' '.join(
        [porter_stemmer.stem(w) for w in data.split() if w not instop_words])

    return data


def create_bags_of_words(text):
    """ Create bags of words from text string"""
    bags_of_words = collections.Counter(re.findall(r'\w+', text))
    return bags_of_words