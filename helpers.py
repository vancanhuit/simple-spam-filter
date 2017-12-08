import re
import collections
import math
import os

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from functools import partial


def pre_process_text(text):
    ''' Pre-process textual data loaded from file '''
    stop_words = set(stopwords.words('english'))
    non_words = re.compile(r'[^A-Za-z]+')

    # Lowercase text and remove non-words characters
    data = re.sub(non_words, ' ', text.lower())

    # Stem each word in text which is not a stop word
    porter_stemmer = PorterStemmer()
    words = [
        porter_stemmer.stem(w) for w in data.split() if w not in stop_words]
    return collections.Counter(words)


def load_file(file_name):
    ''' Load data and label from file '''
    text = ''
    label = ''
    with open(file_name, 'r') as f:
        text = f.read()
    
    spam = re.match(r'^spm', os.path.basename(file_name))
    if spam is not None:
        label = 'spam'
    else:
        label = 'non-spam'

    return label, text


def load_dataset(data_path):
    ''' Load data in a directory '''
    data = []
    target = []
    for file_name in os.listdir(data_path):
        label, text = load_file(os.path.join(data_path, file_name))
        data.append(text)
        target.append(label)

    return target, data


def prior(target, label):
    ''' Calculate prior probability for a label '''
    count = 0
    for t in target:
        if t == label:
            count += 1

    return count / len(target)


def get_labels(target):
    ''' Get all labels from target '''
    return list(set(target))


def get_words(bags_of_words):
    ''' Get all words from bags of words '''
    words = set()
    for row in bags_of_words:
        words = words.union(set(row.keys()))

    return words


def get_total_words(text):
    ''' Get all words in the text '''
    values = text.values()
    return sum(values)


def posterior(bags_of_words, words, word, target, label):
    ''' Calculate posterior probability of the word given label '''
    count = 0
    total = 0
    for index, text in enumerate(bags_of_words):
        if target[index] == label:
            if word in text.keys():
                count += text.get(word)
            total += get_total_words(text)

    # Normalize using Lagrange smoothing
    prob = (count + 1) / (total + len(words))
    # print('p({}|{}): {}'.format(word, label, prob))
    return prob


def get_label_probs(target, labels):
    ''' Calculate prior probability for each label '''
    probs = [prior(target, label) for label in labels]
    return probs


def get_probs_per_label(bags_of_words, words, target, labels):
    ''' Calculate conditional probability of each word for each given label '''
    logs = {}
    for word in words:
        post_prob = partial(posterior, bags_of_words, words, word, target)
        logs[word] = [post_prob(label) for label in labels]

    return logs


def train(bags_of_words, words, target, labels):
    label_probs = get_label_probs(target, labels)
    log_probs_per_label = get_probs_per_label(
        bags_of_words, words, target, labels)

    return (label_probs, log_probs_per_label)


def predict(label_probs, probs_per_label, words, labels, text):
    test_words = text.split()
    result_each_label = []

    for index, prob in enumerate(label_probs):
        result = math.log(prob)
        for word in test_words:
            if word in words:
                result += math.log(probs_per_label[word][index])

        result_each_label.append(result)

    return labels[result_each_label.index(max(result_each_label))]
