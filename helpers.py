import re
import collections
import os
import math
from functools import partial


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
