import re
import collections
import os
import math


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
    for index, row in enumerate(bags_of_words):
        if target[index] == label:
            if word in row.keys():
                count += row.get(word)
            total += get_total_words(row)

    # Normalize using Lagrange smoothing
    return (count + 1) / (total + len(words))


def predict(bags_of_words, words, target, labels, text):
    ''' Predict a label for a given text '''
    test_words = text.split()

    log_probs_per_label = []

    # Calculate log-probability of each word for each label
    for word in test_words:
        logs = [
            math.log(posterior(
                bags_of_words, words, word, target, label)) for label in labels
        ]
        log_probs_per_label.append(logs)

    # Calculate final result for each label
    result_each_label = []
    for index, label in enumerate(labels):
        result = math.log(prior(target, label))
        for pos, word in enumerate(test_words):
            result += log_probs_per_label[pos][index]
        result_each_label.append(result)

    # Select label which has greatest result
    return labels[result_each_label.index(max(result_each_label))]
