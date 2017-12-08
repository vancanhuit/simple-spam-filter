import collections
import re
from functools import partial
import helpers

data = [
    'Chinese Beijing Chinese',
    'Chinese Chinese Shanghai',
    'Chinese Macao',
    'Tokyo Japan Chinese'
]

target = ['C', 'C', 'C', 'J']
labels = helpers.get_labels(target)

bags_of_words = [
    collections.Counter(re.findall(r'\w+', text)) for text in data
]

words = helpers.get_words(bags_of_words)

label_probs, probs_per_label = helpers.train(
    bags_of_words, words, target, labels)

print('Bags of words: {}'.format(bags_of_words))
print('Words: {}'.format(words))
print('Labels: {}'.format(labels))
print('Label probs: {}'.format(label_probs))
print('Probs per label: {}'.format(probs_per_label))

predictor = partial(
    helpers.predict, label_probs, probs_per_label, words, labels)

text1 = 'Chinsese Chinese Chinese Tokyo Japan'
label1 = predictor(text1)
print('Found label: {}'.format(label1))

# predictor = partial(helpers.predict, bags_of_words, words, target, labels)

# text1 = 'Chinese Chinese Chinese Tokyo Japan'

# label1 = predictor(text1)
# print(label1)
