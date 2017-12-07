import re
import collections
import helpers
from functools import partial

data = [
    'hanoi pho chaolong hanoi',
    'hanoi buncha pho omai',
    'pho banhgio omai',
    'saigon hutiu banhbo pho'
]

target = ['B', 'B', 'B', 'N']
labels = helpers.get_labels(target)

bags_of_words = [
    collections.Counter(re.findall(r'\w+', text)) for text in data
]

words = helpers.get_words(bags_of_words)

predictor = partial(helpers.predict, bags_of_words, words, target, labels)

text1 = 'hanoi hanoi buncha hutiu'

label1 = predictor(text1)
print(label1)

text2 = 'pho hutiu banhbo'
label2 = predictor(text2)
print(label2)
