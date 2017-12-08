import sys
from functools import partial
import helpers

args = sys.argv

train_dataset_path = args[1]
test_dataset_path = args[2]

train_target, train_data = helpers.load_dataset(train_dataset_path)
# print('Train dataset size: {}'.format(len(train_target)))

bags_of_words = helpers.create_bags_of_words(train_data)
# print('Bags of words: {}'.format(bags_of_words))

words = helpers.get_words(bags_of_words)
labels = helpers.get_labels(train_target)

label_probs, probs_per_label = helpers.train(
    bags_of_words, words, train_target, labels)

predictor = partial(
    helpers.predict, label_probs, probs_per_label, words, labels)

count = 0
test_target, test_data = helpers.load_dataset(test_dataset_path)
for index, data in enumerate(test_data):
    label = predictor(data)
    if label == test_target[index]:
        count += 1

test_data_size = len(test_data)

print('Accuracy: {0:.2f}%'.format(count / test_data_size * 100))
