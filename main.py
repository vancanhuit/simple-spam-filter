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

test_target, test_data = helpers.load_dataset(test_dataset_path)
accuarcy = helpers.get_accuracy(test_data, test_target, predictor)

print('Accuracy: {0:.2f}%'.format(accuarcy * 100))
