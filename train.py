import sys
import pickle
import os
import helpers

""" Train data and store model to file """
args = sys.argv
train_dataset_path = os.path.abspath(args[1])
print('Dataset path: {}'.format(train_dataset_path))

# print('Loading dataset...')
train_target, train_data = helpers.load_dataset(train_dataset_path)

bags_of_words = helpers.create_bags_of_words(train_data)
words = helpers.get_words(bags_of_words)
labels = helpers.get_labels(train_target)

model_path = os.path.join(os.getcwd(), 'models')
if not os.path.exists(model_path):
    os.mkdir(model_path)

# print('Training data...')
label_probs, probs_per_label = helpers.train(
    bags_of_words, words, train_target, labels)

with open(os.path.join(model_path, 'train.pickle'), 'wb') as f:
    pickle.dump((label_probs, probs_per_label, words, labels), f)

print('Training done.')
print('============== INFO ===============')
print('Dataset size: {}'.format(len(bags_of_words)))
print('Total words: {}'.format(len(words)))
print('Labels: {}'.format(labels))
print('Label probs: {}'.format(label_probs))
print('====================================')
