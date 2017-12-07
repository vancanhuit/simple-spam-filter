import helpers
import re
import os

file_name = './datasets/3-1msg1.txt'

data = helpers.process_data(file_name)
print(data)
bags_of_words = helpers.create_bags_of_words(data)

train_data = []
train_target = []

match = re.match(r'^spm', os.path.basename(file_name))
label = 'non-spam'
if match is not None:
    label = 'spam'
train_labels.append(label)

train_dataset.append(bags_of_words)
print(train_dataset)
print(train_labels)