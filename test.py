import pickle
import sys
import os
from functools import partial
import helpers

""" Test dataset """
args = sys.argv

model_path = os.path.join(os.getcwd(), 'model')
input_path = os.path.abspath(args[1])
print('Test data path: {}'.format(input_path))

print('Getting model...')
result = None
with open(os.path.join(model_path, 'train.pickle'), 'rb') as f:
    result = pickle.load(f)

label_probs = result[0]
probs_per_label = result[1]
words = result[2]
labels = result[3]

predictor = partial(helpers.predict,
                    label_probs, probs_per_label, words, labels)

if os.path.isdir(input_path):
    print('Loading dataset...')
    test_target, test_data = helpers.load_dataset(input_path)

    print('Testing dataset...')
    print("============= RESULT ===============")
    accuracy = helpers.get_accuracy(test_data, test_target, predictor)
    print('Accuracy: {0:.2f}%'.format(accuracy * 100))
    print('=====================================')
else:
    print('Testing...')
    label, tokens = helpers.load_file(input_path)
    predict_label = predictor(tokens)
    print('================= RESULT ================')
    print('Expected label for the text: {}'.format(label))
    print('Predicted label for the text: {}'.format(predict_label))
    print('==========================================')
