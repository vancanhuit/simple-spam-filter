import pytest
import collections
import helpers


class TestClass(object):
    ''' Test some functions in helpers module '''
    def test_prior(self):
        target = ['A', 'A', 'A', 'B']
        prob_A = helpers.prior(target, 'A')
        prob_B = helpers.prior(target, 'B')
        print(prob_A)
        print(prob_B)
        assert prob_A == pytest.approx(0.75)
        assert prob_B == pytest.approx(0.25)

    def test_get_labels(self):
        target = ['A', 'A', 'B']
        labels = helpers.get_labels(target)
        assert len(labels) == 2
        assert 'A' in labels
        assert 'B' in labels

    def test_get_words(self):
        bags_of_words = [{'aa': 2, 'bb': 1, 'cc': 3}, {'cc': 2, 'dd': 1}]
        words = helpers.get_words(bags_of_words)
        print(words)
        assert len(words) == 4
        assert 'aa' in words
        assert 'bb' in words
        assert 'cc' in words
        assert 'dd' in words

    def test_get_total_words(self):
        text = {'aa': 2, 'bb': 1, 'cc': 3}
        total = helpers.get_total_words(text)
        assert total == 6

    def test_posterior(self):
        bags_of_words = [{'aa': 2, 'bb': 1}, {'cc': 1}, {'dd': 2}]
        words = {'aa', 'bb', 'cc', 'dd'}
        word = 'aa'
        target = ['A', 'A', 'B']
        label = 'A'

        prob = helpers.posterior(bags_of_words, words, word, target, label)
        print(prob)
        assert prob == pytest.approx(3 / 8)

    def test_pre_process_text(self):
        text = 'a is go'
        new_text = helpers.pre_process_text(text)
        print(new_text)
        assert 'a' not in new_text

    def test_create_bags_of_words(self):
        data = [['aa', 'bb', 'cc' 'cc'], ['aa', 'bb', 'bb', 'cc']]
        bags_of_words = helpers.create_bags_of_words(data)

        txt = bags_of_words[0]

        print('\nBags of words: {}'.format(bags_of_words))
        assert len(bags_of_words) == 2
        assert type(txt) is collections.Counter
        assert 'aa' in txt.keys()
        assert txt['aa'] == 1
