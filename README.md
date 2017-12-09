# Simple text classification using Multinomial Naive Bayes

## Dataset

[Ling-Spam dataset](http://csmining.org/index.php/ling-spam-datasets.html)

## Tools

- Python 3.5+
- [NLTK](http://www.nltk.org/)
- Install additional NLTK packages:

```sh
$ python3
>> import nltk
>> nltk.download('stopwords')
>> nltk.download('wordnet')
```

## Run program

Train:

```sh
python3 train.py /path/to/train_dataset
```

Test:

```sh
python3 main.py /path/to/test_dataset
```
