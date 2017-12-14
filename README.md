# Simple text classification using Multinomial Naive Bayes

## Dataset

[Ling-Spam dataset](http://csmining.org/index.php/ling-spam-datasets.html)

## Tools

- Python 3.5+
- [NLTK](http://www.nltk.org/):

    ```sh
    [sudo] pip install nltk
    ```
- Install additional NLTK packages:

    ```sh
    $ python3
    >> import nltk
    >> nltk.download('stopwords')
    >> nltk.download('wordnet')
    ```
- Install pytest for unit testing (optional):
    ```sh
    [sudo] pip install pytest
    ```
    Run unit tests:
    ```sh
    pytest helper_tests.py
    ```

## CLI

Train:

```sh
python3 train.py /path/to/train_dataset
```

Test:

```sh
python3 test.py /path/to/test_dataset
```

## GUI

```sh
python main.py
```
