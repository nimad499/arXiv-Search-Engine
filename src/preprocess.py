import multiprocessing as mp
from functools import partial


def clean_text(text):
    text = text.lower()

    return text


def preprocess_text(text, word_tokenizer, stop_words):
    tokens = word_tokenizer(text)

    tokens = [
        word for word in tokens if word.isalpha() and word not in stop_words
    ]

    return tokens


def parallel_preprocess_text(text, word_tokenize, stop_words):
    num_cores = mp.cpu_count()

    with mp.Pool(processes=num_cores) as pool:
        tokens = pool.map(
            partial(
                preprocess_text,
                word_tokenizer=word_tokenize,
                stop_words=stop_words,
            ),
            (text),
        )

    return tokens
