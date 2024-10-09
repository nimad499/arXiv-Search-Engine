import multiprocessing as mp
from functools import partial

from . import utils


def clean_text(text):
    text = text.lower()

    return text


def preprocess_text(text, word_tokenizer, word_stemmer, stop_words):
    tokens = word_tokenizer(text)

    tokens = [
        word_stemmer(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return tokens


def parallel_preprocess_text(text, word_tokenize, word_stemmer, stop_words):
    num_cores = mp.cpu_count()

    with mp.Pool(processes=num_cores) as pool:
        tokens = pool.map(
            partial(
                preprocess_text,
                word_tokenizer=word_tokenize,
                word_stemmer=word_stemmer,
                stop_words=stop_words,
            ),
            (text),
        )

    return tokens


def preprocess(
    text_dir,
    preprocessed_path,
    word_tokenizer,
    word_stemmer,
    stop_words,
    vectorizer,
):
    text_data, titles = utils.load_text_files(text_dir, clean_text)

    preprocessed_data = parallel_preprocess_text(
        text_data, word_tokenizer, word_stemmer, stop_words
    )

    processed_docs = [" ".join(tokens) for tokens in preprocessed_data]

    tfidf_matrix = vectorizer.fit_transform(processed_docs)

    utils.create_preprocessed_file(titles, preprocessed_path / "titles.pkl")
    utils.create_preprocessed_file(
        vectorizer, preprocessed_path / "vectorizer.pkl"
    )
    utils.create_preprocessed_file(
        tfidf_matrix, preprocessed_path / "tfidf_matrix.pkl"
    )

    return titles, vectorizer, tfidf_matrix
