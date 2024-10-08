import logging
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config, pdf_download, preprocess, search, text_extract, utils
from src.config import (
    max_results,
    pdf_dir,
    preprocessed_path,
    search_query,
    text_dir,
)


@lru_cache(1)
def _stop_words():
    return set(stopwords.words("english"))


def _download_pdfs():
    pdf_download.download_pdfs_from_arxiv(search_query, max_results, pdf_dir)


def _extract_text():
    text_extract.extract_text_from_pdfs(
        str(pdf_dir).encode("utf-8"), str(text_dir).encode("utf-8")
    )


def _download_nltk_modules():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    if config.download_article:
        _download_pdfs()

    if config.extract_text:
        _extract_text()

    if config.search:
        if config.load_preprocessed:
            titles = utils.load_preprocessed_file(
                preprocessed_path / "titles.pkl"
            )
            vectorizer = utils.load_preprocessed_file(
                preprocessed_path / "vectorizer.pkl"
            )
            tfidf_matrix = utils.load_preprocessed_file(
                preprocessed_path / "tfidf_matrix.pkl"
            )
        else:
            _download_nltk_modules()

            titles, vectorizer, tfidf_matrix = preprocess.preprocess(
                text_dir,
                preprocessed_path,
                word_tokenize,
                _stop_words(),
                TfidfVectorizer(),
            )

        query = config.search
        cleaned_query = preprocess.clean_text(query)
        preprocessed_query = preprocess.preprocess_text(
            cleaned_query, word_tokenize, _stop_words()
        )

        top_n = config.top_n
        top_docs_idx, scores = search.search(
            preprocessed_query, tfidf_matrix, vectorizer, top_n
        )

        print(f"Top documents for query '{query}':")
        for idx in top_docs_idx:
            print(
                f"{utils.file_name_to_title(titles[idx])} (Score: {scores[idx]:.4f})"
            )
