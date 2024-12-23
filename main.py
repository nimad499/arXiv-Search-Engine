import logging
import os
from functools import lru_cache
from pathlib import Path

import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config, pdf_download, preprocess, search, text_extract, utils
from src.config import (
    arXiv_search_query,
    pdf_dir,
    preprocessed_path,
    text_dir,
)


@lru_cache(1)
def _stop_words():
    return set(stopwords.words("english"))


def _download_pdfs():
    pdf_download.download_pdfs_from_arxiv(
        arXiv_search_query, config.download_article, pdf_dir
    )


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

    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    if not os.path.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    if config.download_article:
        _download_pdfs()

    if config.extract_text:
        _extract_text()

    if config.search:
        if config.load_preprocessed:
            titles = utils.load_preprocessed_file(preprocessed_path / "titles.pkl")
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
                PorterStemmer().stem,
                _stop_words(),
                TfidfVectorizer(),
            )

        query = config.search
        cleaned_query = preprocess.clean_text(query)
        preprocessed_query = preprocess.preprocess_text(
            cleaned_query, word_tokenize, PorterStemmer().stem, _stop_words()
        )

        top_n = config.top_n
        top_docs_idx, scores = search.search(
            preprocessed_query, tfidf_matrix, vectorizer, top_n
        )

        print(f"Top documents for query '{query}':")
        for i, idx in enumerate(top_docs_idx, 1):
            print(
                f"{i}-{utils.file_name_to_title(titles[idx])} (Score: {scores[idx]:.4f})"
            )

        print("\nPlease select an article: ", end="")
        idx = top_docs_idx[utils.get_int(1, top_n) - 1]
        pdf_file_path = Path(
            pdf_dir / ("".join(titles[idx].split(".txt")[:-1]) + ".pdf")
        )
        utils.open_file(pdf_file_path)
