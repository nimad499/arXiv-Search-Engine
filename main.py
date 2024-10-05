import logging
import multiprocessing as mp
import os
import pickle
from functools import partial

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config, downloader, pdf_tool
from src.config import (
    max_results,
    pdf_dir,
    preprocessed_path,
    search_query,
    text_dir,
)


def load_text_files(text_dir):
    text_data = []
    filenames = []

    for filename in os.listdir(text_dir):
        file_path = os.path.join(text_dir, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                cleaned_content = clean_text(content)
                text_data.append(cleaned_content)
                filenames.append(filename)

    return text_data, filenames


def clean_text(text):
    text = text.lower()

    return text


def preprocess_text(text):
    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens if word.isalpha() and word not in stop_words
    ]

    return tokens


def parallel_preprocess_text(text):
    num_cores = mp.cpu_count()

    with mp.Pool(processes=num_cores) as pool:
        tokens = pool.map(partial(preprocess_text), text)

    return tokens


def file_name_to_title(file_name: str):
    file_name = file_name.split(".")[0]

    file_name = file_name.replace("\n__", " ").replace("_", " ")

    return file_name


def search(query, tfidf_matrix, vectorizer, top_n=5):
    cleaned_query = clean_text(query)
    preprocessed_query = preprocess_text(cleaned_query)

    query_str = " ".join(preprocessed_query)
    query_vec = vectorizer.transform([query_str])

    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_n_idx = cosine_similarities.argsort()[-top_n:][::-1]

    return top_n_idx, cosine_similarities


if __name__ == "__main__":
    logging.getLogger().setLevel(20)

    if config.download_article:
        downloader.download_pdfs_from_arxiv(search_query, max_results, pdf_dir)

    if config.extract_text:
        pdf_tool.extract_text_from_pdfs(
            str(pdf_dir).encode("utf-8"), str(text_dir).encode("utf-8")
        )

    if config.search or config.preprocess:
        query = config.search

        if config.preprocess:
            nltk.download("punkt")
            nltk.download("punkt_tab")
            nltk.download("stopwords")

        stop_words = set(stopwords.words("english"))

        text_data, titles = load_text_files(text_dir)

        if config.preprocess:
            preprocessed_data = parallel_preprocess_text(text_data)
            with open(preprocessed_path / "data.pkl", "wb") as f:
                pickle.dump(preprocessed_data, f)
        else:
            with open(preprocessed_path / "data.pkl", "rb") as f:
                preprocessed_data = pickle.load(f)

        processed_docs = [" ".join(tokens) for tokens in preprocessed_data]

        if config.preprocess:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(processed_docs)

            with open(preprocessed_path / "vectorizer.pkl", "wb") as f:
                pickle.dump(vectorizer, f)
            with open(preprocessed_path / "tfidf_matrix.pkl", "wb") as f:
                pickle.dump(tfidf_matrix, f)
        else:
            with open(preprocessed_path / "vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)
            with open(preprocessed_path / "tfidf_matrix.pkl", "rb") as f:
                tfidf_matrix = pickle.load(f)

        if config.search:
            top_n = config.top_n
            top_docs_idx, scores = search(
                query, tfidf_matrix, vectorizer, top_n
            )

            print(f"Top documents for query '{query}':")
            for idx in top_docs_idx:
                print(
                    f"{file_name_to_title(titles[idx])} (Score: {scores[idx]:.4f})"
                )
