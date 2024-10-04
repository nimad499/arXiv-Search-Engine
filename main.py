import argparse
import logging
import os
import re
from ctypes import cdll
from pathlib import Path

import feedparser
import nltk
import requests
import toml
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_pdf(pdf_file_path):
    text = extract_text(pdf_file_path)
    return text


def download_pdfs_from_arxiv(
    search_query: str,
    max_results: int,
    download_dir: Path,
    re_download=False,
):
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query={search_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    feed_url = base_url + query

    feed = feedparser.parse(feed_url)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for entry in feed.entries:
        title = entry.title

        pdf_url = entry.id.replace("abs", "pdf") + ".pdf"
        pdf_path = os.path.join(download_dir, title.replace(" ", "_") + ".pdf")
        pdf_path = Path(pdf_path)

        if not pdf_path.exists() or re_download:
            logging.info(f"Downloading: {title}")
            response = requests.get(pdf_url, allow_redirects=True, timeout=1)

            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            logging.info(f"Saved: {pdf_path}")
        else:
            logging.info(f"Exist: {title}")


def get_files_with_extension(directory, extension):
    pdf_files = []
    for item in os.listdir(directory):
        full_path = Path(os.path.join(directory, item))
        if os.path.isfile(full_path) and item.lower().endswith(
            f".{extension}"
        ):
            pdf_files.append(full_path)
    return pdf_files


def extract_text_from_pdfs(pdf_dir, text_dir):
    pdf_files = get_files_with_extension(pdf_dir, "pdf")
    for p in pdf_files:
        t = extract_text_from_pdf(p)
        with open(text_dir / Path(p.stem + ".txt"), "w") as f:
            f.write(t)
        logging.info(f"Saved: {text_dir / Path(p.stem + '.txt')}")


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

    with open("config.toml", "r") as f:
        config = toml.load(f)

    search_query = config["search_query"]
    max_results = config["max_results"]
    pdf_dir = Path(config["pdf_dir"])
    text_dir = Path(config["text_dir"])

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--download-article", action="store_true")
    parser.add_argument("-e", "--extract-text", action="store_true")
    parser.add_argument("-s", "--search", type=str)
    args = parser.parse_args()

    if args.download_article:
        download_pdfs_from_arxiv(search_query, max_results, pdf_dir)

    if args.extract_text:
        extract_text_from_pdfs = cdll.LoadLibrary(
            "./lib/pdf_to_text/pdf_to_text.so"
        ).extract_text_from_pdfs_c
        extract_text_from_pdfs(
            str(pdf_dir).encode("utf-8"), str(text_dir).encode("utf-8")
        )

    if args.search:
        query = args.search

        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("stopwords")

        stop_words = set(stopwords.words("english"))

        text_data, titles = load_text_files(text_dir)
        preprocessed_data = [preprocess_text(doc) for doc in text_data]

        processed_docs = [" ".join(tokens) for tokens in preprocessed_data]

        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(processed_docs)

        feature_names = vectorizer.get_feature_names_out()

        top_docs_idx, scores = search(query, tfidf_matrix, vectorizer)

        print(f"\nTop documents for query '{query}':")
        for idx in top_docs_idx:
            print(
                f"{file_name_to_title(titles[idx])} (Score: {scores[idx]:.4f})"
            )
