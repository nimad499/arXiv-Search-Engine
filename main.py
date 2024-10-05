import logging

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config, downloader, pdf_tool, utils, preprocess
from src.config import (
    max_results,
    pdf_dir,
    preprocessed_path,
    search_query,
    text_dir,
)


def download_pdfs():
    downloader.download_pdfs_from_arxiv(search_query, max_results, pdf_dir)


def extract_text():
    pdf_tool.extract_text_from_pdfs(
        str(pdf_dir).encode("utf-8"), str(text_dir).encode("utf-8")
    )


def search(preprocessed_query, tfidf_matrix, vectorizer, top_n=5):
    query_str = " ".join(preprocessed_query)
    query_vec = vectorizer.transform([query_str])

    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_n_idx = cosine_similarities.argsort()[-top_n:][::-1]

    return top_n_idx, cosine_similarities


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    if config.download_article:
        download_pdfs()

    if config.extract_text:
        extract_text()

    if config.search or config.preprocess:
        if config.preprocess:
            downloader.download_nltk_modules()

        stop_words = set(stopwords.words("english"))

        if config.preprocess:
            text_data, titles = utils.load_text_files(
                text_dir, preprocess.clean_text
            )

            preprocessed_data = preprocess.parallel_preprocess_text(
                text_data, word_tokenize, stop_words
            )
            vectorizer = TfidfVectorizer()

            processed_docs = [" ".join(tokens) for tokens in preprocessed_data]

            tfidf_matrix = vectorizer.fit_transform(processed_docs)

            utils.create_preprocessed_file(
                titles, preprocessed_path / "titles.pkl"
            )
            utils.create_preprocessed_file(
                vectorizer, preprocessed_path / "vectorizer.pkl"
            )
            utils.create_preprocessed_file(
                tfidf_matrix, preprocessed_path / "tfidf_matrix.pkl"
            )
        else:
            titles = utils.load_preprocessed_file(
                preprocessed_path / "titles.pkl"
            )
            vectorizer = utils.load_preprocessed_file(
                preprocessed_path / "vectorizer.pkl"
            )
            tfidf_matrix = utils.load_preprocessed_file(
                preprocessed_path / "tfidf_matrix.pkl"
            )

        if config.search:
            query = config.search
            cleaned_query = preprocess.clean_text(query)
            preprocessed_query = preprocess.preprocess_text(
                cleaned_query, word_tokenize, stop_words
            )

            top_n = config.top_n
            top_docs_idx, scores = search(
                preprocessed_query, tfidf_matrix, vectorizer, top_n
            )

            print(f"Top documents for query '{query}':")
            for idx in top_docs_idx:
                print(
                    f"{utils.file_name_to_title(titles[idx])} (Score: {scores[idx]:.4f})"
                )
