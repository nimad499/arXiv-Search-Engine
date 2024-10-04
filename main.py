import logging
import os
from ctypes import cdll
from pathlib import Path

import feedparser
import requests
import toml
from pdfminer.high_level import extract_text


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


if __name__ == "__main__":
    logging.getLogger().setLevel(20)

    with open("config.toml", "r") as f:
        config = toml.load(f)

    search_query = config["search_query"]
    max_results = config["max_results"]
    pdf_dir = Path(config["pdf_dir"])
    text_dir = Path(config["text_dir"])

    download_pdfs_from_arxiv(search_query, max_results, pdf_dir)

    extract_text_from_pdfs = cdll.LoadLibrary(
        "./lib/pdf_to_text/pdf_to_text.so"
    ).extract_text_from_pdfs_c
    extract_text_from_pdfs(
        str(pdf_dir).encode("utf-8"), str(text_dir).encode("utf-8")
    )
