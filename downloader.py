import logging
import os
from pathlib import Path

import feedparser
import requests


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
        pdf_path = os.path.join(
            download_dir, title.replace(" ", "_").replace("/", "⧸") + ".pdf"
        )
        pdf_path = Path(pdf_path)

        if not pdf_path.exists() or re_download:
            logging.info(f"Downloading: {title}")
            response = requests.get(pdf_url, allow_redirects=True, timeout=1)

            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(response.content)

            logging.info(f"Saved: {pdf_path}")
        else:
            logging.info(f"Exist: {title}")
