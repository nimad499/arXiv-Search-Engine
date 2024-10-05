import logging
import os
from ctypes import cdll

from pathlib import Path

from pdfminer.high_level import extract_text


def get_files_with_extension(directory, extension):
    pdf_files = []
    for item in os.listdir(directory):
        full_path = Path(os.path.join(directory, item))
        if os.path.isfile(full_path) and item.lower().endswith(
            f".{extension}"
        ):
            pdf_files.append(full_path)
    return pdf_files


def extract_text_from_pdf(pdf_file_path):
    text = extract_text(pdf_file_path)
    return text


def _extract_text_from_pdfs(pdf_dir, text_dir):
    pdf_files = get_files_with_extension(pdf_dir, "pdf")
    for p in pdf_files:
        t = extract_text_from_pdf(p)
        with open(text_dir / Path(p.stem + ".txt"), "w") as f:
            f.write(t)
        logging.info(f"Saved: {text_dir / Path(p.stem + '.txt')}")


extract_text_from_pdfs = cdll.LoadLibrary(
    "./lib/pdf_to_text/pdf_to_text.so"
).extract_text_from_pdfs_c
