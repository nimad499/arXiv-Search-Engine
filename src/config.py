import argparse
from pathlib import Path
from sys import argv

import toml

with open("config.toml", "r") as f:
    config_file = toml.load(f)

arXiv_search_query = config_file["arXiv_search_query"]
max_results = config_file["max_results"]
pdf_dir = Path(config_file["pdf_dir"])
text_dir = Path(config_file["text_dir"])
preprocessed_path = Path(config_file["preprocessed_path"])


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--download-article", action="store_true")
parser.add_argument("-e", "--extract-text", action="store_true")
parser.add_argument("-s", "--search", type=str)
parser.add_argument("-l", "--load_preprocessed", action="store_true")
parser.add_argument("-n", "--top-n", type=int, default=5)
args = parser.parse_args()
if len(argv) < 2:
    print(parser.print_help())

download_article = args.download_article
extract_text = args.extract_text
search = args.search
load_preprocessed = args.load_preprocessed
top_n = args.top_n
