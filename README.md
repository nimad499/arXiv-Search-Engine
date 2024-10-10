# arXiv Search Engine

A search engine designed to download large amounts of articles from arXiv with a 
specific query, extract text from the PDFs, vectorize the text using TF-IDF, and 
enable searching within the extracted corpus.

## Features

* Download articles from arXiv based on a specified search query
* Extract text from downloaded PDFs
* Vectorize extracted text using Term Frequency-Inverse Document Frequency (TF-IDF)
* Search within the extracted corpus using cosine similarity

## Configuration

### 1. `config.toml`

This file contains configuration settings in TOML format:
```toml
arXiv_search_query = "cat:cs.LG+OR+cat:stat.ML"
pdf_dir = "data/pdf"
text_dir = "data/text"
preprocessed_path = "data/preprocessed"
```
### 2. CLI Flags

The following flags can be passed as command-line arguments:

* `-d`, `--download-article`: Download articles from arXiv based on the specified 
search query in the `config.toml`
* `-e`, `--extract-text`: Extract text from downloaded PDFs 
* `-s`, `--search`: Search within the extracted corpus using cosine similarity
* `-l`, `--load_preprocessed`: Load preprocessed text instead of processing new texts
* `-n`, `--top-n`: Return top-N results for search queries (default: 5)

If you don't pass `-l`, the program will process texts and save them to their 
respective paths.

## Example Usage

### 1. Download Articles
To download 10 articles based on the query specified in `config.toml`:
```bash
python main.py -d 10
```

### 2. Extract Text
To extract text from the downloaded PDFs:
```bash
python main.py -e
```

### 3. Search
To search for specific terms in the vectorized text:
```bash
python main.py -s "your search query"
```

### 4. Load Preprocessed Data
To use preprocessed text instead of processing new texts, add the `-l` flag:
```bash
python your_script.py -l -s "your search query"
```
