import os
import pickle


def load_preprocessed_file(path):
    with open(path, "rb") as f:
        file = pickle.load(f)

    return file


def create_preprocessed_file(object, path):
    with open(path, "wb") as f:
        pickle.dump(object, f)


def file_name_to_title(file_name: str):
    file_name = file_name.split(".")[0]

    file_name = file_name.replace("\n__", " ").replace("_", " ")

    return file_name


def load_text_files(text_dir, clean_text_func):
    text_data = []
    filenames = []

    for filename in os.listdir(text_dir):
        file_path = os.path.join(text_dir, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                cleaned_content = clean_text_func(content)
                text_data.append(cleaned_content)
                filenames.append(filename)

    return text_data, filenames