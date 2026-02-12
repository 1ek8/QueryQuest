from .search_utils import load_movies
from pathlib import Path
from nltk.stem import PorterStemmer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/'data'/'stopwords.txt'

import string

def search_command(query: str, top_results: int = None):
    movies = load_movies()
    result = []
    query = cleanse(query)
    for movie in movies:
        if token_match( tokenize(query), tokenize(movie["title"])):
            result.append(movie)
        if len(result) == top_results:
            break
    return result

def cleanse (input: str) -> str:
    input = input.lower()
    result = input.translate(str.maketrans("", "", string.punctuation))
    return input

def tokenize (input : str) -> str:
    result = cleanse(input)
    tokens = result.split()
    tokens = [token for token in tokens if token]
    return tokens

def token_match (input_tokens: list[str], output_tokens: list[str] ) -> list[str]:
    output_set = set(filter(stem(output_tokens)))
    for token in filter(stem(input_tokens)):
        if token in output_set:
            return True
    return False

def filter (input_tokens: list[str]) -> list[str]:
    with open(DATA_PATH, "r") as f:
        stopwords = f.read().splitlines()

    return [token for token in input_tokens if token not in stopwords]

def stem (input_tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in input_tokens]