from .search_utils import load_movies
from .search_utils import read_stopwords
from nltk.stem import PorterStemmer

import string

def search_command(query: str, top_results: int = None):
    movies = load_movies()
    result = []
    for movie in movies:
        if token_match( preprocess(query), preprocess(movie["title"])):
            result.append(movie)
        if len(result) == top_results:
            break
    return result

def preprocess(input: str) -> str:
    input = cleanse(input)
    input = tokenize(input)
    input = stem(input)
    input = filter(input)
    return input

def cleanse (input: str) -> str:
    input = input.lower()
    result = input.translate(str.maketrans("", "", string.punctuation))
    return result

def tokenize (input : str) -> list[str]:
    result = cleanse(input)
    tokens = result.split()
    tokens = [token for token in tokens if token]
    return tokens

def stem (input_tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in input_tokens]

def filter(input_token: list[str]) -> list[str]:
    return [token for token in input_token if token not in read_stopwords()]

def token_match (input_tokens: list[str], output_tokens: list[str] ) -> list[str]:
    # output_set = set(output_tokens)
    for token in input_tokens:
        for output_token in  output_tokens:
            if token in output_token:
                return True
    return False
