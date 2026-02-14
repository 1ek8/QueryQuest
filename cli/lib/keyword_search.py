from .search_utils import load_movies
from .search_utils import read_stopwords
from .search_utils import CACHE_PATH
from nltk.stem import PorterStemmer
from pickle import dump
import os

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

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = preprocess(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        term = cleanse(term)
        return sorted(list(self.index[term]))
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            movie_input = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, movie_input)

    def save(self):
        CACHE_PATH.mkdir(exist_ok=True)
        with open(CACHE_PATH/'index.pkl', 'wb') as f:
            dump(self.index, f)
        with open(CACHE_PATH/'docmap.pkl', 'wb') as f:
            dump(self.docmap, f)

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
    tokens = input.split()
    tokens = [token for token in tokens if token]
    return tokens

def stem (input_tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in input_tokens]

def filter(input_token: list[str]) -> list[str]:
    return [token for token in input_token if token not in read_stopwords()]

def token_match (input_tokens: list[str], output_tokens: list[str] ) -> list[str]:
    output_set = set(output_tokens)
    for token in input_tokens:
        # for output_token in  output_tokens:
        if token in output_set:
            return True
    return False
