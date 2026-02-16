from collections import Counter
from .search_utils import BM25_K1, load_movies
from .search_utils import read_stopwords
from .search_utils import CACHE_PATH
from nltk.stem import PorterStemmer
from pickle import dump, load
import math
import os

import string

def search_command(query: str, top_results: int | None = 5):
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        return []
    
    tokens = preprocess(query)

    seen: set[int] = set()
    results: list[dict] = []
    
    for token in tokens:
        docs = index.get_documents(token)
        for doc in docs:
            if doc in seen:
                continue
            seen.add(doc)
            results.append(index.docmap[doc])
            if(len(results) >= top_results):
                return results
    
    return results

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1)
        return bm25_tf

    def __add_document(self, doc_id: int, text: str):
        tokens = preprocess(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        term = cleanse(term)
        return sorted(self.index.get(term, set()))
    
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
        with open(CACHE_PATH/'term_frequencies.pkl', 'wb') as f:
            dump(self.term_frequencies, f)

    def load(self):
        INDEX_PATH = CACHE_PATH/'index.pkl'
        DOCMAP_PATH = CACHE_PATH/'docmap.pkl'
        TF_PATH = CACHE_PATH/'term_frequencies.pkl'

        if not INDEX_PATH.exists() or not DOCMAP_PATH.exists() or not TF_PATH.exists():
            raise FileNotFoundError("Index, docmap or TF file not found in cache directory")
        
        with open(INDEX_PATH, "rb") as f:
            self.index = load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = load(f)
        with open(TF_PATH, "rb") as f:
            self.term_frequencies = load(f)

    def get_tf(self, doc_id, term):
        tokens = preprocess(term)
        if len(tokens) == 0:
            return 0
        if len(tokens) > 1:
            raise ValueError("Expected a single token term for TF calculation")
        term = tokens[0]
        count_dict = self.term_frequencies.get(doc_id)
        if not count_dict:
            return 0
        return int(count_dict.get(term, 0))

    def get_idf(self, term: str):
        total_doc_count = len(self.docmap)
        tokens = preprocess(term)

        if(len(tokens) != 1):
            return 0
        
        token = tokens[0]
        term_match_doc_count = len(self.index.get(token, set()))
        
        return math.log( (total_doc_count + 1)/(term_match_doc_count + 1) )
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = preprocess(term)
        if(len(tokens) != 1):
            return 0
        token = tokens[0]

        N = len(self.docmap)
        df = len(self.index.get(token, set()))

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

def preprocess(input: str) -> list[str]:
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
