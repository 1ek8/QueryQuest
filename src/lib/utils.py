import math
import string
from nltk.stem import PorterStemmer
import numpy as np
from lib.file_handler import load_stopwords

stemmer = PorterStemmer()

BM25_K1 = 1.5
BM25_B = 0.75

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
    return [token for token in input_token if token not in load_stopwords()]

def token_match (input_tokens: list[str], output_tokens: list[str] ) -> list[str]:
    output_set = set(output_tokens)
    for token in input_tokens:
        if token in output_set:
            return True
    return False

def vector_magnitude (vector : list[float]):
    sum: float = 0
    for element in vector:
        sum += element**2

    return math.sqrt(sum)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
