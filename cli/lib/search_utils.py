import json
from pathlib import Path

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOVIES_PATH = PROJECT_ROOT/'data'/'movies.json'
STOPWORDS_PATH = PROJECT_ROOT/'data'/'stopwords.txt'
CACHE_PATH = PROJECT_ROOT/'cache'

def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data['movies']

def read_stopwords () -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stopwords = f.read().splitlines()

    return stopwords