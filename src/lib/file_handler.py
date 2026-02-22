import json
import pickle
from pathlib import Path
from typing import Any, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / 'data'
CACHE_DIR = PROJECT_ROOT / 'cache'

MOVIES_PATH = DATA_DIR / 'movies.json'
STOPWORDS_PATH = DATA_DIR / 'stopwords.txt'

CACHE_DIR.mkdir(exist_ok=True)

def load_movies() -> list[dict]:
    if not MOVIES_PATH.exists():
        raise FileNotFoundError(f"Movies file not found at {MOVIES_PATH}")
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data['movies']

def load_stopwords() -> list[str]:
    if not STOPWORDS_PATH.exists():
        raise FileNotFoundError(f"STOPWORDS file not found at {STOPWORDS_PATH}")
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()

def save_file(filename: str, data: Any) -> None:
    with open(CACHE_DIR / filename, "wb") as f:
        pickle.dump(data, f)

def load_file(filename: str) -> Any:
    path = CACHE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Cache file {filename} not found.")
    with open(path, "rb") as f:
        return pickle.load(f)
