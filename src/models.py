from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Movie:
    id: int
    title: str
    description: str

@dataclass
class SearchResult:
    movie_id: int
    title: str
    score: float
