from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Movie:
    id: int
    title: str
    description: str
    year: int
    raw_data: Dict[str, Any]  # To store extra fields if needed

@dataclass
class SearchResult:
    movie_id: int
    title: str
    score: float
