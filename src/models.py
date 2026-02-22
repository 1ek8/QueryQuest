from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class Movie:
    id: int
    title: str
    description: str

@dataclass
class SearchResult:
    movie_id: int
    title: str
    score: Optional[float] = None

    def __str__(self) -> str:
        if self.score is not None:
            return f"({self.movie_id}) {self.title} - {self.score}"
        return f"({self.movie_id}) {self.title}"
