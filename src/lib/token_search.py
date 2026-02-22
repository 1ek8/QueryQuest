from index.inverted_index import InvertedIndex
from models import Movie, SearchResult
from lib.utils import preprocess

def token_search(query: str, top_results: int | None = 5):
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        return []
    
    tokens = preprocess(query)
    if not tokens:
        return []

    seen: set[int] = set()
    results: list[dict] = []
    
    for token in tokens:
        docs = index.get_documents(token)
        for doc in docs:
            if doc in seen:
                continue
            seen.add(doc)
            movie = index.docmap[doc]
            results.append(SearchResult(
                movie['id'],
                movie['title']  
            ))
            if(len(results) >= top_results):
                return results
    
    return results