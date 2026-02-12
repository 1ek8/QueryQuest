from .search_utils import load_movies

def search_command(query: str, top_results: int = None):
    movies = load_movies()
    result = []
    for movie in movies:
        if query.lower() in movie["title"].lower():
            result.append(movie)
        if len(result) == top_results:
            break
    return result