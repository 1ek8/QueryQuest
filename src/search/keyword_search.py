from collections import Counter
import math

from utils import BM25_B, BM25_K1, cleanse
from file_handler import load_file, load_stopwords, load_movies, save_file
from utils import preprocess

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: Counter[int, int] = Counter()
        self.stopwords = load_stopwords()

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            movie_input = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, movie_input)

    def load(self):
        self.index = load_file(f'index.pkl')
        self.docmap = load_file('docmap.pkl')
        self.term_frequencies = load_file('term_frequencies.pkl')
        self.doc_lengths = load_file('doc_lengths.pkl')

    def save(self):
        save_file('index.pkl', self.index)
        save_file('docmap.pkl', self.docmap)
        save_file('term_frequencies.pkl', self.term_frequencies)
        save_file('doc_lengths.pkl', self.doc_lengths)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values())/len(self.doc_lengths)
    
    def __add_document(self, doc_id: int, text: str):
        tokens = preprocess(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
            self.doc_lengths[doc_id] += 1

    def get_documents(self, term: str) -> list[int]:
        term = cleanse(term)
        return sorted(self.index.get(term, set()))
    
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

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        b = BM25_B
        avg = self.__get_avg_doc_length()
        doc_len = self.doc_lengths.get(doc_id, 0)

        if avg == 0:
            return 0.0
        
        l_norm = 1 - b + b * (doc_len / avg)

        tf = self.get_tf(doc_id, term)
        if tf == 0:
            return 0.0
        
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * l_norm)
        return bm25_tf
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = preprocess(term)
        if(len(tokens) != 1):
            return 0
        token = tokens[0]

        N = len(self.docmap)
        df = len(self.index.get(token, set()))

        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25(self, doc_id, term):
        bm25 = self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
        return bm25
    
    def bm25_search(self, query, limit:int = 5):
        tokens = preprocess(query)
        if not tokens:
            return []
        scores: dict[int, float] = {}
        for doc_id in self.docmap.keys():
            score = 0.0
            for token in tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
        ranked = sorted(scores.items(), key = lambda x: (-x[1], x[0]))
        return ranked[:limit]