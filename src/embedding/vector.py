from pydoc import doc
from sentence_transformers import SentenceTransformer
from torch import embedding
from lib.utils import  vector_magnitude

import numpy as np

from lib.file_handler import CACHE_DIR, load_movies
from models import SearchResult

EMBEDDINGS_PATH = CACHE_DIR / 'movie_embeddings.npy'

class Vector:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = load_movies()
        self.document_map = {}

    def add_vectors(self, vec1, vec2):
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")
        
        return [vec1[i] + vec2[i] for i in range(len(vec1))]

    def subtract_vectors(self, vec1, vec2):

        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")
        
        return [vec1[i] - vec2[i] for i in range(len(vec1))]
    
    def dot_product(self, vec1, vec2):

        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")

        sum: float = 0
        
        for i in range(len(vec1)):
            sum += vec1[i] * vec2[i] 
        
        return sum

    def cosine_similar(self, vec1, vec2):

        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")

        magnitude_vec1 = vector_magnitude(vec1)
        magnitude_vec2 = vector_magnitude(vec2)

        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            return 0.0
        
        dot_product = self.dot_product(vec1, vec2)
        
        return dot_product/(magnitude_vec1*magnitude_vec2)

    def generate_embedding(self, text):
        if not text or text.strip() == "":
            raise ValueError("Input text cannot be empty or contain only whitespace.")
        embedding = self.model.encode(text)
        return embedding

    def build_embeddings(self, documents):
        self.documents = documents
        docs_info = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            doc_rep = f"{doc['title']}: {doc['description']}"
            docs_info.append(doc_rep)
        self.embeddings = self.model.encode(docs_info, show_progress_bar=True)
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(str(EMBEDDINGS_PATH), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc

        if EMBEDDINGS_PATH.exists():
            self.embeddings = np.load(str(EMBEDDINGS_PATH))
            if len(self.embeddings) == len(documents):
                return self.embeddings
            
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity = self.cosine_similar(query_embedding, embedding)
            doc = self.documents[i]
            similarities.append((similarity, doc))

        similarities.sort(key = lambda x: x[0], reverse=True)
        
        results = []
        for score, doc in similarities[:limit]:
            results.append(SearchResult(doc['id'], doc['title'], score))

        return results

def verify_model():
    vector = Vector()
    
    print(f"Model loaded: {vector.model}")
    print(f"Max sequence length: {vector.model.max_seq_length}")

def embed_text(text):
    vector = Vector()
    embedding = vector.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    vector = Vector()
    documents = load_movies()
    embeddings = vector.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query(query):
    vector = Vector()
    embedding = vector.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")