import json
import re

import numpy as np
from lib.file_handler import CACHE_DIR
from semantics.embedding import Embeddings
from models import ChunkSearchResult
from lib.utils import split_semantic_sentences

CHUNK_EMBEDDING_PATH = CACHE_DIR / 'chunk_embeddings.npy'
CHUNK_METADATA_PATH = CACHE_DIR / 'chunk_metadata.json'

class Chunking(Embeddings):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc['id']] = doc
    
        all_chunks = []
        chunk_metadata = []

        for idx, doc in enumerate(documents):
            description = doc.get('description', '')
            if not description.strip():
                continue
            chunks = semantic_chunk(description, 4, 1)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'movie_idx': idx,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks)
                })
        
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        CACHE_DIR.mkdir(exist_ok=True)
        np.save(CHUNK_EMBEDDING_PATH, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, 'w') as f:
            json.dump({
                'chunks': chunk_metadata,
                'total_chunks': len(all_chunks)
            }, f,indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc['id']] = doc

        if CHUNK_EMBEDDING_PATH.exists() and CHUNK_METADATA_PATH.exists():
            self.chunk_embeddings = np.load(CHUNK_EMBEDDING_PATH)
            with open(CHUNK_METADATA_PATH, 'r') as f:
                data = json.load(f)
                self.chunk_metadata = data['chunks']
            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        embedder = Embeddings()
        query_embedding = embedder.generate_embedding(query)
        chunk_scores = []

        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = embedder.cosine_similar(chunk_embedding, query_embedding)
            metadata = self.chunk_metadata[idx]
            
            chunk_scores.append({
                "chunk_idx": metadata["chunk_idx"],
                "movie_idx": metadata["movie_idx"],
                "score": similarity
            })

        movie_scores = {}

        for chunk_score in chunk_scores:
            if chunk_score["movie_idx"] not in movie_scores or chunk_score["score"] > movie_scores[chunk_score["movie_idx"]]["score"]:
                movie_scores[chunk_score["movie_idx"]] = chunk_score

        movie_scores = sorted(movie_scores.values(), key = lambda x: x["score"], reverse=True)
        movie_scores = movie_scores[:limit]

        final_result = []
        for item in movie_scores:
            movie = self.documents[item["movie_idx"]]

            id = item["movie_idx"]
            title = movie.get("title")
            document = movie.get("description", "")
            score = item["score"]

            result = ChunkSearchResult(
                id,
                title,
                document = document[:100],
                score = round(score, 4),
                metadata = {
                    "chunk_idx": item["chunk_idx"],
                    "movie_idx": id
                }
            )

            final_result.append(result)

        return final_result

def semantic_chunk(text: str, max_chunk_size: int, overlap: int):
    sentences = split_semantic_sentences(text)

    if not sentences:
        return []
    
    total_sentences = len(sentences)
    chunks = []
    step = max(1, max_chunk_size - overlap)
    
    for j in range(0, total_sentences, step):
        chunks.append(' '.join(sentences[j: j+max_chunk_size]))
    return chunks
    