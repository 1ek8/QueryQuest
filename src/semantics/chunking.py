import json
import re

import numpy as np
from lib.file_handler import CACHE_DIR
from semantics.embedding import Embeddings

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

def semantic_chunk(text: str, max_chunk_size: int, overlap: int):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    total_sentences = len(sentences)
    chunks = []
    step = max(1, max_chunk_size - overlap)
    for j in range(0, total_sentences, step):
        chunks.append(' '.join(sentences[j: j+max_chunk_size]))
    return chunks
        