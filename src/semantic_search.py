#!/usr/bin/env python3

import argparse
import re

from click import command

from semantics.chunking import Chunking, semantic_chunk
from semantics.embedding import Embeddings, embed_query, embed_text, verify_embeddings, verify_model
from lib.file_handler import load_movies
from lib.utils import cleanse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="commands")

    verify_parser = subparsers.add_parser("verify", help="verify model")

    embedding_parser = subparsers.add_parser("embed_text", help="embed text")
    embedding_parser.add_argument("text", type = str, help = "text to be embedded")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verift cache movie embeddings")
    
    embedquery_parser = subparsers.add_parser("embedquery", help="embed given query and view the query and its embeddings' info")
    embedquery_parser.add_argument("query", type = str, help = "query to be embedded")

    search_parser = subparsers.add_parser("search", help="Search movies by meaning")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")

    chunk_parser = subparsers.add_parser('chunk', help='Chunk text into fixed-size word groups')
    chunk_parser.add_argument('text', type=str, help='Text to chunk')
    chunk_parser.add_argument('--chunk_size', type=int, default=200, help='Words per chunk (default: 200)')
    chunk_parser.add_argument('--overlap', type=int, default=40, help='no. of words to overlap b/w chunks')

    semantic_chunk_parser = subparsers.add_parser('semantic_chunk', help='Chunk text semantically into fixed-size word groups')
    semantic_chunk_parser.add_argument('text', type=str, help='Text to chunk')
    semantic_chunk_parser.add_argument('--max_chunk_size', type=int, default=4, help='sentences per chunk (default: 4)')
    semantic_chunk_parser.add_argument('--overlap', type=int, default=0, help='no. of words to overlap b/w chunks')

    embed_chunks_parser = subparsers.add_parser('embed-chunks', help='Embed document chunks')

    args = parser.parse_args()

    match args.command:

        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query(args.query)

        case "search":
            vector = Embeddings()
            movies = load_movies()
            vector.load_or_create_embeddings(movies)
            results = vector.search(args.query, args.limit)
            for result in results:
                print(result)

        case "chunk":
            words = cleanse(args.text).split()
            total_words = len(words)
            chunks = []
            for j in range(0, total_words, args.chunk_size):
                chunks.append(' '.join(words[j: j+args.chunk_size+args.overlap]))
            for chunk in chunks:
                print(chunk)

        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            for chunk in chunks:
                print(f"{chunk}\n")

        case 'embed-chunks':
            movies = load_movies()
            vector = Chunking()
            embeddings = vector.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()