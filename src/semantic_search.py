#!/usr/bin/env python3

import argparse

from click import command

from embedding.vector import Vector, embed_query, embed_text, verify_embeddings, verify_model
from lib.file_handler import load_movies



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
            vector = Vector()
            movies = load_movies()
            vector.load_or_create_embeddings(movies)
            results = vector.search(args.query, args.limit)
            for result in results:
                print(result)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()