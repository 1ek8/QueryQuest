#!/usr/bin/env python3

import argparse
import json
from lib.keyword_search import search_command, InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="commands")

    search_parser = subparsers.add_parser("search", help="Search movies ")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build Inverse Index ")
    # search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("tf", help="get term frequency ")
    search_parser.add_argument("doc_id", type = int, help="search document")    
    search_parser.add_argument("term", type = str, help="search term")

    search_parser = subparsers.add_parser("idf", help="get inverse document frequency ")
    search_parser.add_argument("term", type = str, help="search term for idf")    

    search_parser = subparsers.add_parser("tfidf", help="get term frequency ")
    search_parser.add_argument("doc_id", type = int, help="search document")    
    search_parser.add_argument("term", type = str, help="search term")

    search_parser = subparsers.add_parser("bm25idf", help="get inverse document frequency for bm25")
    search_parser.add_argument("term", type = str, help="search term for idf")    

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, movie in enumerate(results):
                print(f"{i}: {movie["title"]}")

        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        
        case "tf":
            index = InvertedIndex()
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            print(f"{tf}")

        case "idf":
            index = InvertedIndex()
            index.load()
            idf = index.get_idf(args.term)
            print(f"{idf:.2f}")

        case "tfidf":
            index = InvertedIndex()
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            idf = index.get_idf(args.term)
            print(f"{idf*tf:.2f}")

        case "bm25idf":
            index = InvertedIndex()
            index.load()
            bm25idf = index.get_bm25_idf(args.term)
            print(f"{bm25idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()