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

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, movie in enumerate(results):
                print(f"{i}: {movie["title"]}")
            pass
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        
        case "tf":
            index = InvertedIndex()
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            print(f"{tf}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()