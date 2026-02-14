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
            docs = index.get_documents('merida')
            print(docs[0])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()