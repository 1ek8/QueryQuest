#!/usr/bin/env python3

import argparse
import json
from lib.search_utils import BM25_B, BM25_K1
from lib.keyword_search import search_command, InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="commands")

    search_parser = subparsers.add_parser("search", help="Search movies ")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Inverse Index ")
    # search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="get term frequency ")
    tf_parser.add_argument("doc_id", type = int, help="search document")    
    tf_parser.add_argument("term", type = str, help="search term")

    idf_parser = subparsers.add_parser("idf", help="get inverse document frequency ")
    idf_parser.add_argument("term", type = str, help="search term for idf")    

    tfidf_parser = subparsers.add_parser("tfidf", help="get term frequency ")
    tfidf_parser.add_argument("doc_id", type = int, help="search document")    
    tfidf_parser.add_argument("term", type = str, help="search term")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="get inverse document frequency for bm25")
    bm25idf_parser.add_argument("term", type = str, help="search term for idf")   

    bm25tf_parser = subparsers.add_parser("bm25tf", help="get bm25 saturated term frequency ")
    bm25tf_parser.add_argument("doc_id", type = int, help="search document")    
    bm25tf_parser.add_argument("term", type = str, help="search term") 
    bm25tf_parser.add_argument("k1", type = float, help="k1 value", nargs='?', default=BM25_K1) 
    bm25tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25_parser = subparsers.add_parser("bm25", help="get bm25 score ")
    bm25_parser.add_argument("doc_id", type = int, help="search document")    
    bm25_parser.add_argument("term", type = str, help="search term") 
    bm25_parser.add_argument("k1", type = float, help="k1 value", nargs='?', default=BM25_K1) 
    bm25_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, help="Search limit", nargs='?', default=5)

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

        case "bm25tf":
            index = InvertedIndex()
            index.load()
            bm25_tf = index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"{bm25_tf:.2f}")

        case "bm25search":
            index = InvertedIndex()
            index.load()
            result = index.bm25_search(args.query, args.limit)
            for i, (doc_id, score) in enumerate(result, start=1):
                movie = index.docmap[doc_id]
                title = movie["title"]
                print(f"{i}. ({doc_id}) {title} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()