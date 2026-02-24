#!/usr/bin/env python3

import argparse

from click import command

from embedding.vector import embed_text, verify_embeddings, verify_model



def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="commands")

    verify_parser = subparsers.add_parser("verify", help="verify model")

    embedding_parser = subparsers.add_parser("embed_text", help="embed text")
    embedding_parser.add_argument("text", type = str, help = "text to be embedded")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verift cache movie embeddings")
    
    args = parser.parse_args()

    match args.command:

        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()