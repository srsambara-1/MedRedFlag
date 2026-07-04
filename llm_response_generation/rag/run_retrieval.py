#!/usr/bin/env python3
"""RAG: Retrieves relevant medical documents for patient questions using MedRAG."""

import argparse
import json
import pandas as pd
from typing import List, Dict, Any
from src.utils import RetrievalSystem
from datetime import datetime


def load_data(csv_path: str, n_rows: int = None) -> pd.DataFrame:
    """
    Load data from CSV.

    Args:
        csv_path: Path to CSV file
        n_rows: Number of rows to load (None for all)

    Returns:
        DataFrame with rows
    """
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if n_rows:
        df = df.head(n_rows)

    print(f"Loaded {len(df)} rows")

    if "redirection_id" not in df.columns:
        raise ValueError("ERROR: 'redirection_id' column not found in CSV")

    return df


def run_retrieval(
    retriever_name: str,
    corpus_name: str,
    questions: List[str],
    redirection_ids: List[str],
    k: int = 32,
    rrf_k: int = 100,
    db_dir: str = "./corpus"
) -> List[Dict[str, Any]]:
    """
    Run retrieval using patient questions.

    Args:
        retriever_name: Name of retriever (MedCPT, RRF-2, etc.)
        corpus_name: Name of corpus (MedCorp, Textbooks, etc.)
        questions: List of questions
        redirection_ids: List of unique redirection IDs
        k: Number of documents to retrieve
        rrf_k: RRF parameter
        db_dir: Corpus directory

    Returns:
        List of results with documents and scores
    """
    print(f"\n{'='*80}")
    print(f"Running Retrieval")
    print(f"Retriever: {retriever_name} | Corpus: {corpus_name}")
    print(f"Number of queries: {len(questions)} | Top-k: {k}")
    print(f"{'='*80}\n")

    # Initialize retrieval system
    print(f"Initializing {retriever_name} retriever...")
    retrieval_system = RetrievalSystem(
        retriever_name=retriever_name,
        corpus_name=corpus_name,
        db_dir=db_dir,
        cache=False
    )
    print("Retriever initialized.\n")

    # Run retrieval
    results = []
    for idx, (question, redirection_id) in enumerate(zip(questions, redirection_ids)):
        print(f"[{idx+1}/{len(questions)}] Processing (ID: {redirection_id}): {question[:80]}...")

        docs, scores = retrieval_system.retrieve(
            question=question,
            k=k,
            rrf_k=rrf_k,
            id_only=False
        )

        result = {
            "redirection_id": redirection_id,
            "question": question,
            "retriever": retriever_name,
            "corpus": corpus_name,
            "k": k,
            "documents": docs,
            "scores": scores,
        }
        results.append(result)

        # Print top-3
        print(f"  Top-3 documents:")
        for i in range(min(3, len(docs))):
            print(f"    [{i+1}] Score: {scores[i]:.4f} | {docs[i]['title'][:60]}...")
        print()

    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file."""
    serializable_results = []
    for result in results:
        serializable_result = {
            "redirection_id": result["redirection_id"],
            "question": result["question"],
            "retriever": result["retriever"],
            "corpus": result["corpus"],
            "k": result["k"],
            "documents": [
                {
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "id": doc.get("id", "")
                } for doc in result["documents"]
            ],
            "scores": [float(score) for score in result["scores"]]
        }
        serializable_results.append(serializable_result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Medical Document Retrieval for RAG")

    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file path")
    parser.add_argument("--retriever", type=str, required=True,
                        choices=["BM25", "Contriever", "SPECTER", "MedCPT", "RRF-2", "RRF-4"],
                        help="Retriever to use")
    parser.add_argument("--corpus", type=str, required=True,
                        choices=["PubMed", "Textbooks", "StatPearls", "Wikipedia", "MedText", "MedCorp"],
                        help="Corpus to use")

    # Optional arguments
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: auto-generated)")
    parser.add_argument("--n_rows", type=int, default=None,
                        help="Number of rows to process (default: all)")
    parser.add_argument("--k", type=int, default=32,
                        help="Number of documents to retrieve (default: 32)")
    parser.add_argument("--rrf_k", type=int, default=100,
                        help="RRF parameter for ensemble retrievers (default: 100)")
    parser.add_argument("--db_dir", type=str, default="./corpus",
                        help="Corpus directory (default: ./corpus)")

    args = parser.parse_args()

    # Load data
    df = load_data(args.input, n_rows=args.n_rows)
    questions = df["patient_question"].tolist()
    redirection_ids = df["redirection_id"].tolist()

    print(f"\nConfiguration:")
    print(f"  Retriever: {args.retriever}")
    print(f"  Corpus: {args.corpus}")
    print(f"  Number of questions: {len(questions)}")
    print(f"  Top-k documents: {args.k}")

    # Run retrieval
    results = run_retrieval(
        retriever_name=args.retriever,
        corpus_name=args.corpus,
        questions=questions,
        redirection_ids=redirection_ids,
        k=args.k,
        rrf_k=args.rrf_k,
        db_dir=args.db_dir
    )

    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"retrieval_{args.retriever.lower()}_{args.corpus.lower()}_{timestamp}.json"

    save_results(results, output_file)

    print(f"\n{'='*80}")
    print("Retrieval completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
