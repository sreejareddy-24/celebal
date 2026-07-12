"""
main.py
-------
CLI for the RAG system.

Usage:
    # Interactive Q&A loop
    python main.py --docs sample_document.txt

    # Single question
    python main.py --docs sample_document.txt --question "What is RAG?"

    # Multiple documents / PDFs
    python main.py --docs notes.pdf resume.pdf research_paper.pdf
"""

import argparse
from rag_pipeline import RAGSystem


def main():
    parser = argparse.ArgumentParser(description="Simple RAG Question-Answering System")
    parser.add_argument("--docs", nargs="+", required=True, help="Path(s) to .txt/.md/.pdf document(s)")
    parser.add_argument("--question", type=str, default=None, help="Ask a single question and exit")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--no_llm", action="store_true", help="Force extractive (no-API) mode")
    args = parser.parse_args()

    print("Loading and indexing documents...")
    rag = RAGSystem(doc_paths=args.docs)
    print(f"Indexed {len(rag.chunks)} chunks from {len(args.docs)} document(s).\n")

    def ask_and_print(question: str):
        result = rag.ask(question, top_k=args.top_k, use_llm=not args.no_llm)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}\n")
        print("Top sources:")
        for s in result["sources"]:
            print(f"  [chunk {s['chunk_id']} | score {s['score']}] {s['text']}")
        print("-" * 70)

    if args.question:
        ask_and_print(args.question)
        return

    print("Enter your questions (type 'exit' to quit):")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        ask_and_print(q)


if __name__ == "__main__":
    main()
