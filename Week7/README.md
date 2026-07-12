# Simple RAG (Retrieval-Augmented Generation) System

A lightweight, dependency-friendly RAG pipeline that answers questions from
your own documents (notes, resumes, research papers, PDFs, etc.) by
retrieving the most relevant passages and using a language model to
generate a grounded answer.

## How it works

```
 document(s)            question
     │                      │
     ▼                      ▼
 ┌─────────┐          ┌───────────┐
 │  Load   │          │  Encode   │
 └────┬────┘          └─────┬─────┘
      ▼                     │
 ┌─────────┐                │
 │  Chunk  │                │
 └────┬────┘                │
      ▼                     │
 ┌─────────────┐            │
 │ TF-IDF Index│◄───────────┘
 └────┬────────┘
      ▼
 ┌───────────────┐
 │ Top-k Retrieve │
 └────┬───────────┘
      ▼
 ┌───────────────────┐
 │ LLM Generation      │──► grounded answer + cited chunks
 │ (Claude API, or      │
 │ extractive fallback) │
 └───────────────────┘
```

1. **Load** — reads `.txt`, `.md`, or `.pdf` files.
2. **Chunk** — splits text into overlapping word-based chunks (keeps
   context across chunk boundaries).
3. **Index** — vectorizes chunks with TF-IDF (`scikit-learn`), so it runs
   with no model downloads and no internet connection required.
4. **Retrieve** — ranks chunks against the question with cosine similarity.
5. **Generate** — passes the question + retrieved chunks to Claude (via the
   Anthropic API) to produce a grounded answer. If no API key is set, the
   system falls back to returning the best-matching passage directly, so the
   whole pipeline is still runnable and testable offline.

## Project structure

```
rag_system/
├── rag_pipeline.py     # Core pipeline: load, chunk, retrieve, generate
├── main.py             # CLI entry point
├── sample_document.txt # Demo document for testing
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

To enable LLM-generated (rather than extractive) answers, set your Anthropic
API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

**Interactive mode:**
```bash
python main.py --docs sample_document.txt
```

**Single question:**
```bash
python main.py --docs sample_document.txt --question "What are the four stages of a RAG pipeline?"
```

**Multiple documents / PDFs:**
```bash
python main.py --docs notes.pdf resume.pdf research_paper.pdf
```

**Force extractive mode (no API calls):**
```bash
python main.py --docs sample_document.txt --question "What is RAG?" --no_llm
```

## Using it as a library

```python
from rag_pipeline import RAGSystem

rag = RAGSystem(doc_paths=["sample_document.txt"])
result = rag.ask("Why does RAG reduce hallucination?")

print(result["answer"])
for src in result["sources"]:
    print(src["chunk_id"], src["score"], src["text"])
```

## Design choices & upgrade paths

This project intentionally starts simple so every stage is easy to inspect
and explain:

| Component  | Current (beginner)      | Upgrade path                                   |
|------------|--------------------------|-------------------------------------------------|
| Chunking   | Fixed-size word windows  | Sentence/semantic-aware splitting               |
| Retrieval  | TF-IDF + cosine similarity | Dense embeddings (sentence-transformers) + FAISS/Chroma |
| Generation | Claude API / extractive fallback | Any hosted or local LLM                |

Swap-in points are marked with comments directly in `rag_pipeline.py`.

## Example dataset options

- Your own PDFs: notes, resume, research papers, books
- Any public Hugging Face QA dataset, for benchmarking retrieval/generation quality at scale
