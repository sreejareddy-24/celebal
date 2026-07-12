"""
rag_pipeline.py
----------------
A simple, dependency-light Retrieval-Augmented Generation (RAG) pipeline.

Pipeline stages:
    1. Load       -> read a .txt / .pdf document from disk
    2. Chunk      -> split the document into overlapping text chunks
    3. Index      -> vectorize chunks with TF-IDF (no model download required)
    4. Retrieve   -> rank chunks by cosine similarity to the question
    5. Generate   -> feed the top chunks + question to an LLM to produce
                     a grounded answer. If no LLM API key is configured,
                     falls back to an extractive answer built from the
                     best-matching chunk(s), so the pipeline still runs
                     end-to-end offline.

Swap-in points (clearly marked below) let you upgrade this beginner
pipeline to production components later:
    - Retrieval: TF-IDF -> sentence-transformers / OpenAI / Cohere embeddings + FAISS
    - Generation: extractive fallback -> Anthropic Claude / OpenAI GPT / local LLM
"""

import os
import re
import glob
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------- #
# 1. LOAD
# --------------------------------------------------------------------------- #
def load_document(path: str) -> str:
    """Load a .txt or .pdf file and return its raw text."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt" or ext == ".md":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"Unsupported file type: {ext}. Use .txt, .md, or .pdf")


def load_documents(paths: List[str]) -> str:
    """Load and concatenate multiple documents (supports glob patterns)."""
    all_text = []
    resolved = []
    for p in paths:
        resolved.extend(glob.glob(p) if any(c in p for c in "*?[]") else [p])

    for p in resolved:
        all_text.append(load_document(p))
    return "\n\n".join(all_text)


# --------------------------------------------------------------------------- #
# 2. CHUNK
# --------------------------------------------------------------------------- #
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    chunk_size / overlap are measured in words, which keeps this
    dependency-free (no tokenizer download needed) while still producing
    reasonably sized chunks for retrieval.
    """
    # Normalize whitespace first
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")

    if not words or words == [""]:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start = end - overlap  # slide window back by `overlap` words
    return chunks


# --------------------------------------------------------------------------- #
# 3 & 4. INDEX + RETRIEVE
# --------------------------------------------------------------------------- #
@dataclass
class RetrievedChunk:
    text: str
    score: float
    chunk_id: int


class Retriever:
    """
    TF-IDF based retriever.

    Swap-in point: replace TfidfVectorizer + cosine_similarity with
    sentence-transformer embeddings + a FAISS/Chroma index for semantic
    (rather than lexical) retrieval once model downloads are available.
    """

    def __init__(self, chunks: List[str]):
        if not chunks:
            raise ValueError("No chunks to index. Check that the document loaded correctly.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(text=self.chunks[i], score=float(scores[i]), chunk_id=int(i))
            for i in top_idx
            if scores[i] > 0
        ]


# --------------------------------------------------------------------------- #
# 5. GENERATE
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.
If the answer isn't in the context, say you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, retrieved: List[RetrievedChunk]) -> str:
    context = "\n\n---\n\n".join(r.text for r in retrieved)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_with_anthropic(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """Call the Anthropic API for generation. Requires ANTHROPIC_API_KEY env var."""
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def generate_extractive_fallback(question: str, retrieved: List[RetrievedChunk]) -> str:
    """
    No-API-key fallback so the pipeline is runnable end-to-end offline.
    Returns the most relevant chunk(s) as a grounded, if unpolished, answer.
    """
    if not retrieved:
        return "I couldn't find anything relevant to that question in the document."
    best = retrieved[0]
    return (
        f"(Extractive answer — no LLM API key configured)\n"
        f"The most relevant passage I found (similarity={best.score:.2f}):\n\n"
        f"{best.text}"
    )


def generate_answer(question: str, retrieved: List[RetrievedChunk], use_llm: bool = True) -> str:
    prompt = build_prompt(question, retrieved)
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return generate_with_anthropic(prompt)
        except Exception as e:
            return f"[LLM call failed: {e}]\n\n" + generate_extractive_fallback(question, retrieved)
    return generate_extractive_fallback(question, retrieved)


# --------------------------------------------------------------------------- #
# END-TO-END PIPELINE
# --------------------------------------------------------------------------- #
class RAGSystem:
    def __init__(self, doc_paths: List[str], chunk_size: int = 500, overlap: int = 100):
        raw_text = load_documents(doc_paths)
        self.chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
        self.retriever = Retriever(self.chunks)

    def ask(self, question: str, top_k: int = 3, use_llm: bool = True) -> dict:
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        answer = generate_answer(question, retrieved, use_llm=use_llm)
        return {
            "question": question,
            "answer": answer,
            "sources": [{"chunk_id": r.chunk_id, "score": round(r.score, 3), "text": r.text[:200] + "..."} for r in retrieved],
        }
