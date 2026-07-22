# pdf-rag-from-scratch
RAG pipeline built from scratch to demonstrate retrieval mechanics and hallucination-refusal testing

# PDF RAG — Grounded Question Answering, Built From Scratch

Ask a question about a PDF and get an answer grounded in that document's actual content, with no hallucination.

Built without a RAG framework — no LangChain, no LlamaIndex. Each stage (chunking, embedding, cosine similarity retrieval, prompt assembly) is implemented by hand, so the retrieval mechanics are visible rather than hidden behind a library.

## Why build it from scratch

This project implements each stage by hand — chunking, embedding, retrieval — to understand it fully, and to test the failure modes directly (see the hallucination test below).

The final step is a manual paste into Claude rather than an API call — kept manual to keep every pipeline stage inspectable.

For everyday PDF Q&A, a tool like Claude Projects already does this faster. This is a learning vehicle that demonstrates the fundamentals, not a daily-driver product.

## Pipeline

```
PDF → Extract text → Clean (fix line-wraps) → Chunk (paragraph-aware)
    → Embed (local, sentence-transformers) → Retrieve (cosine similarity)
    → Assemble prompt → Answer (pasted into Claude)
```

## Evaluation

### Test 1 — grounded retrieval

Question: "Tell me about DBSCAN"

Top match (score 0.336): *"Searching for the centroids is convenient. Though, in real life clusters not always circles... the clusters can be weirdly shaped and even nested..."*

Retrieved chunks were relevant to clustering and unsupervised learning, and the resulting answer stayed grounded in that content.

### Test 2 — refusing to answer when content isn't present

Question: "Where was the French Revolution?" (unrelated to this PDF, which covers machine learning)

Top match score: 0.117, compared to roughly 0.3–0.7 for genuine matches — a clear signal nothing relevant exists.

The prompt instructed: *"Answer ONLY using the sources below. If the answer is not explicitly stated, respond exactly: 'I don't know based on the provided content.'"*

Result: the model declined to answer rather than fabricating a response from unrelated chunks.

Retrieval always returns some result even when nothing relevant exists, so both the similarity scores and the prompt's refusal instruction needed to be checked directly rather than assumed.

## How to run it

1. Install dependencies: `pip install -r requirements.txt`
2. Place your PDF in the project folder and set `pdf_filename` to its name
3. Run Cell 3 to extract and chunk the text
4. Run Cell 4 to embed the chunks (downloads a small local model, ~90MB, first run only)
5. Edit the `question` variable in Cell 5, then run it to retrieve the top matching chunks
6. Run Cell 6 — it prints an assembled prompt
7. Copy that output and paste it into a new Claude chat to get the grounded answer

No API key required. Embedding runs locally via sentence-transformers.

## Key concepts implemented

- **Cosine similarity** — measures the angle between two vectors rather than raw distance, avoiding the "everything is far from everything" problem in high-dimensional space
- **Paragraph-aware chunking** — splits on natural paragraph breaks first, falling back to word-count chunking only when a paragraph is too long, keeping single ideas intact rather than split mid-thought
- **Local embedding** — uses `all-MiniLM-L6-v2` via sentence-transformers, runs entirely on-device with no API key or account needed

## Design choices

- **No vector database** (ChromaDB, Pinecone, etc.) — at this scale, a linear cosine similarity scan is simpler and equally correct. A vector database solves a scaling problem this project doesn't have.
- **Manual final answer step** — kept manual to keep each pipeline stage inspectable while learning.

## Tech stack

Python, pypdf, sentence-transformers, numpy

## Tested on

"Machine Learning for Everyone" (vas3k.com)
