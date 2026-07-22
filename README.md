# pdf-rag-from-scratch
RAG pipeline built from scratch to demonstrate retrieval mechanics and hallucination-refusal testing

PDF RAG — Grounded Question Answering, Built From Scratch
Ask a question about a PDF and get an answer grounded only in that document's actual content — no hallucination, no guessing.
Built deliberately without a RAG framework — no LangChain, no LlamaIndex. Every stage (chunking, embedding, cosine similarity retrieval, prompt assembly) is built by hand so the retrieval mechanics are visible and understood, not hidden behind a library. This is the point of the project: to demonstrate what's actually happening under the frameworks, including where and why RAG fails.
Why build it from scratch
This project implements each stage by hand — chunking, embedding, retrieval — to understand it fully, and to test the failure modes directly (see the hallucination test below).
The final step is a manual paste into Claude rather than an API call — kept manual to keep every pipeline stage inspectable.
For everyday PDF Q&A, a tool like Claude Projects already does this faster. This is a learning vehicle that demonstrates the fundamentals, not a daily-driver product.
Pipeline
PDF → Extract text → Clean (fix line-wraps) → Chunk (paragraph-aware)
    → Embed (local, sentence-transformers) → Retrieve (cosine similarity)
    → Assemble prompt → Answer (pasted into Claude)

Evaluation
Test 1 — grounded retrieval working correctly
Question: "Tell me about DBSCAN" Top match (score 0.336): "Searching for the centroids is convenient. Though, in real life clusters not always circles... the clusters can be weirdly shaped and even nested..." Retrieved chunks were genuinely relevant to clustering and unsupervised learning, and the answer built from them stayed grounded in that content.
Test 2 — correctly refusing to answer (hallucination test)
Question: "Where was the French Revolution?" (unrelated to this PDF, which is about machine learning) Top match score: 0.117 — versus ~0.3–0.7 for genuine matches, a clear signal that nothing relevant exists. The prompt instructed: "Answer ONLY using the sources below. If the answer is not explicitly stated, respond exactly: 'I don't know based on the provided content.'" Result: the model correctly declined to answer rather than fabricating from unrelated chunks.
This was a deliberate test, not an assumption. Retrieval always returns something even when nothing is relevant — so the similarity scores and the prompt's refusal instruction both had to be verified, not just the happy path. Testing the failure mode is the point.
How to run it
1. Install dependencies: pip install -r requirements.txt
2. Place your PDF in the project folder and set pdf_filename to its name
3. Run Cell 3 to extract and chunk the text
4. Run Cell 4 to embed the chunks (downloads a small local model, ~90MB, first run only)
5. Edit the question variable in Cell 5, run it to retrieve the top matching chunks
6. Run Cell 6 — it prints an assembled prompt
7. Copy that output and paste it into a new Claude chat to get the grounded answer

No API key required. Embedding runs locally via sentence-transformers.
Key concepts implemented
Cosine similarity — measures the angle between two vectors rather than raw distance, which avoids the "everything is far from everything" problem in high-dimensional space
Paragraph-aware chunking — splits on natural paragraph breaks first, falling back to word-count chunking only when a paragraph is too long, keeping single ideas intact rather than split mid-thought
Local embedding — uses all-MiniLM-L6-v2 via sentence-transformers, runs entirely on-device, no API key or account needed
Deliberate design choices (not gaps)
No vector database (ChromaDB, Pinecone, etc.) — at this scale, a linear cosine similarity scan is simpler, fully transparent, and equally correct. A vector DB solves a scaling problem this project doesn't have.
Manual final step — see above; kept manual on purpose to keep every stage inspectable.

Tech stack
Python, pypdf, sentence-transformers, numpy
Tested on
"Machine Learning for Everyone" (vas3k.com)
