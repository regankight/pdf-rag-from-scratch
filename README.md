# PDF RAG From Scratch
A small, inspectable Retrieval-Augmented Generation pipeline built in Python without LangChain, LlamaIndex, or a vector database.
The project implements the retrieval path directly:
**PDF → text extraction → paragraph-aware, tokenizer-aware chunking → embeddings → cosine-similarity retrieval → prompt assembly**
It also includes a fixed retrieval-evaluation benchmark for measuring whether relevant evidence is actually being retrieved rather than judging results by inspection alone.
## Why this project exists
RAG systems always return a ranked result.
That does not mean the retrieved result is relevant enough to answer the question.
This project was built to make those mechanics visible and testable.
Rather than hiding retrieval behind a framework, each stage is implemented directly so chunking behavior, embeddings, similarity scores, ranking, and failure cases can be inspected.
## Pipeline
```text
PDF
↓
Text extraction with pypdf
↓
Paragraph-aware, tokenizer-aware chunking
↓
Local embeddings with sentence-transformers
↓
Cosine-similarity retrieval
↓
Top-k ranked chunks
↓
Grounded prompt assembly
↓
Answer generation
```
Running `rag_project.py` directly stops at prompt assembly: the assembled prompt is printed for manual use, e.g. pasted into an LLM such as Claude.
The [API](#serving-the-pipeline-as-an-api) described below automates that last step instead, using a local model via Ollama.
The retrieval system itself requires no API key — and neither does the generation step, for the same reason (see [Local generation](#local-generation-ollama)).
## Tech stack
- Python
- pypdf
- sentence-transformers
- all-MiniLM-L6-v2
- NumPy
- FastAPI + Uvicorn (serving)
- Ollama + Llama 3.2 (local generation)
## Chunking
Chunk size is measured in **tokens, not words**. `all-MiniLM-L6-v2` has a hard `max_seq_length` of 256 tokens, and word count is a poor proxy for token count — a word-piece tokenizer routinely produces 1.3-2x as many tokens as words, especially for punctuation-heavy or technical text. Chunking by a word limit let chunks reach the embedding model oversized, where `sentence-transformers` truncates silently (no exception, only a low-level log line) rather than failing loudly.
The chunk-content token budget is **derived from the active embedding model at runtime**, not hardcoded: `model.max_seq_length` minus however many special tokens (`[CLS]`/`[SEP]`) its tokenizer adds automatically (`compute_max_content_tokens` in [rag_project.py](rag_project.py)). For `all-MiniLM-L6-v2` that's `256 - 2 = 254` content tokens, with a 50-token overlap between windows of an oversized paragraph.
A paragraph that fits within budget is kept whole, preserving paragraph boundaries. An oversized paragraph is split at whole-word boundaries — not raw token offsets, which could (and did, before this was tightened) cut a chunk in the middle of a word — by walking the paragraph word by word and summing each word's real token cost from the tokenizer's fast offset mapping. Each chunk is a slice of the *original* text at exact character boundaries, not `tokenizer.decode()` output, which can reintroduce word-piece artifacts not present in the source. (One narrow, currently-unreachable exception: a single word whose own token count alone exceeds the budget falls back to a token-exact cut inside it, since the hard `max_seq_length` limit has to take priority — see `_split_oversized_paragraph`'s docstring in [rag_project.py](rag_project.py) for why real text can't trigger this with this tokenizer.)
Every chunk is verified against the model's real `max_seq_length` (including special tokens) immediately before embedding (`validate_chunk_fits`); if one would still be too large, the pipeline raises `ValueError` rather than embedding it truncated and silent.
## Retrieval evaluation
The project includes a fixed 17-question benchmark tied to a canonical PDF and chunking configuration.
The benchmark contains:
- 10 single-chunk direct lookups
- 2 multi-chunk questions
- 5 unanswerable questions, consisting of:
  - 1 clearly out-of-domain question
  - 4 in-domain-sounding but unsupported questions
Each answerable case contains expected chunk IDs that act as ground truth.
Evaluation reports both:
### Hit@k
Whether at least one expected relevant chunk appears within the top-k retrieved results.
### Recall@k
The fraction of all expected relevant chunks retrieved within the top-k results.
Metrics are calculated at k = 1, 3, and 5.
## Fixed: silent embedding truncation
Chunking used to be measured in words (`max_chunk_size=500`), not tokens. Since `all-MiniLM-L6-v2` truncates any input over 256 tokens **silently** — no exception, just a log line — chunks over roughly 190-200 words were having their tail cut off before embedding, invisibly.
On this benchmark corpus, **12 of the 35 old chunks (34%)** exceeded the model's limit and were being truncated, by as much as 105 tokens (~30% of that chunk's content) in the worst case. Chunking is now derived from the embedding model's own tokenizer and `max_seq_length` (254 content tokens for `all-MiniLM-L6-v2`, see [Chunking](#chunking)) with an explicit guard that raises rather than truncates. Re-chunking the same PDF under the fixed pipeline produces **47 chunks (up from 35)**.
Re-running the 17-question benchmark against the re-chunked corpus (with `expected_chunk_ids` remapped by inspecting the source evidence, not by trusting the retriever — see `eval/eval_dataset.json`) reproduced the **same Hit@k/Recall@k numbers as the old, truncated baseline**. That's a real, checked result, not an assumption the bug was harmless: none of the 12 truncated chunks that carried this benchmark's specific answers happened to lose their answer-bearing text in the discarded tail. The truncated ~30% of content in the worst-case chunks was real, dropped information — this benchmark's 17 questions simply didn't happen to depend on any of it. A larger or differently-targeted benchmark could well surface a real regression from the old bug that this one couldn't see.
## Baseline retrieval results
| Metric | Result |
| --- | ---: |
| Hit@1 | 0.92 |
| Recall@1 | 0.83 |
| Hit@3 | 0.92 |
| Recall@3 | 0.92 |
| Hit@5 | 0.92 |
| Recall@5 | 0.92 |
These numbers describe this specific benchmark corpus and configuration. They are not intended as general RAG-performance claims. They are unchanged from before the chunking fix (see above) — independently reproduced against the re-chunked, remapped corpus, not carried over.
## Failure analysis
The benchmark exposed two different retrieval limitations.
### 1. Chunk-boundary context loss
The question:
> Which algorithms are mentioned for dimensionality reduction?
had one expected answer-bearing chunk.
That chunk contained the algorithm list:
PCA, SVD, LDA, LSA, and t-SNE.
But PDF extraction introduced a paragraph boundary between the section heading and the algorithm list.
The heading:
```text
Dimensionality Reduction (Generalization)
```
landed in the preceding chunk.
As a result, the answer-bearing chunk contained the algorithm names but no explicit dimensionality-reduction label.
It ranked 15th (previously 11th, before the token-based chunking fix — this chunk's text is byte-for-byte identical in both versions; the rank moved because the pool of competing chunks changed size and content around it, not because this chunk changed) instead of appearing in the top 5.
This was a preprocessing/chunk-context failure rather than an embedding-model failure, and the token-based chunking fix does not address it — it's unrelated to chunk size.
### 2. Compound-query embedding similarity
A deliberately unanswerable question asked:
> How does reinforcement learning from human feedback train a language model?
The source contained reinforcement learning and recurrent neural-network material, but nothing about RLHF.
The top retrieved chunk scored 0.4617 because it discussed training a neural network to generate sequences and correcting its errors.
The embedding therefore matched one strong facet of the compound question while ignoring the missing concepts of reinforcement and human feedback.
This illustrates why raw dense-vector similarity is not equivalent to answerability.
## Tested retrieval change
The chunk-boundary failure suggested a possible improvement:
prepend trailing context from the previous chunk when generating the embedding for the next chunk.
The change fixed the original miss:
```text
Expected chunk rank:
11 → 1
```
It also improved:
```text
Hit@5:
0.92 → 1.00
Recall@5:
0.92 → 1.00
```
But the same change caused neighboring chunks to inherit too much semantic content from their predecessors.
Top-rank performance regressed:
```text
Hit@1:
0.92 → 0.75
Recall@1:
0.83 → 0.67
```
The change was therefore rejected and the original retrieval behavior restored.
This experiment is intentionally not part of the final retrieval implementation.
## Why the rejected change matters
Improving one retrieval metric did not mean the retrieval system became better overall.
The experiment improved top-5 coverage while degrading ranking precision.
The fixed benchmark made that regression visible.
Without evaluation, the change could easily have appeared to be an improvement because it solved the original failure case.
## Project structure
```text
pdf-rag-from-scratch/
├── rag_project.py
├── requirements.txt
├── README.md
├── LICENSE
├── app/                        # FastAPI service — see Serving the pipeline as an API
│   ├── main.py                 # routes + startup lifecycle
│   ├── rag.py                  # retrieval -> prompt -> local-LLM generation adapter
│   └── schemas.py              # request/response models
├── data/
│   └── benchmark.pdf          # local benchmark corpus; ignored by Git
├── eval/
│   ├── list_chunks.py
│   ├── run_eval.py
│   ├── eval_dataset.example.json
│   ├── eval_dataset.json
│   └── README.md
└── tests/
    └── test_chunking.py
```
## Running the RAG pipeline
Run all commands below from the project root — `rag_project.py` points at `data/benchmark.pdf`, a relative path, so running from any other directory will raise a `FileNotFoundError`.
Install dependencies:
```bash
pip install -r requirements.txt
```
Place the PDF at:
```text
data/benchmark.pdf
```
Run:
```bash
python rag_project.py
```
## Inspecting chunks
To inspect the chunk IDs used by the evaluation dataset (each shown with its token count and word count):
```bash
python eval/list_chunks.py
```
## Running the tests
```bash
pytest tests/
```
Covers the tokenizer-aware chunker: paragraphs that fit are kept whole, oversized paragraphs are split into overlapping token-bounded windows, no generated chunk ever exceeds the embedding model's real limit, and the pre-embedding guard rejects one that would.
## Running the retrieval benchmark
Run:
```bash
python eval/run_eval.py
```
The evaluation script compares the active corpus and retrieval configuration against the configuration recorded in the evaluation dataset. If they differ, it prints a warning and continues rather than blocking the run — evaluation still executes, but the mismatch means expected_chunk_ids may no longer be reliable.
## Serving the pipeline as an API
A FastAPI service in [app/](app/) wraps the same retrieval pipeline behind two HTTP endpoints. The index (embedding model + embedded chunks) is built once, at server startup, not per request — see `lifespan` in [app/main.py](app/main.py).
### Running the server
Requires [Ollama](https://ollama.com) installed and running locally, with a model pulled:
```bash
ollama pull llama3.2
```
Then, from the project root:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Interactive API docs (Swagger UI, auto-generated from the schemas in [app/schemas.py](app/schemas.py)) are then available at `http://127.0.0.1:8000/docs`.
### `POST /search`
Retrieval only — wraps `retrieve_top_chunks()` directly, no generation involved. Returns the matched chunks and their cosine-similarity scores:
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "What caused the first AI winter?", "top_n": 3}'
```
```json
{"results": [{"chunk_id": "chunk_0033", "text": "...", "score": 0.3506}, ...]}
```
### `POST /chat`
Retrieves chunks, assembles the same grounded prompt `rag_project.py` builds (`assemble_grounded_prompt`), and streams the answer back from a local model as it's generated — the response body arrives incrementally, not as one blocked-on-completion JSON payload. `curl -N` disables curl's own buffering so the streaming is visible client-side too:
```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What caused the first AI winter?", "top_n": 3}'
```
The grounding behavior carries over from retrieval: a question the corpus doesn't support gets a refusal, not a hallucinated answer. Asking `/chat` "Where was the french revolution" (unrelated to this ML-focused corpus) is correctly answered with a refusal rather than an invented one — the same grounded-prompt contract as the manual pasted-into-Claude workflow, just automated.
Streaming was verified, not just assumed: for a representative question, curl's own `time_starttransfer` (0.002s) was two orders of magnitude below `time_total` (1.06s), and a per-chunk timestamp trace showed each word arriving roughly 20-30ms after the last — consistent with Ollama's real token-generation pace, not a fast response that merely looks incremental.
## Design choices
### No RAG framework
LangChain and LlamaIndex are intentionally omitted so the retrieval mechanics remain visible.
### No vector database
The benchmark corpus is small enough that a linear cosine-similarity scan is simpler and sufficient.
A vector database would solve a scaling problem this project does not have.
### Local embeddings
`all-MiniLM-L6-v2` runs locally through sentence-transformers.
### Local generation (Ollama)
`/chat` generates through a local Llama 3.2 via Ollama rather than a paid hosted API, for the same reason retrieval uses no API key: no cost, no external network call, nothing to configure beyond pulling a model. The tradeoff is answer quality below what a frontier hosted model would give — acceptable here because the point of this project is retrieval mechanics and serving architecture, not generation quality.
The generation call is isolated to a single function, `stream_answer()` in [app/rag.py](app/rag.py) — `app/main.py` only knows it gets a string of text back, not that Ollama is involved. Swapping in a hosted API instead is a change to that one function, not to the routing or schema layers.
### No auth, rate limiting, or persistence
Out of scope by design, not by oversight: this is a single-user local service demonstrating a serving pattern, not a multi-tenant deployment. Adding them would solve problems this project doesn't have.
### Fixed benchmark corpus
The evaluation dataset is tied to a specific PDF, chunking configuration, and embedding model so repeated retrieval changes can be compared against the same ground truth.
## What this project demonstrates
- RAG retrieval mechanics
- PDF text extraction and chunking
- local embedding generation
- cosine-similarity ranking
- retrieval ground-truth construction
- Hit@k and Recall@k evaluation
- answerable vs. unanswerable query testing
- retrieval failure diagnosis
- measured experimentation
- rejecting a proposed change when evaluation showed a regression
- serving a RAG pipeline behind a REST API (FastAPI)
- streaming an LLM response over HTTP, verified rather than assumed
- separating retrieval, generation, and HTTP routing into independently swappable modules
## Scope
This is a deliberately small retrieval system designed for inspectability and evaluation.
It is not presented as a production-scale RAG platform or a comprehensive retrieval benchmark.
The goal is to make retrieval behavior measurable and failure modes visible.
