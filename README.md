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
The final answer step is intentionally manual: the assembled prompt can be pasted into an LLM such as Claude.
The retrieval system itself requires no API key.
## Tech stack
- Python
- pypdf
- sentence-transformers
- all-MiniLM-L6-v2
- NumPy
## Chunking
Chunk size is measured in **tokens, not words**. `all-MiniLM-L6-v2` has a hard `max_seq_length` of 256 tokens, and word count is a poor proxy for token count — a word-piece tokenizer routinely produces 1.3-2x as many tokens as words, especially for punctuation-heavy or technical text. Chunking by a word limit let chunks reach the embedding model oversized, where `sentence-transformers` truncates silently (no exception, only a low-level log line) rather than failing loudly.
The chunk-content token budget is **derived from the active embedding model at runtime**, not hardcoded: `model.max_seq_length` minus however many special tokens (`[CLS]`/`[SEP]`) its tokenizer adds automatically (`compute_max_content_tokens` in [rag_project.py](rag_project.py)). For `all-MiniLM-L6-v2` that's `256 - 2 = 254` content tokens, with a 50-token overlap between windows of an oversized paragraph.
A paragraph that fits within budget is kept whole, preserving paragraph boundaries. An oversized paragraph is split using the tokenizer's fast offset mapping, slicing the *original* text at exact character boundaries — not `tokenizer.decode()` output, which can reintroduce word-piece artifacts not present in the source.
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
## Design choices
### No RAG framework
LangChain and LlamaIndex are intentionally omitted so the retrieval mechanics remain visible.
### No vector database
The benchmark corpus is small enough that a linear cosine-similarity scan is simpler and sufficient.
A vector database would solve a scaling problem this project does not have.
### Local embeddings
`all-MiniLM-L6-v2` runs locally through sentence-transformers.
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
## Scope
This is a deliberately small retrieval system designed for inspectability and evaluation.
It is not presented as a production-scale RAG platform or a comprehensive retrieval benchmark.
The goal is to make retrieval behavior measurable and failure modes visible.
