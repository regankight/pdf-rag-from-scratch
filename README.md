# PDF RAG From Scratch
A small, inspectable Retrieval-Augmented Generation pipeline built in Python without LangChain, LlamaIndex, or a vector database.
The project implements the retrieval path directly:
**PDF → text extraction → paragraph-aware chunking → embeddings → cosine-similarity retrieval → prompt assembly**
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
Paragraph-aware chunking
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
## Baseline retrieval results
| Metric | Result |
| --- | ---: |
| Hit@1 | 0.92 |
| Recall@1 | 0.83 |
| Hit@3 | 0.92 |
| Recall@3 | 0.92 |
| Hit@5 | 0.92 |
| Recall@5 | 0.92 |
These numbers describe this specific benchmark corpus and configuration. They are not intended as general RAG-performance claims.
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
It ranked 11th instead of appearing in the top 5.
This was a preprocessing/chunk-context failure rather than an embedding-model failure.
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
└── eval/
    ├── list_chunks.py
    ├── run_eval.py
    ├── eval_dataset.example.json
    ├── eval_dataset.json
    └── README.md
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
To inspect the chunk IDs used by the evaluation dataset:
```bash
python eval/list_chunks.py
```
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
