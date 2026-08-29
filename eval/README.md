# Retrieval evaluation

A small harness for checking retrieval quality across many questions instead
of eyeballing one or two. It calls the exact same `chunk_text` /
`build_index` / `retrieve_top_chunks` functions as the main pipeline — no
retrieval logic is duplicated here, and `chunks`/`chunk_embeddings`/`model`
are passed into the retriever explicitly rather than read from module
globals.

## Chunk IDs

A chunk's ID is a readable string like `chunk_0000`, `chunk_0001`, ...,
derived from its position in the list `load_chunks()` returns (see
`format_chunk_id` in [../rag_project.py](../rag_project.py)).

These IDs are stable **only** for a fixed (PDF, `max_chunk_size`, `overlap`)
combination — not stable across document versions or chunking changes yet.
If you change the PDF or either parameter, re-run `list_chunks.py` and
update `eval_dataset.json`, since the old IDs may now point at different
text. No hashing or versioning is implemented for this first version.

## The `corpus` block

Because `expected_chunk_ids` are only meaningful for the exact config they
were written against, `eval_dataset.json` records that config up front:

```json
{
  "corpus": {
    "pdf_filename": "your_file.pdf",
    "max_chunk_size": 500,
    "overlap": 100,
    "embedding_model": "all-MiniLM-L6-v2"
  },
  "test_cases": [ ... ]
}
```

`run_eval.py` prints this block before running, and compares it against the
config actually active in `rag_project.py` (`pdf_filename`,
`DEFAULT_MAX_CHUNK_SIZE`, `DEFAULT_OVERLAP`, `EMBEDDING_MODEL_NAME`). A
mismatch prints a warning but doesn't stop the run — this is a simple
sanity check, not a versioning system, so treat a warning as "the
expected_chunk_ids are probably stale, go regenerate them."

## Setup

1. Make sure `pdf_filename` in [../rag_project.py](../rag_project.py) points at your PDF.
2. `python eval/list_chunks.py` — prints every chunk with its `chunk_id` and a preview.
3. `cp eval/eval_dataset.example.json eval/eval_dataset.json`
4. Update the `corpus` block to match your actual `pdf_filename`, chunking config, and embedding model.
5. For 10-20 questions against your PDF, fill in each test case:
   - `question`
   - `answerable` — `true` if the PDF contains the answer, `false` for deliberately out-of-domain questions
   - `expected_chunk_ids` — the chunk_id(s) from step 2 that contain the answer (leave `[]` for unanswerable questions)

   Aim for a mix: some questions answered by a single obvious chunk, some
   spanning two, and several unrelated-to-the-document questions to probe
   refusal behavior.
6. `python eval/run_eval.py`

## Metrics: Hit@k vs. Recall@k

For a question with multiple `expected_chunk_ids`, "did retrieval work" can
mean two different things, so both are reported separately rather than
treated as one number:

- **Hit@k** — did *at least one* expected chunk appear in the top k? (Did
  the model get *something* relevant to work with?)
- **Recall@k** — of *all* expected chunks for this question, what fraction
  appeared in the top k? (Did it get *everything* relevant, e.g. all parts
  of a multi-chunk answer?)

For a question with exactly one expected chunk, Hit@k and Recall@k are
identical by construction — the distinction only shows up once a question's
`expected_chunk_ids` has more than one entry.

Both are reported at k = 1, 3, 5. Rank is computed by searching the full
ranking over *all* chunks, not just the top 5, so a near-miss just outside
the top 5 is still visible in the per-question detail.

## What it reports

Per question: expected chunk IDs, the rank each one was found at, the
retrieved top-5 chunk IDs and their scores, Hit@1/3/5, and Recall@1/3/5.

Summary: aggregate Hit@k and Recall@k across all answerable questions, the
list of questions that missed at k=5, and — kept separate — the top-1
similarity scores for answerable vs. unanswerable questions.

**No threshold is hard-coded.** The score distributions are printed raw so
you can judge from your own data whether answerable and unanswerable
questions separate cleanly enough to justify a cutoff, rather than that
cutoff being guessed from a single example. Deciding on a threshold is a
deliberate later step, not something this script does for you.

Full per-question records are also written to `eval/results.json` (gitignored).
