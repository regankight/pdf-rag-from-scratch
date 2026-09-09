# ============================================================
# Retrieval evaluation harness.
#
# Runs every question in eval_dataset.json through the existing retriever
# (unchanged cosine-similarity ranking from rag_project.py), records what
# came back, and reports Hit@k, Recall@k, and raw similarity-score
# distributions. No threshold is chosen here — see the "SCORE
# DISTRIBUTIONS" section of the printed report and judge for yourself
# whether answerable/unanswerable scores separate cleanly.
#
# Hit@k   — did AT LEAST ONE expected chunk appear in the top k?
# Recall@k — of ALL expected chunks for this question, what fraction
#            appeared in the top k? (For single-expected-chunk questions
#            these are identical — the distinction only matters once a
#            question has multiple expected_chunk_ids.)
#
# The dataset's "corpus" block records which PDF and chunking config the
# expected_chunk_ids were written against, since those IDs are only valid
# for that exact combination. This script prints that recorded config and
# warns (without stopping) if it no longer matches the active config in
# rag_project.py.
#
# Setup:
#   1. cp eval/eval_dataset.example.json eval/eval_dataset.json
#   2. Run eval/list_chunks.py, read the previews, fill in real questions
#      and expected_chunk_ids for your PDF. Update the "corpus" block to
#      match your actual pdf_filename / chunking config / embedding model.
#   3. Run: python eval/run_eval.py
# ============================================================

import sys
import os
import json
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag_project import (
    pdf_filename,
    build_index,
    retrieve_top_chunks,
    compute_max_content_tokens,
    DEFAULT_OVERLAP_TOKENS,
    EMBEDDING_MODEL_NAME,
)

EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
K_VALUES = (1, 3, 5)


def load_eval_dataset(path):
    """Returns (corpus_metadata, test_cases). The dataset file is expected
    to be {"corpus": {...}, "test_cases": [...]} — see eval_dataset.example.json."""
    if not os.path.exists(path):
        example_path = os.path.join(os.path.dirname(__file__), "eval_dataset.example.json")
        raise FileNotFoundError(
            f"No eval dataset found at {path}.\n"
            f"Copy {example_path} to {path} and fill in real questions "
            f"for your PDF (use list_chunks.py to find chunk_ids)."
        )
    with open(path) as f:
        dataset = json.load(f)
    return dataset["corpus"], dataset["test_cases"]


def check_corpus_config(recorded_corpus, model):
    """The expected_chunk_ids in the dataset are only meaningful for the
    exact (PDF, max_content_tokens, overlap_tokens, embedding model) it was
    built against — warn (don't block) if the active config has drifted.

    max_content_tokens is derived from the embedding model (see
    compute_max_content_tokens), so the model must already be loaded —
    that's why this now runs after build_index() instead of before it."""
    active_corpus = {
        "pdf_filename": pdf_filename,
        "max_content_tokens": compute_max_content_tokens(model),
        "overlap_tokens": DEFAULT_OVERLAP_TOKENS,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }

    print("Dataset was built against corpus/config:")
    for field, value in recorded_corpus.items():
        print(f"    {field}: {value}")

    mismatches = [
        field for field, value in active_corpus.items()
        if recorded_corpus.get(field) != value
    ]
    if mismatches:
        print("\nWARNING: active config differs from the dataset's recorded config "
              "in: " + ", ".join(mismatches))
        print("Chunk IDs (and therefore expected_chunk_ids) may no longer point at "
              "the same text. Re-run eval/list_chunks.py and update eval_dataset.json "
              "if this is not intentional.\n")
    else:
        print("Active config matches the dataset's recorded config.\n")


def evaluate_case(case, chunks, chunk_embeddings, model):
    """Retrieval only depends on (chunks, chunk_embeddings, model), all
    passed in explicitly by the caller — no hidden global state here."""
    question = case["question"]
    answerable = case["answerable"]
    expected_chunk_ids = case.get("expected_chunk_ids", [])

    # Rank every chunk so we can find the true rank of each expected chunk
    # even if it falls outside the top 5.
    full_ranking = retrieve_top_chunks(question, chunks, chunk_embeddings, model, top_n=len(chunks))
    ranked_ids = [chunk_id for chunk_id, _, _ in full_ranking]

    top5 = full_ranking[:5]
    top5_ids = [chunk_id for chunk_id, _, _ in top5]
    top5_scores = [round(float(score), 4) for _, _, score in top5]
    top_score = top5_scores[0] if top5_scores else None

    record = {
        "question": question,
        "answerable": answerable,
        "expected_chunk_ids": expected_chunk_ids,
        "retrieved_chunk_ids_top5": top5_ids,
        "top5_scores": top5_scores,
        "top_score": top_score,
    }

    if answerable:
        ranks_by_id = {}
        for eid in expected_chunk_ids:
            if eid not in ranked_ids:
                print(f"  WARNING: expected_chunk_id {eid!r} in question {question!r} "
                      f"is not a valid chunk_id — check it against list_chunks.py.")
                continue
            ranks_by_id[eid] = ranked_ids.index(eid) + 1  # 1-based rank

        found_ranks = list(ranks_by_id.values())
        best_rank = min(found_ranks) if found_ranks else None
        total_expected = len(expected_chunk_ids)

        record["expected_chunk_ranks"] = ranks_by_id  # {chunk_id: rank}, only ids found anywhere
        record["best_expected_chunk_rank"] = best_rank
        record["hit_at_k"] = {
            k: (best_rank is not None and best_rank <= k) for k in K_VALUES
        }
        record["recall_at_k"] = {
            k: (sum(1 for r in found_ranks if r <= k) / total_expected if total_expected else None)
            for k in K_VALUES
        }
        record["retrieval_succeeded"] = record["hit_at_k"][max(K_VALUES)]
    else:
        record["expected_chunk_ranks"] = {}
        record["best_expected_chunk_rank"] = None
        record["hit_at_k"] = None
        record["recall_at_k"] = None
        record["retrieval_succeeded"] = None  # not applicable — there is no correct chunk

    return record


def summarize(records):
    answerable_records = [r for r in records if r["answerable"]]
    unanswerable_records = [r for r in records if not r["answerable"]]
    n = len(answerable_records)

    print("\n" + "=" * 60)
    print("RETRIEVAL PERFORMANCE (answerable questions only)")
    print("=" * 60)
    for k in K_VALUES:
        hit_rate = sum(1 for r in answerable_records if r["hit_at_k"][k]) / n if n else float("nan")
        recall_rate = statistics.mean(r["recall_at_k"][k] for r in answerable_records) if n else float("nan")
        print(f"Hit@{k}:    {hit_rate:.2f}   (fraction of questions with >=1 expected chunk in top {k})")
        print(f"Recall@{k}: {recall_rate:.2f}   (avg. fraction of each question's expected chunks in top {k})")

    print("\n" + "=" * 60)
    print("FAILED QUESTIONS (no expected chunk in top 5)")
    print("=" * 60)
    failed = [r for r in answerable_records if not r["hit_at_k"][5]]
    if not failed:
        print("None — every answerable question had a hit within top 5.")
    for r in failed:
        print(f"- {r['question']!r}")
        print(f"    expected_chunk_ids: {r['expected_chunk_ids']}  "
              f"ranks found: {r['expected_chunk_ranks']}")
        print(f"    retrieved top5: {r['retrieved_chunk_ids_top5']}  scores: {r['top5_scores']}")

    print("\n" + "=" * 60)
    print("PER-QUESTION DETAIL (answerable)")
    print("=" * 60)
    for r in answerable_records:
        h = r["hit_at_k"]
        rc = r["recall_at_k"]
        print(f"- {r['question']!r}")
        print(f"    expected: {r['expected_chunk_ids']}  ranks found: {r['expected_chunk_ranks']}")
        print(f"    hit@1/3/5:    {h[1]}/{h[3]}/{h[5]}")
        print(f"    recall@1/3/5: {rc[1]:.2f}/{rc[3]:.2f}/{rc[5]:.2f}")
        print(f"    top5 scores: {r['top5_scores']}")

    print("\n" + "=" * 60)
    print("SCORE DISTRIBUTIONS (no threshold assumed — inspect for separation)")
    print("=" * 60)
    answerable_top_scores = sorted((r["top_score"] for r in answerable_records), reverse=True)
    unanswerable_top_scores = sorted((r["top_score"] for r in unanswerable_records), reverse=True)

    print(f"Answerable top-1 scores   (n={len(answerable_top_scores)}): {answerable_top_scores}")
    if answerable_top_scores:
        print(f"    min={min(answerable_top_scores):.3f}  "
              f"max={max(answerable_top_scores):.3f}  "
              f"mean={statistics.mean(answerable_top_scores):.3f}")

    print(f"Unanswerable top-1 scores (n={len(unanswerable_top_scores)}): {unanswerable_top_scores}")
    if unanswerable_top_scores:
        print(f"    min={min(unanswerable_top_scores):.3f}  "
              f"max={max(unanswerable_top_scores):.3f}  "
              f"mean={statistics.mean(unanswerable_top_scores):.3f}")

    if answerable_top_scores and unanswerable_top_scores:
        gap = min(answerable_top_scores) - max(unanswerable_top_scores)
        print(f"\nLowest answerable top-1 score minus highest unanswerable top-1 score: {gap:.3f}")
        print("(Positive with a clear margin suggests a defensible score-based cutoff might exist;")
        print(" near-zero or negative means the two groups overlap and a fixed threshold")
        print(" would misclassify some questions either way. This script does not pick one —")
        print(" that decision needs more data than 10-20 questions can responsibly support.)")


if __name__ == "__main__":
    recorded_corpus, eval_cases = load_eval_dataset(EVAL_DATASET_PATH)
    print(f"Loaded {len(eval_cases)} evaluation questions from {EVAL_DATASET_PATH}")

    chunks, chunk_embeddings, model = build_index(pdf_filename)
    print(f"Indexed {len(chunks)} chunks from '{pdf_filename}'")
    check_corpus_config(recorded_corpus, model)

    records = [evaluate_case(case, chunks, chunk_embeddings, model) for case in eval_cases]

    with open(RESULTS_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nFull per-question records written to {RESULTS_PATH}")

    summarize(records)
