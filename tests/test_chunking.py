# ============================================================
# Tests for the tokenizer-aware chunking fix in rag_project.py.
#
# Run: pytest tests/test_chunking.py -v
#
# Uses the real embedding model's tokenizer (all-MiniLM-L6-v2) rather than
# a mock, since the whole point of the fix is agreement with *this specific*
# model's max_seq_length and special-token handling — a mock tokenizer
# could pass while the real one still truncates.
# ============================================================

import os
import sys

import pytest
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag_project import (
    chunk_text,
    compute_max_content_tokens,
    validate_chunk_fits,
    EMBEDDING_MODEL_NAME,
)


@pytest.fixture(scope="module")
def model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@pytest.fixture(scope="module")
def max_content_tokens(model):
    return compute_max_content_tokens(model)


def token_count(text, model, add_special_tokens=False):
    return len(model.tokenizer.encode(text, add_special_tokens=add_special_tokens))


_COMMON_WORDS = (
    "the quick brown fox jumps over lazy dog while a small cat "
    "sleeps near warm sunny window during long slow afternoon"
).split()


def make_paragraph(word_count):
    """A paragraph of `word_count` real dictionary words (cycled from a
    small pool), each of which tokenizes to a single whole-word token in
    the BERT/WordPiece vocab this model uses — unlike synthetic tokens
    (e.g. "word123"), which BERT's tokenizer splits into several
    sub-word pieces and would make word-boundary assertions in these
    tests unreliable. Using real words keeps word count and token count
    aligned 1:1, and keeps chunk boundaries landing on word boundaries,
    matching how the real PDF text this chunker runs on tokenizes."""
    return " ".join(_COMMON_WORDS[i % len(_COMMON_WORDS)] for i in range(word_count))


# --- a. a normal paragraph that fits -----------------------------------

def test_paragraph_that_fits_is_kept_whole(model, max_content_tokens):
    para = "Supervised learning uses labeled data to train a model."
    text = para  # single paragraph, no blank-line splits
    chunks = chunk_text(text, model.tokenizer, max_content_tokens, overlap_tokens=10)

    assert chunks == [para]
    assert token_count(chunks[0], model) <= max_content_tokens


# --- b. multiple adjacent paragraphs, each within budget ---------------
#
# Note: the current design chunks paragraphs independently — a paragraph
# that fits becomes exactly one chunk, and paragraphs are never merged
# together into a single chunk. This test verifies several adjacent short
# paragraphs are each processed correctly and none exceeds the limit; it
# deliberately does not assert they get merged into fewer chunks, since
# merging isn't part of the repair this fix makes (see PR description).

def test_multiple_adjacent_short_paragraphs_all_stay_within_limit(model, max_content_tokens):
    paragraphs = [
        "Regression predicts a continuous value instead of a category.",
        "Clustering groups similar data points without labels.",
        "Dimensionality reduction compresses features while preserving structure.",
    ]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, model.tokenizer, max_content_tokens, overlap_tokens=10)

    assert chunks == paragraphs  # each paragraph preserved as its own chunk
    for chunk in chunks:
        assert token_count(chunk, model) <= max_content_tokens


# --- c. one oversized paragraph -----------------------------------------

def test_oversized_paragraph_is_split_into_multiple_chunks(model, max_content_tokens):
    para = make_paragraph(max_content_tokens * 3)  # way over budget
    chunks = chunk_text(para, model.tokenizer, max_content_tokens, overlap_tokens=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert token_count(chunk, model) <= max_content_tokens


# --- d. overlap between oversized-paragraph chunks ----------------------

def test_oversized_paragraph_chunks_overlap(model, max_content_tokens):
    para = make_paragraph(max_content_tokens * 2)
    overlap_tokens = 20
    chunks = chunk_text(para, model.tokenizer, max_content_tokens, overlap_tokens)

    assert len(chunks) >= 2
    # Each word here is exactly one token (see make_paragraph), and both
    # chunks are full-length, so chunk N's last overlap_tokens words must
    # equal chunk N+1's first overlap_tokens words exactly.
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-overlap_tokens:] == second_words[:overlap_tokens]


# --- e. no generated chunk exceeds the model's effective limit ---------

def test_no_chunk_from_a_mixed_document_exceeds_the_model_limit(model, max_content_tokens):
    text = "\n\n".join(
        [
            "A short paragraph that fits easily.",
            make_paragraph(max_content_tokens // 2),  # fits, but close-ish
            make_paragraph(max_content_tokens * 5),  # badly oversized
            "Another short one.",
        ]
    )
    chunks = chunk_text(text, model.tokenizer, max_content_tokens, overlap_tokens=30)

    assert len(chunks) > 1
    for chunk in chunks:
        # This is the real guarantee: including the special tokens the
        # model actually adds, every chunk must fit max_seq_length.
        assert token_count(chunk, model, add_special_tokens=True) <= model.max_seq_length


# --- f. the embedding guard rejects an oversized input ------------------

def test_validate_chunk_fits_raises_on_oversized_chunk(model):
    too_long = make_paragraph(model.max_seq_length * 3)
    with pytest.raises(ValueError):
        validate_chunk_fits(too_long, model)


def test_validate_chunk_fits_accepts_a_normal_chunk(model):
    fine = "A short chunk that comfortably fits the model."
    validate_chunk_fits(fine, model)  # should not raise
