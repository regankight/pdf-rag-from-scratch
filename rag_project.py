# ============================================================
# PDF RAG Starter — run top to bottom as a single script.
# ============================================================

# CELL 2 — set your PDF filename
# Canonical path for the evaluation corpus — place your PDF here.
pdf_filename = "data/benchmark.pdf"

# CELL 3 — extract & chunk text
from pypdf import PdfReader
import re

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        page_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', page_text)
        full_text += page_text + "\n\n"
    return full_text

def _find_words(para):
    """(start_char, end_char) for each whitespace-delimited run, in the
    same splitting semantics as str.split(). A token's offset span never
    straddles a whitespace gap (whitespace itself isn't tokenized), so
    every token falls entirely within exactly one of these spans — which
    is what makes the token<->word mapping in _split_oversized_paragraph
    reliable regardless of how the tokenizer sub-splits punctuation."""
    return [m.span() for m in re.finditer(r'\S+', para)]

def _split_oversized_paragraph(para, offsets, max_content_tokens, overlap_tokens):
    """Split one over-budget paragraph into overlapping chunks whose cuts
    land on whole-word boundaries, not mid-word — while still guaranteeing
    every chunk's real token count (offsets already exclude special
    tokens) stays within max_content_tokens.

    Windows are built by walking whole words and summing each word's own
    token count (from `offsets`) rather than slicing at a raw token index,
    which is what let a chunk start or end mid-word before.

    One deliberate exception: a single word whose own token count exceeds
    max_content_tokens cannot be kept whole without violating the hard
    max_seq_length guarantee, which takes priority — that case falls back
    to a token-exact cut inside that one word. This is currently dead code
    for all-MiniLM-L6-v2's BERT/WordPiece tokenizer specifically: its
    WordpieceTokenizer caps any single word at max_input_chars_per_word=100
    characters (confirmed: a 99-char word tops out at 49 tokens; a
    101-char one collapses straight to a single [UNK] token instead of
    being split further) — well under any max_content_tokens this model
    could produce (254), so no real word can ever take this branch with
    this model. It's kept as a defensive fallback for portability to a
    different embedding model whose tokenizer doesn't cap word length
    the same way (see test_split_oversized_paragraph_falls_back_safely_
    for_a_single_giant_word, which exercises it directly since real text
    can't).
    """
    words = _find_words(para)

    # Map each token to its word by walking both lists once (both are in
    # left-to-right character order), and record each word's first token
    # index and total token count along the way.
    token_word_idx = []
    word_first_token = [0] * len(words)
    word_token_counts = [0] * len(words)
    w = 0
    for i, (tok_start, _tok_end) in enumerate(offsets):
        while w + 1 < len(words) and tok_start >= words[w][1]:
            w += 1
        if not token_word_idx or token_word_idx[-1] != w:
            word_first_token[w] = i
        token_word_idx.append(w)
        word_token_counts[w] += 1

    chunks = []
    n_words = len(words)
    start_w = 0
    while start_w < n_words:
        end_w = start_w
        token_total = 0
        # Grow the window one whole word at a time while it still fits.
        while end_w < n_words and token_total + word_token_counts[end_w] <= max_content_tokens:
            token_total += word_token_counts[end_w]
            end_w += 1

        if end_w == start_w:
            # A lone word already exceeds the budget — see docstring.
            # There's no word boundary to align to *within* it, so slice
            # it directly by token offsets, looping until every one of
            # its tokens has been consumed — not just the first window's
            # worth (a single window here would silently drop the rest
            # of the word for anything more than one budget past it).
            first_tok = word_first_token[start_w]
            last_tok = first_tok + word_token_counts[start_w]  # exclusive
            tok_start = first_tok
            while tok_start < last_tok:
                tok_end = min(tok_start + max_content_tokens, last_tok)
                char_start = offsets[tok_start][0]
                char_end = offsets[tok_end - 1][1]
                chunks.append(para[char_start:char_end])
                if tok_end == last_tok:
                    break
                tok_start += max_content_tokens - overlap_tokens
            end_w = start_w + 1
        else:
            char_start = words[start_w][0]
            char_end = words[end_w - 1][1]
            chunks.append(para[char_start:char_end])

        if end_w >= n_words:
            break

        # Step back roughly overlap_tokens worth of whole words for the
        # next window's start. Bounded to > start_w so start_w strictly
        # increases every iteration — guarantees forward progress.
        new_start_w = end_w
        back_tokens = 0
        while new_start_w > start_w + 1 and back_tokens < overlap_tokens:
            new_start_w -= 1
            back_tokens += word_token_counts[new_start_w]
        start_w = new_start_w

    return chunks

def chunk_text(text, tokenizer, max_content_tokens, overlap_tokens):
    """Paragraph-aware, tokenizer-aware chunking.

    Sizes are measured in the embedding model's own tokens, not words —
    words are a poor proxy for token count (word-piece tokenizers routinely
    produce 1.3-2x as many tokens as words), which is what let oversized
    chunks reach the embedding model silently truncated in the first place.

    max_content_tokens is the token budget for chunk *content*, i.e. after
    the tokenizer's special tokens (e.g. [CLS]/[SEP]) are already accounted
    for — see compute_max_content_tokens(). overlap_tokens is likewise a
    token count, not a word count (the actual overlap is an approximation
    of it, rounded to whole words — see _split_oversized_paragraph).

    A paragraph that fits is kept whole (paragraph boundaries are
    preserved). An oversized paragraph is split into overlapping windows
    aligned to whole-word boundaries (not raw token offsets, which could
    cut a chunk mid-word), using the tokenizer's fast offset mapping to
    measure each word's real token cost. Each chunk is a slice of the
    *original* text at exact character boundaries — not
    tokenizer.decode() output, which can reintroduce word-piece artifacts
    (dropped spacing, merged sub-words) that don't appear in the source.
    """
    # Step 1: split on paragraph breaks (blank lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        encoding = tokenizer(para, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]

        if len(offsets) <= max_content_tokens:
            # paragraph is short enough — keep it whole
            chunks.append(para)
        else:
            chunks.extend(_split_oversized_paragraph(para, offsets, max_content_tokens, overlap_tokens))

    return chunks

# CELL 4 — embed chunks (local, no API key)
from sentence_transformers import SentenceTransformer
import numpy as np

# Named so eval/run_eval.py can compare the config an eval dataset was built
# against to the config actually in use, instead of duplicating these values.
# There is no default max-chunk-size constant any more: the content-token
# budget is *derived* from the active embedding model (see
# compute_max_content_tokens), not a fixed number, because it depends on
# that model's max_seq_length and how many special tokens its tokenizer
# adds. Only the overlap is a free choice, so it stays a constant.
DEFAULT_OVERLAP_TOKENS = 50
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # small, fast, free, local

def compute_max_content_tokens(model):
    """The token budget available for chunk *content*: the model's
    max_seq_length minus however many special tokens (e.g. [CLS]/[SEP]) its
    tokenizer adds automatically. SentenceTransformer truncates the *total*
    sequence (content + special tokens) to max_seq_length, so special
    tokens must be reserved out of the budget up front, not appended on
    top of it."""
    reserved = model.tokenizer.num_special_tokens_to_add(pair=False)
    return model.max_seq_length - reserved

def validate_chunk_fits(chunk, model):
    """Guard against a chunk silently reaching the embedding model
    truncated. Raises ValueError (loud, not silent) if a chunk — with the
    same special tokens the model will actually add — would exceed
    max_seq_length.

    tokenizer.encode() here is called with no truncation argument, which
    HF tokenizers default to off — this measures the chunk's real,
    untruncated token count, not a tensor SentenceTransformer.encode()
    (or .tokenize()) would already have truncated to max_seq_length. A
    guard built on the latter could never fire, since a truncated tensor
    is always <= max_seq_length by construction — see
    test_validate_chunk_fits_measures_untruncated_length."""
    token_count = len(model.tokenizer.encode(chunk, add_special_tokens=True))
    if token_count > model.max_seq_length:
        raise ValueError(
            f"Chunk has {token_count} tokens (including special tokens), "
            f"which exceeds {EMBEDDING_MODEL_NAME}'s max_seq_length of "
            f"{model.max_seq_length}. This chunk would have been silently "
            f"truncated during embedding. Chunk preview: {chunk[:80]!r}..."
        )

def format_chunk_id(index):
    """chunk_0000, chunk_0001, ... — readable IDs derived from list position.

    These are stable only for a fixed (PDF, embedding model, overlap)
    combination — the content-token budget is derived from the embedding
    model, so changing the model changes it too. Re-chunking the same PDF
    with a different overlap or model, or swapping the PDF, changes which
    text lands at a given index. No hashing/versioning is done here yet —
    if you change the PDF or chunking config, re-run eval/list_chunks.py
    and update eval_dataset.json's IDs to match."""
    return f"chunk_{index:04d}"

def load_chunks(pdf_path, model, overlap_tokens=DEFAULT_OVERLAP_TOKENS):
    """Extract + chunk a PDF against a given (already-loaded) embedding
    model's tokenizer and token budget. A chunk's ID (see format_chunk_id)
    is just its position in this list.

    Takes the model explicitly (rather than loading one internally, or
    reading a module global) so callers control which model's tokenizer
    and max_seq_length the chunk boundaries are computed against — the
    same explicit-dependency style retrieve_top_chunks already uses."""
    full_text = extract_text(pdf_path)
    max_content_tokens = compute_max_content_tokens(model)
    return chunk_text(full_text, model.tokenizer, max_content_tokens, overlap_tokens)

def build_index(pdf_path, overlap_tokens=DEFAULT_OVERLAP_TOKENS, embedding_model_name=EMBEDDING_MODEL_NAME):
    """Load chunks, verify each one actually fits the model, and embed
    them. Returns (chunks, chunk_embeddings, model)."""
    model = SentenceTransformer(embedding_model_name)
    chunks = load_chunks(pdf_path, model, overlap_tokens)
    for chunk in chunks:
        validate_chunk_fits(chunk, model)
    chunk_embeddings = model.encode(chunks, show_progress_bar=True)
    return chunks, chunk_embeddings, model

# CELL 5 — cosine similarity retrieval
def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b)

def retrieve_top_chunks(question, chunks, chunk_embeddings, model, top_n=5):
    """Returns the top_n matches as (chunk_id, chunk_text, score) tuples,
    ranked by cosine similarity descending. chunk_id is the chunk's readable
    ID (see format_chunk_id). Takes model explicitly rather than relying on
    a module-level global, so callers (e.g. eval/run_eval.py) control which
    index/model a retrieval call runs against."""
    question_embedding = model.encode(question)
    scores = [cosine_similarity(question_embedding, emb) for emb in chunk_embeddings]
    # pair each chunk (with its index) and score, sort by score descending
    ranked = sorted(enumerate(zip(chunks, scores)), key=lambda x: x[1][1], reverse=True)
    return [(format_chunk_id(idx), chunk, score) for idx, (chunk, score) in ranked[:top_n]]

# CELL 6 — assemble a grounded prompt from retrieved chunks
def assemble_grounded_prompt(question, top_matches):
    """Builds the "answer only from this context" prompt from a
    retrieve_top_chunks() result. Pulled out as its own function so the
    FastAPI /chat endpoint (app/rag.py) can reuse the exact same prompt
    the script below prints — not a second, hand-copied version of it."""
    context = "\n\n---\n\n".join([chunk for chunk_id, chunk, score in top_matches])

    return f"""Using only the context below, answer the question.
If the answer isn't in the context, say so.

CONTEXT:
{context}

QUESTION:
{question}
"""

if __name__ == "__main__":
    chunks, chunk_embeddings, model = build_index(pdf_filename)
    print(f"Total chunks: {len(chunks)}")
    print("--- Sample chunk ---")
    print(chunks[0][:300])
    print(f"Embedding shape: {chunk_embeddings.shape}")
    # e.g. (350, 384) -> 350 chunks, each represented by 384 numbers

    question = "Where was the french revolution"  # <-- change this
    top_matches = retrieve_top_chunks(question, chunks, chunk_embeddings, model)

    for i, (chunk_id, chunk, score) in enumerate(top_matches):
        print(f"\n--- Match {i+1} (chunk_id: {chunk_id}, score: {score:.3f}) ---")
        print(chunk[:300])

    print(assemble_grounded_prompt(question, top_matches))
    # Copy the printed output above and paste into claude.ai chat
