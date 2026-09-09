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

def chunk_text(text, tokenizer, max_content_tokens, overlap_tokens):
    """Paragraph-aware, tokenizer-aware chunking.

    Sizes are measured in the embedding model's own tokens, not words —
    words are a poor proxy for token count (word-piece tokenizers routinely
    produce 1.3-2x as many tokens as words), which is what let oversized
    chunks reach the embedding model silently truncated in the first place.

    max_content_tokens is the token budget for chunk *content*, i.e. after
    the tokenizer's special tokens (e.g. [CLS]/[SEP]) are already accounted
    for — see compute_max_content_tokens(). overlap_tokens is likewise a
    token count, not a word count.

    A paragraph that fits is kept whole (paragraph boundaries are
    preserved). An oversized paragraph is split into overlapping windows
    using the tokenizer's fast offset mapping, so each chunk is a slice of
    the *original* text at exact character boundaries — not
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
            # paragraph too long — fall back to token-count chunking,
            # slicing the original string by the token offsets so the
            # chunk text is exactly what was in the source
            start = 0
            while start < len(offsets):
                end = min(start + max_content_tokens, len(offsets))
                char_start = offsets[start][0]
                char_end = offsets[end - 1][1]
                chunks.append(para[char_start:char_end])
                if end == len(offsets):
                    break
                start += max_content_tokens - overlap_tokens

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
    max_seq_length."""
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

    # CELL 6 — assemble prompt to paste into claude.ai
    context = "\n\n---\n\n".join([chunk for chunk_id, chunk, score in top_matches])

    prompt = f"""Using only the context below, answer the question.
If the answer isn't in the context, say so.

CONTEXT:
{context}

QUESTION:
{question}
"""

    print(prompt)
    # Copy the printed output above and paste into claude.ai chat
