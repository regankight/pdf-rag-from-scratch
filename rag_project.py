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

def chunk_text(text, max_chunk_size=500, overlap=100):
    # Step 1: split on paragraph breaks (blank lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_chunk_size:
            # paragraph is short enough — keep it whole
            chunks.append(para)
        else:
            # paragraph too long — fall back to word-count chunking
            start = 0
            while start < len(words):
                end = start + max_chunk_size
                chunks.append(" ".join(words[start:end]))
                start += max_chunk_size - overlap

    return chunks

# CELL 4 — embed chunks (local, no API key)
from sentence_transformers import SentenceTransformer
import numpy as np

# Named so eval/run_eval.py can compare the config an eval dataset was built
# against to the config actually in use, instead of duplicating these values.
DEFAULT_MAX_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # small, fast, free, local

def format_chunk_id(index):
    """chunk_0000, chunk_0001, ... — readable IDs derived from list position.

    These are stable only for a fixed (PDF, max_chunk_size, overlap) triple:
    re-chunking the same PDF with different parameters, or swapping the PDF,
    changes which text lands at a given index. No hashing/versioning is done
    here yet — if you change the PDF or chunking config, re-run
    eval/list_chunks.py and update eval_dataset.json's IDs to match."""
    return f"chunk_{index:04d}"

def load_chunks(pdf_path, max_chunk_size=DEFAULT_MAX_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Extract + chunk a PDF. A chunk's ID (see format_chunk_id) is just its
    position in this list."""
    full_text = extract_text(pdf_path)
    return chunk_text(full_text, max_chunk_size, overlap)

def build_index(pdf_path, max_chunk_size=DEFAULT_MAX_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Load chunks and embed them. Returns (chunks, chunk_embeddings, model)."""
    chunks = load_chunks(pdf_path, max_chunk_size, overlap)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
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
