# ============================================================
# PDF RAG Starter — run top to bottom as a single script.
# ============================================================

# CELL 2 — set your PDF filename
# Place your PDF in the same folder as this script, then update the line below
pdf_filename = "your_file.pdf"  # <-- change this to your actual PDF filename

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

full_text = extract_text(pdf_filename)
chunks = chunk_text(full_text)
print(f"Total chunks: {len(chunks)}")
print("--- Sample chunk ---")
print(chunks[0][:300])

# CELL 4 — embed chunks (local, no API key)
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, free, local

chunk_embeddings = model.encode(chunks, show_progress_bar=True)
print(f"Embedding shape: {chunk_embeddings.shape}")
# e.g. (350, 384) -> 350 chunks, each represented by 384 numbers

# CELL 5 — cosine similarity retrieval
def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b)

def retrieve_top_chunks(question, chunks, chunk_embeddings, top_n=5):
    question_embedding = model.encode(question)
    scores = [cosine_similarity(question_embedding, emb) for emb in chunk_embeddings]
    # pair each chunk with its score, sort by score descending
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

question = "Where was the french revolution"  # <-- change this
top_matches = retrieve_top_chunks(question, chunks, chunk_embeddings)

for i, (chunk, score) in enumerate(top_matches):
    print(f"\n--- Match {i+1} (score: {score:.3f}) ---")
    print(chunk[:300])

# CELL 6 — assemble prompt to paste into claude.ai
context = "\n\n---\n\n".join([chunk for chunk, score in top_matches])

prompt = f"""Using only the context below, answer the question.
If the answer isn't in the context, say so.

CONTEXT:
{context}

QUESTION:
{question}
"""

print(prompt)
# Copy the printed output above and paste into claude.ai chat
