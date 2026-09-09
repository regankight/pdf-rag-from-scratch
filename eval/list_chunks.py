# ============================================================
# Dump every chunk with its stable chunk_id and a text preview.
#
# Use this to build eval_dataset.json: read the previews, find which
# chunk(s) answer each of your test questions, and note their chunk_id.
#
# Run: python eval/list_chunks.py
# ============================================================

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sentence_transformers import SentenceTransformer
from rag_project import pdf_filename, load_chunks, format_chunk_id, EMBEDDING_MODEL_NAME

PREVIEW_CHARS = 300

if __name__ == "__main__":
    # Chunk boundaries depend on the embedding model's tokenizer and
    # max_seq_length (see compute_max_content_tokens in rag_project.py),
    # so the model is loaded here too — not just at embedding time.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chunks = load_chunks(pdf_filename, model)
    print(f"Loaded {len(chunks)} chunks from '{pdf_filename}'\n")

    for index, chunk in enumerate(chunks):
        preview = chunk[:PREVIEW_CHARS].replace("\n", " ")
        token_count = len(model.tokenizer.encode(chunk, add_special_tokens=False))
        print(f"[{format_chunk_id(index)}] ({token_count} tokens, {len(chunk.split())} words)")
        print(preview)
        print("-" * 60)
