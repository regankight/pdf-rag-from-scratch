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
from rag_project import pdf_filename, load_chunks, format_chunk_id

PREVIEW_CHARS = 300

if __name__ == "__main__":
    chunks = load_chunks(pdf_filename)
    print(f"Loaded {len(chunks)} chunks from '{pdf_filename}'\n")

    for index, chunk in enumerate(chunks):
        preview = chunk[:PREVIEW_CHARS].replace("\n", " ")
        print(f"[{format_chunk_id(index)}] ({len(chunk.split())} words)")
        print(preview)
        print("-" * 60)
