# ============================================================
# FastAPI service — generation adapter.
#
# Bridges retrieval (rag_project.py, unchanged) to a local Ollama model.
# Kept as its own module rather than inline in main.py so the "how do we
# turn a question into an answer" logic isn't tangled up with HTTP
# routing concerns.
# ============================================================

import ollama

from rag_project import assemble_grounded_prompt, retrieve_top_chunks

OLLAMA_MODEL = "llama3.2"


def generate_answer(question, chunks, chunk_embeddings, model, top_n=5):
    """Retrieve, assemble the grounded prompt, and get a non-streaming
    answer back from a local Ollama model. No API key, no network call
    outside localhost — same "no black box, no cost" spirit as the
    retrieval side of this project."""
    top_matches = retrieve_top_chunks(question, chunks, chunk_embeddings, model, top_n=top_n)
    prompt = assemble_grounded_prompt(question, top_matches)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content
