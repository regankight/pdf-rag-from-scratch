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


def _build_prompt(question, chunks, chunk_embeddings, model, top_n):
    """Retrieval + prompt assembly — identical for the streaming and (if
    ever needed again) non-streaming path, so it isn't duplicated between
    them."""
    top_matches = retrieve_top_chunks(question, chunks, chunk_embeddings, model, top_n=top_n)
    return assemble_grounded_prompt(question, top_matches)


async def stream_answer(question, chunks, chunk_embeddings, model, top_n=5):
    """Retrieve, assemble the grounded prompt, and yield the answer back
    from a local Ollama model as it's generated, piece by piece, instead
    of waiting for the whole thing.

    retrieve_top_chunks() itself stays a plain (blocking) call — it's a
    single embedding lookup, a few milliseconds, not worth the extra
    complexity of pushing it to a thread pool. The generation call below
    is the part that actually takes seconds, which is why that's the
    part built as an async generator.
    """
    prompt = _build_prompt(question, chunks, chunk_embeddings, model, top_n)

    stream = await ollama.AsyncClient().chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for part in stream:
        yield part.message.content
