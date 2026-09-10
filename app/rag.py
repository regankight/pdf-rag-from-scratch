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


class OllamaUnavailableError(RuntimeError):
    """The Ollama daemon isn't reachable, or OLLAMA_MODEL hasn't been
    pulled. Raised as its own type (rather than leaking ConnectionError /
    ollama.ResponseError to the route) so app/main.py can catch exactly
    this and turn it into a clean HTTP error response — see the comment
    on /chat in app/main.py for why that has to happen before the first
    chunk is yielded, not inside the stream."""


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

    Both ways this can fail are caught here and re-raised as
    OllamaUnavailableError: the daemon being unreachable raises
    ConnectionError on the initial call, but a model that hasn't been
    pulled doesn't fail until the *first chunk* is pulled from the
    stream — confirmed by testing both directly against this Ollama
    client rather than assumed. Catching both around the same `async for`
    means either failure surfaces on this generator's first `yield`,
    which is what lets app/main.py detect it before any response has
    been sent to the client.
    """
    prompt = _build_prompt(question, chunks, chunk_embeddings, model, top_n)

    try:
        stream = await ollama.AsyncClient().chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for part in stream:
            yield part.message.content
    except ConnectionError as exc:
        raise OllamaUnavailableError(
            f"Can't reach the Ollama daemon. Is it running? ({exc})"
        ) from exc
    except ollama.ResponseError as exc:
        raise OllamaUnavailableError(
            f"Ollama model '{OLLAMA_MODEL}' isn't available — "
            f"run `ollama pull {OLLAMA_MODEL}`. ({exc})"
        ) from exc
