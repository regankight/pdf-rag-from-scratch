# ============================================================
# FastAPI service — step 2: /search.
#
# Adds the retrieval endpoint by calling straight into rag_project.py —
# no retrieval logic is duplicated here. The one new piece of machinery
# is `lifespan`: building the index (loading the embedding model +
# embedding every chunk) is expensive, so it must run once at server
# startup, not on every request.
# ============================================================

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.rag import generate_answer
from app.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse, SearchResult
from rag_project import build_index, pdf_filename, retrieve_top_chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything before `yield` runs once, before the server accepts its
    # first request. Everything after `yield` would run at shutdown —
    # there's nothing to clean up here, so this just builds the index
    # and stashes it on app.state, where request handlers can read it
    # back without rebuilding it.
    chunks, chunk_embeddings, model = build_index(pdf_filename)
    app.state.chunks = chunks
    app.state.chunk_embeddings = chunk_embeddings
    app.state.model = model
    yield


app = FastAPI(title="pdf-rag-from-scratch API", lifespan=lifespan)


@app.get("/health")
def health():
    """Liveness check — confirms the server is up and responding."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    matches = retrieve_top_chunks(
        request.question,
        app.state.chunks,
        app.state.chunk_embeddings,
        app.state.model,
        top_n=request.top_n,
    )
    return SearchResponse(
        results=[
            # score comes back as numpy.float32 — cast to a plain float
            # so Pydantic (and the JSON encoder) don't choke on it.
            SearchResult(chunk_id=chunk_id, text=text, score=float(score))
            for chunk_id, text, score in matches
        ]
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Non-streaming for now — this step just proves retrieval + a local
    # Ollama call work end to end. Streaming the answer back token by
    # token is the next step, on top of this once it's confirmed correct.
    answer = generate_answer(
        request.question,
        app.state.chunks,
        app.state.chunk_embeddings,
        app.state.model,
        top_n=request.top_n,
    )
    return ChatResponse(answer=answer)
