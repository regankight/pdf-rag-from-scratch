# ============================================================
# FastAPI service — step 1: skeleton.
#
# This file intentionally does nothing RAG-related yet. The goal of this
# step is just proving the server runs and responds — /search and /chat
# (which call into rag_project.py) come in later steps.
# ============================================================

from fastapi import FastAPI

app = FastAPI(title="pdf-rag-from-scratch API")


@app.get("/health")
def health():
    """Liveness check — confirms the server is up and responding."""
    return {"status": "ok"}
