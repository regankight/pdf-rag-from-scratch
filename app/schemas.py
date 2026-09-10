# ============================================================
# Pydantic request/response models for the API.
#
# FastAPI uses these for three things at once: validating incoming
# request bodies, serializing responses, and generating the /docs
# schema — defining a model here is what makes a field show up (with
# its type) in the Swagger UI automatically.
# ============================================================

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str
    top_n: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class ChatRequest(BaseModel):
    question: str
    top_n: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
