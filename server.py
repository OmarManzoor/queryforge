"""
QueryForge Server
================
Starts a FastAPI server that loads the LLM model once at startup and keeps it
in memory, eliminating per-request initialization overhead.

Configuration
-------------
Edit the two constants below to select your provider and model before running.

Usage
-----
    python server.py

The server will be available at http://localhost:8000
Interactive API docs: http://localhost:8000/docs
"""

import uvicorn
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import LLMProvider, LLMModel
from optimizer import QueryOptimizer
from schemas import OptimizedQuery

# ---------------------------------------------------------------------------
# ✏️  Configuration — edit these two lines to change the model
# ---------------------------------------------------------------------------
PROVIDER = LLMProvider.MLX
MODEL = LLMModel.QWEN_3B_INSTRUCT
# ---------------------------------------------------------------------------

optimizer: QueryOptimizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; clean up on shutdown."""
    global optimizer
    print(f"Loading model '{MODEL}' with provider '{PROVIDER}'...")
    optimizer = QueryOptimizer(provider=PROVIDER, model_name=MODEL)
    print("Model loaded. Server is ready.")
    yield
    # Nothing to explicitly release for MLX — Python GC handles it
    optimizer = None
    print("Server shut down.")


app = FastAPI(
    title="QueryForge",
    description="Pre-retrieval query transformation middleware for RAG pipelines.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PrepareRequest(BaseModel):
    query: str
    strategy: Literal["multi_query", "hyde"] = "multi_query"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Utility"])
async def health():
    """Returns OK when the server and model are ready."""
    return {"status": "ok", "provider": PROVIDER, "model": MODEL}


@app.post("/prepare", response_model=OptimizedQuery, tags=["Query Optimization"])
async def prepare_query(request: PrepareRequest) -> OptimizedQuery:
    """
    Transforms a raw user query into an optimized retrieval payload.

    - **query**: The raw user query string.
    - **strategy**: `multi_query` (default) generates 3 alternative search phrases.
                    `hyde` generates a hypothetical ideal document for dense retrieval.
    """
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    result = await optimizer.prepare_query(
        query=request.query,
        strategy=request.strategy,
    )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
