from pydantic import BaseModel, Field
from typing import Optional, List, Literal


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PrepareRequest(BaseModel):
    query: str = Field(
        ...,
        description="The current/latest raw user query string.",
    )
    strategy: Literal["multi_query", "hyde", "sub_queries"] = Field(
        default="multi_query",
        description="Transformation strategy to apply.",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Optional list of recent chat messages (e.g. last 2 turns) "
            "to resolve pronouns and context."
        )
    )


class DensePayload(BaseModel):
    queries: Optional[List[str]] = None
    hyde_document: Optional[str] = None
    sub_queries: Optional[List[str]] = None


class SparsePayload(BaseModel):
    keywords: Optional[List[str]] = None


class OptimizedQuery(BaseModel):
    original_query: str
    standalone_query: Optional[str] = None
    intent: str
    dense_payload: DensePayload
    sparse_payload: SparsePayload
