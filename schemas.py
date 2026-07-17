from pydantic import BaseModel
from typing import Optional, List

class DensePayload(BaseModel):
    queries: Optional[List[str]] = None
    hyde_document: Optional[str] = None

class SparsePayload(BaseModel):
    keywords: Optional[List[str]] = None

class OptimizedQuery(BaseModel):
    original_query: str
    intent: str
    dense_payload: DensePayload
    sparse_payload: SparsePayload
