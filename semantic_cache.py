import functools
from typing import Any, Callable, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class SemanticCache:
    def __init__(self, threshold: float = 0.9, max_size: int = 256):
        self.embedding_model = SentenceTransformer(f"local_models/{EMBEDDING_MODEL}")
        self.threshold = threshold
        self.max_size = max_size
        self.cache: List[Tuple[str, np.ndarray, Any]] = []

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(user_query: str, *args, **kwargs):
            query_vec = self.embedding_model.encode(user_query)
            best_sim = -np.inf
            cached_value = None

            for _, cached_vec, value in self.cache:
                similarity = self._cosine_similarity(query_vec, cached_vec)
                if similarity > best_sim:
                    best_sim = similarity
                    cached_value = value

            if best_sim >= self.threshold:
                return cached_value

            result = func(user_query, *args, **kwargs)

            # FIFO Eviction Policy
            if len(self.cache) >= self.max_size:
                self.cache.pop(0)

            self.cache.append((user_query, query_vec, result))
            return result

        return wrapper
