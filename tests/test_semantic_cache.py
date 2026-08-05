from unittest.mock import MagicMock
import pytest
from semantic_cache import SemanticCache


# 1. Session fixture: loads SentenceTransformer ONCE for all tests
@pytest.fixture(scope="session")
def semantic_cache():
    return SemanticCache(threshold=0.86)


# 2. Function fixture: automatically clears the cache before each test run
@pytest.fixture(autouse=True)
def reset_cache_state(semantic_cache):
    semantic_cache.cache.clear()


class TestSemanticCache:

    def test_cache_hit_on_exact_query(self, semantic_cache):
        mock_hyde_fn = MagicMock(side_effect=lambda q: f"Doc for {q}")
        cached_fn = semantic_cache(mock_hyde_fn)

        query = "How to optimize SQL queries?"

        res1 = cached_fn(query)
        assert res1 == f"Doc for {query}"
        assert mock_hyde_fn.call_count == 1

        res2 = cached_fn(query)
        assert res2 == f"Doc for {query}"
        assert mock_hyde_fn.call_count == 1  # Count stays at 1

    def test_cache_miss_on_new_query(self, semantic_cache):
        # Starts with an empty cache because reset_cache_state ran automatically
        mock_hyde_fn = MagicMock(side_effect=lambda q: f"Doc for {q}")
        cached_fn = semantic_cache(mock_hyde_fn)

        query = "What is database indexing?"

        res = cached_fn(query)
        assert res == f"Doc for {query}"
        assert mock_hyde_fn.call_count == 1

    def test_cache_hit_on_similar_queries(self, semantic_cache):
        # Starts with an empty cache because reset_cache_state ran automatically
        mock_hyde_fn = MagicMock(side_effect=lambda q: f"Doc for {q}")
        cached_fn = semantic_cache(mock_hyde_fn)

        queries = [
            (
                "How are large language models trained?",
                "What is the training process of large language models?",
            ),
            (
                "How to optimize PostgresSQL queries?",
                "Ways to speed up Postgres database queries",
            ),
            (
                "How do I build a REST API in Python?",
                "Building RESTful APIs using Python",
            ),
            (
                "How to evaluate ML model performance?",
                "How to evaluate machine learning model performance?",
            ),
            (
                "Explain how vector embeddings work",
                "What is the concept of vector embeddings?",
            ),
            (
                "Convert pandas DataFrame to SQL table",
                "Write a pandas DataFrame into a SQL database",
            ),
        ]

        for i, (query_orig, query_sim) in enumerate(queries, 1):
            res = res = cached_fn(query_orig)
            assert res == f"Doc for {query_orig}"
            assert mock_hyde_fn.call_count == i
            
            res = cached_fn(query_sim)
            assert res == f"Doc for {query_orig}"
            assert mock_hyde_fn.call_count == i
