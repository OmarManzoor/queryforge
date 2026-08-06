import asyncio
import re
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from schemas import DensePayload, OptimizedQuery, SparsePayload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimizer(mock_engine: MagicMock):
    """Build a QueryOptimizer with a pre-injected mock engine, bypassing __init__."""
    from optimizer import QueryOptimizer

    optimizer = object.__new__(QueryOptimizer)
    optimizer.llm_engine = mock_engine
    with open("./prompts.yaml", "r") as f:
        import yaml
        optimizer.prompts = yaml.safe_load(f)
    return optimizer


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestSchemas:
    """Verify Pydantic schema validation and serialization."""

    def test_optimized_query_valid_construction(self):
        """OptimizedQuery should construct correctly with valid data."""
        result = OptimizedQuery(
            original_query="Why is my code slow?",
            intent="CONCEPTUAL",
            dense_payload=DensePayload(queries=["query A", "query B"]),
            sparse_payload=SparsePayload(),
        )
        assert result.original_query == "Why is my code slow?"
        assert result.intent == "CONCEPTUAL"
        assert result.dense_payload.queries == ["query A", "query B"]
        assert result.dense_payload.hyde_document is None
        assert result.sparse_payload.keywords is None

    def test_dense_payload_defaults_to_none(self):
        """DensePayload fields should default to None when not provided."""
        payload = DensePayload()
        assert payload.queries is None
        assert payload.hyde_document is None

    def test_sparse_payload_defaults_to_none(self):
        """SparsePayload keywords should default to None when not provided."""
        payload = SparsePayload()
        assert payload.keywords is None

    def test_optimized_query_serializes_to_json(self):
        """model_dump_json() should produce valid JSON with correct keys."""
        result = OptimizedQuery(
            original_query="test",
            intent="GREETING",
            dense_payload=DensePayload(),
            sparse_payload=SparsePayload(),
        )
        json_str = result.model_dump_json()
        assert "original_query" in json_str
        assert "intent" in json_str
        assert "dense_payload" in json_str
        assert "sparse_payload" in json_str

    def test_optimized_query_missing_required_field_raises(self):
        """Constructing OptimizedQuery without required fields should raise a ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OptimizedQuery(  # type: ignore[call-arg]
                intent="CONCEPTUAL",
                dense_payload=DensePayload(),
                sparse_payload=SparsePayload(),
            )


# ---------------------------------------------------------------------------
# Parsing Logic Tests (no LLM calls)
# ---------------------------------------------------------------------------

class TestQueryOptimizerParsing:
    """Test stateless parsing methods that don't call the LLM."""

    def test_classify_intent_returns_exact_match(self):
        """'EXACT_MATCH' anywhere in LLM response should map to EXACT_MATCH."""
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "EXACT_MATCH"
        optimizer = _make_optimizer(mock_engine)
        assert optimizer.classify_intent("user_id=42") == "EXACT_MATCH"

    def test_classify_intent_returns_greeting(self):
        """'GREETING' in LLM response should map to GREETING."""
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "GREETING"
        optimizer = _make_optimizer(mock_engine)
        assert optimizer.classify_intent("Hey there!") == "GREETING"

    def test_classify_intent_defaults_to_conceptual(self):
        """Any other LLM response should fall back to CONCEPTUAL."""
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "CONCEPTUAL"
        optimizer = _make_optimizer(mock_engine)
        assert optimizer.classify_intent("Why is Python slow?") == "CONCEPTUAL"

    def test_classify_intent_is_case_insensitive(self):
        """Intent classification should be case-insensitive."""
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "exact_match"
        optimizer = _make_optimizer(mock_engine)
        assert optimizer.classify_intent("error code 404") == "EXACT_MATCH"

    def test_generate_multi_queries_parses_json_response(self):
        """generate_multi_queries should parse JSON response into dict with standalone_query and queries list."""
        mock_engine = MagicMock()
        mock_engine.generate.return_value = '{"standalone_query": "Why is Python slow?", "queries": ["slow python loops", "python performance tips", "optimize python code"]}'
        optimizer = _make_optimizer(mock_engine)
        result = optimizer.generate_multi_queries("Why is Python slow?")
        assert isinstance(result, dict)
        assert len(result["queries"]) == 3
        assert "slow python loops" in result["queries"]

    def test_generate_multi_queries_caps_at_three(self):
        """generate_multi_queries should return at most 3 queries even if LLM returns more."""
        mock_engine = MagicMock()
        mock_engine.generate.return_value = '{"standalone_query": "query", "queries": ["a", "b", "c", "d", "e"]}'
        optimizer = _make_optimizer(mock_engine)
        result = optimizer.generate_multi_queries("query")
        assert len(result["queries"]) <= 3

    def test_generate_multi_queries_fallback_on_empty(self):
        """generate_multi_queries should fallback to original query if parsing fails."""
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "  "  # whitespace only
        optimizer = _make_optimizer(mock_engine)
        result = optimizer.generate_multi_queries("my query")
        assert result == {"standalone_query": "my query", "queries": ["my query"]}

    def test_extract_keywords_extracts_noun_phrases(self):
        """extract_keywords should return a list of extracted phrase keywords via NLTK POS tagging."""
        optimizer = _make_optimizer(MagicMock())
        result = optimizer.extract_keywords("Why is Python slow?")
        assert isinstance(result, list)
        assert "python" in result

    @pytest.mark.parametrize(
        "query,expected_keywords",
        [
            (
                "Where can I buy an expensive car in New York?",
                ["expensive car", "new york"],
            ),
            (
                "Where can I quickly find the best cheap laptops for gaming?",
                ["best cheap laptops", "gaming"],
            ),
            (
                "Looking for high speed wireless noise cancelling headphones",
                ["high speed wireless noise cancelling headphones"],
            ),
        ],
    )
    def test_extract_keywords_sample_queries(self, query, expected_keywords):
        """extract_keywords should accurately parse key noun phrases using NLTK."""
        optimizer = _make_optimizer(MagicMock())
        results = optimizer.extract_keywords(query)
        assert results == expected_keywords




# ---------------------------------------------------------------------------
# Integration Tests (async, mocked LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPrepareQuery:
    """Integration tests for the async prepare_query pipeline."""

    async def test_conceptual_multi_query_populates_dense_payload(self):
        """CONCEPTUAL intent + multi_query strategy should populate dense_payload.queries."""
        mock_engine = MagicMock()
        mock_engine.generate.side_effect = [
            "CONCEPTUAL",  # classify_intent
            '{"standalone_query": "Why is Python slow?", "queries": ["slow python loops", "python performance tips", "optimize python code"]}',  # multi_query
        ]
        optimizer = _make_optimizer(mock_engine)
        result = await optimizer.prepare_query("Why is Python slow?", strategy="multi_query")

        assert result.intent == "CONCEPTUAL"
        assert result.dense_payload.queries is not None
        assert len(result.dense_payload.queries) > 0
        assert result.dense_payload.hyde_document is None
        assert result.sparse_payload.keywords == ["python"]

    async def test_conceptual_hyde_populates_dense_payload(self):
        """CONCEPTUAL intent + hyde strategy should populate dense_payload.hyde_document."""
        mock_engine = MagicMock()
        mock_engine.generate.side_effect = [
            "CONCEPTUAL",  # classify_intent
            '{"standalone_query": "Why is Python slow?", "hypothetical_document": "Python can be slow due to the GIL and inefficient loops."}',  # hyde
        ]
        optimizer = _make_optimizer(mock_engine)
        result = await optimizer.prepare_query("Why is Python slow?", strategy="hyde")

        assert result.intent == "CONCEPTUAL"
        assert result.dense_payload.hyde_document is not None
        assert "Python" in result.dense_payload.hyde_document
        assert result.dense_payload.queries is None
        assert result.sparse_payload.keywords == ["python"]

    async def test_exact_match_returns_empty_payloads(self):
        """EXACT_MATCH intent should skip generation and populate sparse payload keywords."""
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "EXACT_MATCH"
        optimizer = _make_optimizer(mock_engine)
        result = await optimizer.prepare_query("user_id=42", strategy="multi_query")

        assert result.intent == "EXACT_MATCH"
        assert result.dense_payload.queries is None
        assert result.dense_payload.hyde_document is None
        assert isinstance(result.sparse_payload.keywords, list)

    async def test_greeting_returns_empty_payloads(self):
        """GREETING intent should skip all generation and return empty payloads."""
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "GREETING"
        optimizer = _make_optimizer(mock_engine)
        result = await optimizer.prepare_query("Hey there!", strategy="multi_query")

        assert result.intent == "GREETING"
        assert result.dense_payload.queries is None
        assert result.dense_payload.hyde_document is None
        assert result.sparse_payload.keywords is None

    async def test_invalid_strategy_raises_value_error(self):
        """An unknown strategy should raise a ValueError."""
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "CONCEPTUAL"
        optimizer = _make_optimizer(mock_engine)

        with pytest.raises(ValueError, match="Unknown strategy"):
            await optimizer.prepare_query("test query", strategy="unknown_strategy")

    async def test_prepare_query_returns_optimized_query_instance(self):
        """prepare_query should always return an OptimizedQuery Pydantic model."""
        mock_engine = MagicMock()
        mock_engine.generate.side_effect = [
            "CONCEPTUAL",
            '{"standalone_query": "test", "queries": ["query a", "query b", "query c"]}',
        ]
        optimizer = _make_optimizer(mock_engine)
        result = await optimizer.prepare_query("test", strategy="multi_query")
        assert isinstance(result, OptimizedQuery)

    async def test_original_query_is_preserved(self):
        """The original_query field should always exactly match the input query."""
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "GREETING"
        optimizer = _make_optimizer(mock_engine)
        query = "Hello, how are you?"
        result = await optimizer.prepare_query(query, strategy="multi_query")
        assert result.original_query == query
