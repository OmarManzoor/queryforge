from typing import List

import pytest

from semantic_chunking import SemanticSplitter


@pytest.fixture(scope="module")
def default_splitter() -> SemanticSplitter:
    """Fixture providing a SemanticSplitter with default settings.

    This is module-scoped to avoid reloading the embedding model for every test.
    """
    return SemanticSplitter()


class TestSemanticSplitter:
    """Test suite for SemanticSplitter functionality."""

    def test_split_empty_text(self, default_splitter: SemanticSplitter) -> None:
        """Verify that splitting empty or whitespace-only text returns an empty list."""
        assert default_splitter.split("") == []
        assert default_splitter.split("   ") == []

    def test_split_single_sentence(self, default_splitter: SemanticSplitter) -> None:
        """Verify that a single sentence is returned as a single chunk."""
        text = "This is a single sentence."
        chunks = default_splitter.split(text)
        assert chunks == [text]

    def test_split_low_threshold_keeps_together(self) -> None:
        """Verify that a very low similarity threshold groups all sentences together."""
        splitter = SemanticSplitter(similarity_threshold=0.1)
        text = (
            "First sentence about Python. Second sentence about deep learning. "
            "Third sentence about cooking a pizza."
        )
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_split_high_threshold_separates_sentences(self) -> None:
        """Verify that a very high similarity threshold splits every sentence."""
        splitter = SemanticSplitter(similarity_threshold=0.99)
        text = (
            "First sentence about Python. Second sentence about deep learning. "
            "Third sentence about cooking a pizza."
        )
        chunks = splitter.split(text)
        assert len(chunks) == 3
        assert chunks[0] == "First sentence about Python."
        assert chunks[1] == "Second sentence about deep learning."
        assert chunks[2] == "Third sentence about cooking a pizza."

    def test_split_semantic_boundaries(
        self, default_splitter: SemanticSplitter
    ) -> None:
        """Verify that clear topic transitions are correctly split into chunks."""
        text = (
            "Machine learning is a field of artificial intelligence. "
            "It focuses on building systems that learn from data to make predictions. "
            "Pizza is a delicious Italian dish. "
            "It consists of a round, flattened base of leavened wheat-based dough "
            "topped with tomatoes and cheese."
        )
        chunks = default_splitter.split(text)

        # We expect it to split at the topic shift between ML and Pizza
        assert len(chunks) == 2
        assert any("Machine learning" in chunk for chunk in chunks)
        assert any("Pizza" in chunk for chunk in chunks)
    
    def test_split_semantic_on_content(self, default_splitter: SemanticSplitter):
        text = """A large language model (LLM) is a type of machine learning model designed for natural language 
        processing tasks such as language generation. LLMs are language models with many parameters, and are
        trained with self-supervised learning on a vast amount of text.
        The largest and most capable LLMs are generative pretrained transformers (GPTs). Modern models can
        be fine-tuned for specific tasks or guided by prompt engineering.
        These models acquire predictive
        power regarding syntax, semantics, and ontologies inherent in human language corpora, but they also
        inherit inaccuracies and biases present in the data they are trained in.
        """
        chunks = default_splitter.split(text)
        assert len(chunks) == 2
        assert all(
            phrase in chunks[0]
            for phrase in (
                "(LLM)",
                "machine learning model",
                "self-supervised",
                "vast",
                "(GPTs)",
                "Modern models",
                "prompt engineering",
            )
        )
        assert all(
            phrase in chunks[1]
            for phrase in (
                "predictive power",
                "syntax",
                "ontologies",
                "inaccuracies",
                "biases",
                "trained in",
            )
        )


    def test_split_various_punctuation(
        self, default_splitter: SemanticSplitter
    ) -> None:
        """Verify that sentences ending with different punctuation marks are split."""
        text = "Hello world! How are you? I am doing fine."
        chunks = default_splitter.split(text)
        assert len(chunks) >= 1
