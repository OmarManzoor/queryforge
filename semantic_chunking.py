import re
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class SemanticSplitter:
    def __init__(self, similarity_threshold: float = 0.70, buffer_size: int = 1):
        self.embedding_model = SentenceTransformer(f"local_models/{EMBEDDING_MODEL}")
        self.similarity_threshold = similarity_threshold
        self.buffer_size = buffer_size
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Collapses newlines, tabs, and multiple spaces into a single space."""
        return " ".join(text.split())

    def split(self, text: str) -> List[str]:
        cleaned_text = self.preprocess_text(text=text)
        sentences = [
            s.strip() for s in re.split(r'(?<=[.?!])\s+', cleaned_text) if s.strip()
        ]
        if not sentences:
            return []

        if len(sentences) == 1:
            return sentences

        # Build buffered context windows around each sentence
        buffered_texts = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            buffered_texts.append(" ".join(sentences[start:end]))

        embeddings = self.embedding_model.encode(buffered_texts, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # Avoid division by zero
        normalized_embeddings = embeddings / norms

        # Row-wise dot product between adjacent vectors gives cosine similarity
        similarities = np.sum(normalized_embeddings[:-1] * normalized_embeddings[1:], axis=1)

        # Group original sentences into semantic chunks based on similarity drops
        chunks = []
        current_chunk = [sentences[0]]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i + 1]]
            else:
                current_chunk.append(sentences[i + 1])

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
