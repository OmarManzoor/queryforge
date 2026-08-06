import re
from typing import List

import numpy as np
import pysbd
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class SemanticSplitter:
    def __init__(
        self,
        percentile_threshold: float = 25.0,
        buffer_size: int = 1,
        min_chunk_words: int = 40,
        max_chunk_words: int = 200,
    ):
        self.embedding_model = SentenceTransformer(f"local_models/{EMBEDDING_MODEL}")
        self.percentile_threshold = percentile_threshold
        self.buffer_size = buffer_size
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def _clean_text(self, text: str) -> str:
        """Strips citations, cleans broken periods, and normalizes space."""
        # Remove bracketed citations e.g. [128], [133][134]
        text = re.sub(r"\[\d+\]", "", text)
        # Fix broken ellipses/periods (e.g. "man.." or ".. .")
        text = re.sub(r"\s+\.", ".", text)
        text = re.sub(r"\.{2,}", ".", text)
        # Collapse whitespace
        return " ".join(text.split())

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merges chunks smaller than min_chunk_words into adjacent chunks."""
        merged = []
        current = ""

        for chunk in chunks:
            if not current:
                current = chunk
            elif len(current.split()) < self.min_chunk_words:
                current = f"{current} {chunk}"
            else:
                merged.append(current)
                current = chunk

        if current:
            if merged and len(current.split()) < self.min_chunk_words:
                merged[-1] = f"{merged[-1]} {current}"
            else:
                merged.append(current)

        return merged

    def split(self, text: str) -> List[str]:
        cleaned_text = self._clean_text(text=text)
        sentences = [
            s.strip() for s in self.segmenter.segment(cleaned_text) if s.strip()
        ]
        if not sentences:
            return []

        if len(sentences) == 1:
            return sentences

        # 1. Build context buffers
        buffered = [
            " ".join(
                sentences[
                    max(0, i - self.buffer_size) : min(
                        len(sentences), i + self.buffer_size + 1
                    )
                ]
            )
            for i in range(len(sentences))
        ]

        embeddings = self.embedding_model.encode(buffered, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        norm_emb = embeddings / norms
        similarities = np.sum(norm_emb[:-1] * norm_emb[1:], axis=1)

        # Dynamic Thresholding: Split at relative drops rather than a fixed cutoff
        cutoff = np.percentile(similarities, self.percentile_threshold)
        raw_chunks = []
        curr = [sentences[0]]
        for i, sim in enumerate(similarities):
            # Split if similarity drops below percentile OR current chunk exceeds max words
            if sim < cutoff or len(" ".join(curr).split()) >= self.max_chunk_words:
                raw_chunks.append(" ".join(curr))
                curr = [sentences[i + 1]]
            else:
                curr.append(sentences[i + 1])

        if curr:
            raw_chunks.append(" ".join(curr))

        # Post-process: Force-merge micro-chunks
        return self._merge_small_chunks(raw_chunks)
