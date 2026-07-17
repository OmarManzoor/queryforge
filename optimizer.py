import re
import yaml
import asyncio
from transformers import pipeline

from config import LLMProvider, LLMModel
from schemas import OptimizedQuery, DensePayload, SparsePayload


class QueryOptimizer:
    def __init__(self, provider: str, model_name: str):
        if provider not in (LLMProvider.HUGGINGFACE, LLMProvider.MLX):
            raise ValueError(f"{provider} provider is currently not supported")

        if model_name not in (LLMModel.QWEN_3B_INSTRUCT, LLMModel.LLAMA_3B_INSTRUCT):
            raise ValueError(f"{model_name} is current not supported")
        
        if provider == LLMProvider.HUGGINGFACE:
            from llm import HuggingFaceEngine
            self.llm_engine = HuggingFaceEngine(model_name)
        elif provider == LLMProvider.MLX:
            from llm import MLXEngine
            self.llm_engine = MLXEngine(model_name)
        with open("./prompts.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)

    def classify_intent(self, query: str) -> str:
        """Classifies the query intent to avoid unnecessary processing."""
        raw_response = self.llm_engine.generate(
            system_prompt=self.prompts.get("intent", ""),
            user_prompt=query,
            max_tokens=10,
            temperature=0.0
        )
        intent = raw_response.strip().upper()
        if "EXACT_MATCH" in intent:
            return "EXACT_MATCH"
        elif "GREETING" in intent:
            return "GREETING"
        else:
            return "CONCEPTUAL"

    def generate_multi_queries(self, query: str) -> list[str]:
        """Expands a single query into 3 distinct search variations."""
        raw_response = self.llm_engine.generate(
            system_prompt=self.prompts.get("multi_query", ""),
            user_prompt=query
        )
        # Defensive Parsing: Clean up common LLM formatting quirks
        # Remove markdown bolding, quotes, and leading numbers
        clean_text = re.sub(r'[*"`\d\.]', '', raw_response)
        
        # Split by comma and strip whitespace
        queries = [q.strip() for q in clean_text.split(",") if q.strip()]
        
        # Fallback in case parsing fails entirely
        if not queries:
            return [query]
            
        return queries[:3] # Ensure we return at most 3

    def generate_hyde(self, query: str) -> str:
        """Generates a hypothetical ideal passage answering the query."""
        return self.llm_engine.generate(
            system_prompt=self.prompts.get("hyde", ""),
            user_prompt=query,
        )

    def extract_keywords(self, query: str) -> list[str]:
        """Extracts search keywords for sparse retrieval."""
        raw_response = self.llm_engine.generate(
            system_prompt=self.prompts.get("keyword_extraction", ""),
            user_prompt=query,
            max_tokens=30,
            temperature=0.1
        )
        clean_text = re.sub(r'[*"`\d\.]', '', raw_response)
        keywords = [k.strip() for k in clean_text.split(",") if k.strip()]
        return keywords

    async def prepare_query(self, query: str, strategy: str = "multi_query") -> OptimizedQuery:
        """
        Orchestrates the entire query prep pipeline concurrently and returns structured JSON.
        :param strategy: "multi_query" or "hyde". 
        """
        # Run intent classification in a thread so it doesn't block the event loop
        intent = await asyncio.to_thread(self.classify_intent, query)
        
        dense_payload = DensePayload()
        sparse_payload = SparsePayload()
        
        if intent != "CONCEPTUAL":
            pass
        else: # CONCEPTUAL
            if strategy == "multi_query":
                queries = await asyncio.to_thread(self.generate_multi_queries, query)
                dense_payload.queries = queries
            elif strategy == "hyde":
                hyde_doc = await asyncio.to_thread(self.generate_hyde, query)
                dense_payload.hyde_document = hyde_doc
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
        return OptimizedQuery(
            original_query=query,
            intent=intent,
            dense_payload=dense_payload,
            sparse_payload=sparse_payload
        )
