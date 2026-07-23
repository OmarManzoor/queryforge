import json
from json_repair import repair_json
import re
import yaml
import asyncio
from typing import Any, Dict, Optional

from config import LLMProvider, AVAILABLE_MODELS
from schemas import ChatMessage, OptimizedQuery, DensePayload, SparsePayload


class QueryOptimizer:
    def __init__(self, provider: str, model_name: str):
        if provider not in (LLMProvider.HUGGINGFACE, LLMProvider.MLX, LLMProvider.LLAMA):
            raise ValueError(f"{provider} provider is currently not supported")

        available_model_names = set(model.split("/")[-1] for model in AVAILABLE_MODELS)
        if model_name not in available_model_names:
            raise ValueError(f"{model_name} is current not supported")

        if provider == LLMProvider.HUGGINGFACE:
            from llm import HuggingFaceEngine
            self.llm_engine = HuggingFaceEngine(model_name)
        elif provider == LLMProvider.MLX:
            from llm import MLXEngine
            self.llm_engine = MLXEngine(model_name)
        elif provider == LLMProvider.LLAMA:
            from llm import LlamaMLXEngine
            self.llm_engine = LlamaMLXEngine(model_name)
        with open("./prompts.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)

    def classify_intent(self, query: str) -> str:
        """Classifies the query intent to avoid unnecessary processing."""
        raw_response = self.llm_engine.generate(
            system_prompt=self.prompts.get("intent", ""),
            user_prompt=query,
            temperature=0.1
        )
        intent = raw_response.strip().upper()
        if "EXACT_MATCH" in intent:
            return "EXACT_MATCH"
        elif "GREETING" in intent:
            return "GREETING"
        else:
            return "CONCEPTUAL"

    def generate_multi_queries(self, query: str, history: str) -> list[str]:
        """Expands a single query into 3 distinct search variations."""
        user_prompt = self.prompts["multi_query"]["user_template"].format(
            query=query,
            history=history.strip(),
        )
        raw_response = self.llm_engine.generate(
            system_prompt=self.prompts["multi_query"]["system"],
            user_prompt=user_prompt,
        )
        # Defensive JSON Parsing
        try:
            # Extract JSON object even if wrapped in markdown code blocks or chatter
            clean_json_str = repair_json(raw_response)
            data = json.loads(clean_json_str)

            standalone_query = data.get("standalone_query", query)
            queries = data.get("queries", [query])

            if not isinstance(queries, list) or not queries:
                queries = [query]

            clean_queries = [
                q.strip() for q in queries 
                if isinstance(q, str) and q.strip()
            ]

            return {
                "standalone_query": standalone_query,
                "queries": clean_queries[:3] if clean_queries else [query]
            }

        except (json.JSONDecodeError, AttributeError, Exception):
            # Fallback in case parsing fails entirely
            return {
                "standalone_query": query,
                "queries": [query]
            }

    def generate_hyde(self, query: str, history: str = "None") -> Dict[str, Any]:
        """Generates a hypothetical ideal document answering the query using context."""
        user_prompt = self.prompts["hyde"]["user_template"].format(
            query=query,
            history=history.strip()
        )

        raw_response = self.llm_engine.generate(
            system_prompt=self.prompts["hyde"]["system"],
            user_prompt=user_prompt,
        )

        # Defensive JSON Parsing
        try:
            # Extract JSON object even if wrapped in markdown code blocks or chatter
            clean_json_str = repair_json(raw_response)
            data = json.loads(clean_json_str)

            standalone_query = data.get("standalone_query", query)
            hypothetical_doc = data.get("hypothetical_document", "")

            if not isinstance(hypothetical_doc, str) or not hypothetical_doc.strip():
                hypothetical_doc = raw_response.strip()

            return {
                "standalone_query": standalone_query,
                "hypothetical_document": hypothetical_doc.strip()
            }

        except (json.JSONDecodeError, AttributeError, Exception):
            # Fallback in case parsing fails entirely
            return {
                "standalone_query": query,
                "hypothetical_document": raw_response.strip() if raw_response else query
            }

    def generate_sub_queries(self, query: str, history: str) -> Dict[str, Any]:
        """Decomposes a complex query into atomic sub-queries."""
        user_prompt = self.prompts["sub_queries"]["user_template"].format(
            query=query,
            history=history.strip(),
        )
        raw_response = self.llm_engine.generate(
            system_prompt=self.prompts["sub_queries"]["system"],
            user_prompt=user_prompt,
        )
        try:
            clean_json_str = repair_json(raw_response)
            data = json.loads(clean_json_str)

            standalone_query = data.get("standalone_query", query)
            sub_queries = data.get("sub_queries", [query])

            if not isinstance(sub_queries, list) or not sub_queries:
                sub_queries = [query]

            clean_sub_queries = [
                q.strip() for q in sub_queries 
                if isinstance(q, str) and q.strip()
            ]

            return {
                "standalone_query": standalone_query,
                "sub_queries": clean_sub_queries if clean_sub_queries else [query]
            }
        except Exception:
            return {
                "standalone_query": query,
                "sub_queries": [query]
            }

    async def prepare_query(
        self,
        query: str,
        strategy: str = "multi_query",
        history: Optional[list[ChatMessage]] = None,
    ) -> OptimizedQuery:
        """
        Orchestrates the entire query prep pipeline concurrently and returns structured JSON.
        :param strategy: "multi_query" or "hyde". 
        """
        # Run intent classification in a thread so it doesn't block the event loop
        intent = await asyncio.to_thread(self.classify_intent, query)
        
        dense_payload = DensePayload()
        sparse_payload = SparsePayload()
        standalone_query = query
        
        if intent == "CONCEPTUAL":
            history_ = self._format_history(history)
            if strategy == "multi_query":
                res = await asyncio.to_thread(self.generate_multi_queries, query, history_)
                standalone_query = res.get("standalone_query", query)
                dense_payload.queries = res.get("queries")
            elif strategy == "hyde":
                res = await asyncio.to_thread(self.generate_hyde, query, history_)
                standalone_query = res.get("standalone_query", query)
                dense_payload.hyde_document = res.get("hypothetical_document")
            elif strategy == "sub_queries":
                res = await asyncio.to_thread(self.generate_sub_queries, query, history_)
                standalone_query = res.get("standalone_query", query)
                dense_payload.sub_queries = res.get("sub_queries")
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
        return OptimizedQuery(
            original_query=query,
            standalone_query=standalone_query,
            intent=intent,
            dense_payload=dense_payload,
            sparse_payload=sparse_payload
        )

    def _format_history(self, history: list[ChatMessage]) -> str:
        if not history:
            return "No conversation history"

        formatted = []
        for msg in history:
            if msg.role == "user":
                formatted.append(f"The USER said: {msg.content}")
            else:
                formatted.append(f"The ASSISTANT responded: {msg.content}")
        
        return " ".join(formatted)

    # Currently not used but could be useful
    def extract_keywords(self, query: str) -> list[str]:
        """Extracts search keywords for sparse retrieval."""
        raw_response = self.llm_engine.generate(
            system_prompt=self.prompts.get("keyword_extraction", ""),
            user_prompt=query,
            temperature=0.1
        )
        clean_text = re.sub(r'[*"`\d\.]', '', raw_response)
        keywords = [k.strip() for k in clean_text.split(",") if k.strip()]
        return keywords

