from dataclasses import dataclass

@dataclass(frozen=True)
class LLMProvider:
    HUGGINGFACE: str = "huggingface"
    MLX: str = "mlx"

@dataclass(frozen=True)
class LLMModel:
    QWEN_3B_INSTRUCT: str = "Qwen2.5-3B-Instruct"
    LLAMA_3B_INSTRUCT: str = "Llama-3.2-3B-Instruct"


AVAILABLE_MODELS = [
    f"Qwen/{LLMModel.QWEN_3B_INSTRUCT}",
    f"unsloth/{LLMModel.LLAMA_3B_INSTRUCT}",
]

LOCAL_MODELS_DIR = "local_models"