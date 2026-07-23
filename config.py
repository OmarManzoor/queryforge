from dataclasses import dataclass

@dataclass(frozen=True)
class LLMProvider:
    HUGGINGFACE: str = "huggingface"
    MLX: str = "mlx"
    LLAMA: str = "llama"

@dataclass(frozen=True)
class LLMModel:
    QWEN_3B_INSTRUCT: str = "Qwen2.5-3B-Instruct"
    QWEN_3B_INSTRUCT_4_BIT: str = "Qwen2.5-3B-Instruct-4bit"
    QWEN3_4B_Instruct_4_BIT: str = "Qwen3-4B-Instruct-2507-4bit"
    LLAMA_3B_INSTRUCT: str = "Llama-3.2-3B-Instruct"
    LLAMA_3B_INSTRUCT_4_BIT: str = "Llama-3.2-3B-Instruct-4bit"


AVAILABLE_MODELS = [
    f"mlx-community/{LLMModel.QWEN_3B_INSTRUCT_4_BIT}",
    f"mlx-community/{LLMModel.QWEN3_4B_Instruct_4_BIT}",
    f"mlx-community/{LLMModel.LLAMA_3B_INSTRUCT_4_BIT}",
    f"Qwen/{LLMModel.QWEN_3B_INSTRUCT}",
    f"unsloth/{LLMModel.LLAMA_3B_INSTRUCT}",
]

LOCAL_MODELS_DIR = "local_models"

STANDARD_NEW_TOKEN_COUNT = 100
