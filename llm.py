from concurrent.futures import ThreadPoolExecutor
from config import LOCAL_MODELS_DIR, STANDARD_NEW_TOKEN_COUNT


class BaseEngine:
    def __init__(self, new_token_count: int = STANDARD_NEW_TOKEN_COUNT):
        self.max_new_tokens = new_token_count

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ):
        raise NotImplementedError()


class HuggingFaceEngine(BaseEngine):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model_path = f"./{LOCAL_MODELS_DIR}/{self.model_name}"
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ):
        from transformers import pipeline
        pipe = pipeline(
            "text-generation", 
            model=self.model_path,
            torch_dtype="auto", 
            device_map="auto"
        )

        messages = [
            {
                "role": "system", 
                "content": system_prompt,
            },
            {
                "role": "user", 
                "content": user_prompt,
            }
        ]
        outputs = pipe(messages, max_new_tokens=self.max_new_tokens, temperature=temperature)
        final_response = outputs[0]["generated_text"][-1]["content"]
        return final_response.strip()


class MLXEngine(BaseEngine):
    def __init__(self, model_name: str):
        from mlx_lm import load

        super().__init__()
        self.model_name = model_name
        self.model_path = f"./{LOCAL_MODELS_DIR}/{self.model_name}"
        self.model, self.tokenizer = load(self.model_path)
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,  # ignored
    ):
        from mlx_lm import generate
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=self.max_new_tokens, 
            verbose=False
        )
        return response.strip()


class LlamaMLXEngine(BaseEngine):
    """
    Dedicated MLX Engine for Llama models.
    Pins all model loading, cache evaluations, and generation calls to a single
    worker thread to prevent MLX cross-thread Stream(gpu, N) errors.
    """
    def __init__(self, model_name: str):
        super().__init__(new_token_count=128)
        self.model_name = model_name
        self.model_path = f"./{LOCAL_MODELS_DIR}/{self.model_name}"
        
        # Dedicated single-threaded executor for MLX GPU stream isolation
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LlamaWorker")
        
        # Synchronously load model ON the dedicated thread
        self._executor.submit(self._init_model_sync).result()

    def _init_model_sync(self):
        import mlx.core as mx
        from mlx_lm import load

        self.model, self.tokenizer = load(self.model_path)
        # Force-evaluate lazy weights/RoPE parameters on the pinned thread
        mx.eval(self.model.parameters())

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,  # ignored
    ) -> str:
        # Route execution to the pinned thread
        future = self._executor.submit(
            self._generate_sync, system_prompt, user_prompt, temperature
        )
        return future.result()

    def _generate_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        from mlx_lm import generate

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            verbose=False
        )
        return response.strip()
