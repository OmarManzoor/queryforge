from config import LOCAL_MODELS_DIR


class BaseEngine:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 60,
        temperature: float = 0.3,
    ):
        raise NotImplementedError()


class HuggingFaceEngine(BaseEngine):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = f"./{LOCAL_MODELS_DIR}/{self.model_name}"
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 60,
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
        outputs = pipe(messages, max_new_tokens=max_tokens, temperature=temperature)
        final_response = outputs[0]["generated_text"][-1]["content"]
        return final_response.strip()


class MLXEngine(BaseEngine):
    def __init__(self, model_name: str):
        from mlx_lm import load
        self.model_name = model_name
        self.model_path = f"./{LOCAL_MODELS_DIR}/{self.model_name}"
        
        self.model, self.tokenizer = load(self.model_path)
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 60,
        temperature: float = 0.3,
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
            max_tokens=max_tokens, 
            verbose=False
        )
        return response.strip()
