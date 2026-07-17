# QueryForge

Pre-retrieval query transformation middleware for RAG pipelines. QueryForge takes a raw user query and enriches it into a structured retrieval payload using a locally-run LLM — no external API calls.

---

## What it does

Given a query, QueryForge:

1. **Classifies intent** — `EXACT_MATCH`, `GREETING`, or `CONCEPTUAL`
2. For **conceptual** queries, applies one of two strategies:
   - `multi_query` — expands the query into 3 distinct search variations for dense retrieval
   - `hyde` — generates a hypothetical ideal document (HyDE) for dense retrieval
3. Returns a structured JSON payload ready to feed into a retrieval layer

---

## Supported Models

| Model | HuggingFace repo |
|---|---|
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` |
| Llama-3.2-3B-Instruct | `unsloth/Llama-3.2-3B-Instruct` |

## Supported Providers

| Provider | Description |
|---|---|
| `mlx` | Optimised inference on Apple Silicon via MLX |
| `huggingface` | Standard HuggingFace `transformers` pipeline |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

> `HF_TOKEN` is required only for downloading models. The server itself does not need it at runtime.

### 3. Download a model

```bash
python download_model.py
```

Select a model by number. It will be saved to `local_models/`.

---

## Running the server

Open `server.py` and set the provider and model at the top of the file:

```python
PROVIDER = LLMProvider.MLX          # or LLMProvider.HUGGINGFACE
MODEL    = LLMModel.QWEN_3B_INSTRUCT  # or LLMModel.LLAMA_3B_INSTRUCT
```

Then start the server:

```bash
python server.py
```

Server runs at **http://localhost:8000** — interactive docs at **http://localhost:8000/docs**.

---

## API

### `POST /prepare`

```json
{
  "query": "What causes transformer attention to fail on long sequences?",
  "strategy": "multi_query"
}
```

**Strategies:** `multi_query` (default) · `hyde`

### `GET /health`

Returns the current provider and model name.

---

## Project structure

```
queryforge/
├── server.py          # FastAPI server — entry point
├── optimizer.py       # Core query transformation pipeline
├── llm.py             # HuggingFace & MLX engine wrappers
├── config.py          # Provider/model enums and constants
├── schemas.py         # Pydantic response models
├── prompts.yaml       # System prompts for each LLM task
└── download_model.py  # CLI tool to fetch models from HuggingFace
```
