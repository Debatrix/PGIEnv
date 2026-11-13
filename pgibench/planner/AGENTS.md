# CLAUDE.md — Planner & Model Layer

## Architecture

```
VLMPlanner                                # VLM interaction core
    ├── RemoteModel                       # Unified API client (Anthropic SDK + OpenAI-compatible)
    ├── CustomModel                       # Local model server client
    ├── model_configs/*.yaml              # 11 model configs
    └── planner_config/generation_guide.py # LLM/VLM output format instructions
```

## VLMPlanner (`vlm_planner.py`)

Class `VLMPlanner` handles MLLM interaction, action selection, EQA question dispatching.

**Core API:**
- `act(img_path, instruction)` → `(action, reasoning)` — main decision entry point
- `ask_eqa_question(question)` → model response — EQA evaluation Q&A

**VLM Interaction Flow:**
1. Construct prompt (system + instruction + image)
2. Image → base64 for API
3. `RemoteModel` / `CustomModel` sends to provider
4. Response parsed via `fix_json()` → action ID mapped to action space
5. Environment validates and executes action

**VLM Input:** System prompt + task instruction + action space list + history feedback + current RGB image (base64)

**VLM Output:** JSON format:
```json
{
  "visual_state_description": "...",
  "reasoning_and_reflection": "...",
  "executable_plan": [{"action_id": N}]
}
```

## Model Layer

| File | Function |
|------|----------|
| `remote_model.py` | Unified API client (Anthropic SDK + OpenAI-compatible endpoints) |
| `custom_model.py` | Local model server client (Flask-style; posts to the URL in the `server_url` env var, selected with `model_type=custom`) |
| `model_configs/*.yaml` | Per-model configuration (11 models) |
| `planner_config/generation_guide.py` | LLM/VLM output format instructions |

## Supported Models (11 configs)

All configs live in `pgibench/planner/model_configs/`, one `<model-name>.yaml` per model (select with `model_name=<filename without .yaml>`):

| api_type | Endpoint | Models |
| -------- | -------- | ------ |
| `claude` | Official Anthropic API (`ANTHROPIC_API_KEY`) | claude-haiku-4-5-20251001, claude-opus-4-7, claude-sonnet-4-6 |
| `openai` | Official OpenAI API (`OPENAI_API_KEY`, `base_url: null`) | gpt-5.4-mini, gpt-5.4-nano, gpt-5.5 |
| `openai` | Gemini OpenAI-compatible endpoint (`GEMINI_API_KEY`, `base_url: https://generativelanguage.googleapis.com/v1beta/openai/`) | gemini-3-flash-preview |
| `openai` | Local vLLM-style server at `http://127.0.0.1:8000/v1` (`CUSTOM_API_KEY`) | qwen3.5-0.8b, qwen3.5-2b, qwen3.5-4b, Qwen3.6-35B-A3B (config file `Qwen3.6-35B-A3B.yaml`, `model_name: qwen3.6-35b-a3b`) |

## Adding a New Model

Create `pgibench/planner/model_configs/{model-name}.yaml` with:

```yaml
model_name: <model-identifier>   # Model identifier used by the API
api_type: claude | openai        # API protocol
base_url: <url_or_null>          # Endpoint URL (null for provider default)
api_key_env: ENV_VAR_NAME        # Environment variable for API key
supports_vision: true | false    # Whether model accepts images
max_tokens: 2048                 # Max output tokens
temperature: 0                   # Sampling temperature
fix_json: true                   # Apply JSON fix to output
```

No code changes needed for API-compatible models. See `model_configs/README.md` for details.

## Model Deployment

```bash
conda activate stbench

# Remote models (default; model_type comes from pgibench/configs/pgi-alf.yaml)
python -m pgibench.main env=pgi-alf model_name=gpt-5.5

# Local models: serve the model on 127.0.0.1:8000 with any OpenAI-compatible
# server (e.g. vLLM), then run
python -m pgibench.main env=pgi-alf model_name=Qwen3.6-35B-A3B

# Local model server client (Flask-style, model_type=custom; posts to the
# `server_url` env var)
python -m pgibench.main env=pgi-alf model_name=<model> model_type=custom
```
