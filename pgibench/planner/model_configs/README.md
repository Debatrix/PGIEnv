# Model Configuration Guide

This directory contains YAML configuration files for various LLM/MLLM models used in PGIBench.

## Configuration File Format

Each model configuration file is a YAML file with the following structure:

```yaml
model_name: <model_identifier>          # Model identifier used by the API
api_type: <claude|openai>                # API type: 'claude' for Anthropic, 'openai' for OpenAI-compatible
base_url: <url_or_null>                  # Base URL for API (null for default)
api_key_env: <environment_variable>      # Environment variable name containing API key
supports_vision: <true|false>            # Whether the model supports vision input
max_tokens: <number>                     # Maximum tokens for model output
temperature: <number>                    # Temperature for sampling (0 for deterministic)
fix_json: <true|false>                   # Optional: Apply JSON fix to output (default: false)
```

## API Types

### Claude API
- Use `api_type: claude`
- Requires Anthropic SDK
- Only for Claude models
- `base_url` should be `null`

### OpenAI-Compatible API
- Use `api_type: openai`
- Uses OpenAI SDK
- Works with:
  - OpenAI GPT models
  - Qwen (Alibaba Cloud DashScope)
  - Gemini (Google OpenAI-compatible endpoint)
  - Any other OpenAI-compatible API
- Set `base_url` to the API endpoint (or `null` for OpenAI default)

## Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
# OpenAI GPT Models
OPENAI_API_KEY=sk-your-openai-api-key-here

# Custom Deployment Models
CUSTOM_API_KEY=your-custom-api-key-here
```

A template file `.env.template` is provided in the project root.

## Adding New Models

To add a new model:

1. Create a new YAML file in this directory (e.g., `new-model.yaml`)
2. Fill in the required fields following the format above
3. Add the corresponding API key to your `.env` file
4. Use the model by specifying its filename (without .yaml) when initializing RemoteModel

Example:

```python
from pgibench.planner.remote_model import RemoteModel

# Using model name (auto-searches in this directory)
model = RemoteModel('gpt-4o', language_only=False)

# Or using full path
model = RemoteModel('pgibench/planner/model_configs/gpt-4o.yaml', language_only=False)
```

## Notes

- Configuration files do **not** support environment variable interpolation (e.g., `${REMOTE_URL}`)
- All configuration values must be hardcoded in the YAML file
- For custom deployments, specify the complete URL in `base_url`
- Some models (like Qwen) may require `fix_json: true` due to output formatting issues
