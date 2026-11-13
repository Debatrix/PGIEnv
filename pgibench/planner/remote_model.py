import os
import yaml
import anthropic
from openai import OpenAI
from pgibench.planner.planner_config.generation_guide import (
    llm_generation_guide, vlm_generation_guide, eqa_generation_guide,
    get_command_generation_guide, llm_command_generation_guide)
from pgibench.planner.planner_utils import convert_format_2claude, fix_json

# response_format key -> (vlm_schema, llm_schema). Looked up by _call_openai_compatible.
# Add new guides here; action-mode profiles reference them by key
# (see evaluator/config/action_modes.py ActionModeProfile.response_format_key).
#
# command mode keys are per-reference-mode ('command_id', 'command_bbox',
# 'command_point', 'command_ordinal', 'command_type') so the JSON Schema only
# advertises the field the active reference mode consumes (target_bbox only for
# bbox, target_point only for point; none for id/ordinal/type). The shared
# 'command' key is kept as a safe default (action-only schema).
_GUIDE_REGISTRY = {
    'planning': (vlm_generation_guide, llm_generation_guide),
    'command': (get_command_generation_guide(None), llm_command_generation_guide),
    'eqa': (eqa_generation_guide, None),
}
# register one command entry per reference mode
for _ref in ('id', 'bbox', 'point', 'ordinal', 'type'):
    _GUIDE_REGISTRY[f'command_{_ref}'] = (
        get_command_generation_guide(_ref), llm_command_generation_guide)
GUIDE_REGISTRY = _GUIDE_REGISTRY


class RemoteModel:
    """
    Unified model interface supporting Claude API and OpenAI-compatible APIs.

    This class loads model configuration from YAML files and provides a unified
    interface for calling different LLM/MLLM providers.

    Supported API types:
    - 'claude': Uses Anthropic SDK (only for Claude models)
    - 'openai': Uses OpenAI SDK (for GPT, Qwen, Gemini, and any OpenAI-compatible API)
    """

    def __init__(self, model_config_path, language_only=False):
        """
        Initialize RemoteModel with configuration from YAML file.

        Args:
            model_config_path: Path to YAML config file or model name (e.g., 'gpt-4o')
                              If model name is provided, it will auto-search in
                              pgibench/planner/model_configs/
            language_only: If True, disable visual input (text-only mode)
        """
        self.config = self._load_config(model_config_path)
        self.language_only = language_only
        self.client = self._init_client()
        self.message_formatter = self._get_message_formatter()

    def _load_config(self, model_config_path):
        """Load model configuration from YAML file."""
        # If model name is provided (not a path), auto-search in model_configs directory
        if not model_config_path.endswith('.yaml'):
            config_dir = os.path.join(os.path.dirname(__file__),
                                      'model_configs')
            model_config_path = os.path.join(config_dir,
                                             f'{model_config_path}.yaml')

        # Load YAML file
        with open(model_config_path) as f:
            return yaml.safe_load(f)

    def _init_client(self):
        """
        Initialize API client based on api_type.

        Supports two API types:
        - 'claude': Anthropic SDK
        - 'openai': OpenAI SDK (including all OpenAI-compatible APIs)
        """
        if self.config['api_type'] == 'claude':
            return anthropic.Anthropic(
                api_key=os.environ.get(self.config['api_key_env']))
        else:  # openai compatible (GPT, Qwen, Gemini, etc.)
            return OpenAI(
                api_key=os.environ.get(self.config['api_key_env']),
                base_url=self.config.get('base_url')  # None for default
            )

    def _get_message_formatter(self):
        """
        Get message formatting function.

        Only Claude requires special format conversion in multimodal mode.
        Other models (OpenAI, Qwen, Gemini) use standard OpenAI format.
        """
        if self.config['api_type'] == 'claude' and not self.language_only:
            return convert_format_2claude
        return lambda x: x  # No conversion needed for OpenAI-compatible APIs

    def respond(self, message_history: list, response_format=None):
        """
        Unified response entry point.

        Args:
            message_history: List of messages in conversation
            response_format: Optional dict specifying JSON schema. If None, uses default action planning schema.

        Routes to appropriate API call method based on api_type.
        """
        # Format messages if needed
        formatted_messages = self.message_formatter(message_history)

        # Route based on api_type
        if self.config['api_type'] == 'claude':
            return self._call_claude(formatted_messages)
        else:
            return self._call_openai_compatible(formatted_messages,
                                                response_format)

    def _call_claude(self, message_history):
        """Call Claude API using Anthropic SDK."""
        response = self.client.messages.create(
            model=self.config['model_name'],
            max_tokens=self.config['max_tokens'],
            temperature=self.config['temperature'],
            messages=message_history)
        return response.content[0].text

    def _call_openai_compatible(self, message_history, response_format=None):
        """
        Call OpenAI-compatible API (unified method).

        Args:
            message_history: List of messages
            response_format: Optional dict specifying JSON schema. If None, uses default action planning schema.

        This method works for:
        - OpenAI GPT series
        - Qwen (Alibaba Cloud DashScope)
        - Gemini (Google OpenAI-compatible endpoint)
        - Any other OpenAI-compatible API
        """
        guide_key = response_format if response_format is not None else 'planning'
        entry = GUIDE_REGISTRY.get(guide_key)
        if entry is None:
            response = self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=message_history,
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'])
        else:
            vlm_guide, llm_guide = entry
            schema = llm_guide if (self.language_only
                                   and llm_guide is not None) else vlm_guide
            response = self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=message_history,
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
                response_format=dict(type='json_schema',
                                     json_schema=dict(
                                         name=guide_key, schema=schema)))

        out = response.choices[0].message.content

        # Apply fix_json for certain models if needed
        if self.config.get('fix_json', False):
            out = fix_json(out)

        return out
