"""
Pluggable action-mode profiles for the PGI-ALFRED action-display modes.

`action_display_mode` selects one of three parallel display modes:
  - 'full'     : enumerate actions for ALL scene objects
  - 'visible'  : enumerate actions for visible/nearby objects
  - 'command'  : do not enumerate; show action grammar + object catalog, agent emits a command
For 'command', `object_reference_mode` (id|bbox|point|ordinal|type) selects how the
agent refers to a specific object instance.

Each ActionModeProfile bundles everything the planner / env need:
  - system_prompt            (with reference convention injected for command modes)
  - response_format_key      (remote_model looks up the JSON-schema guide by this key)
  - output_kind              ('index' -> int action_id, or 'command' -> command string)
  - needs_image_annotation   (draw bboxes on the planner-facing image)
  - annotate_with_id         (label bboxes with id_2_name IDs, e.g. mug_0)
  - needs_catalog_text       (list an object catalog in the prompt)

Adding a new mode = add a profile here (+ a resolver if it is a new reference scheme).
"""
from dataclasses import dataclass
from typing import Optional, Dict

from pgibench.evaluator.config.system_prompts import (
    alfred_system_prompt, alfred_command_system_prompt,
    get_command_reference_section, get_command_response_fields,
    get_command_response_note)

VALID_DISPLAY_MODES = ('full', 'visible', 'command')
VALID_REFERENCE_MODES = ('id', 'bbox', 'point', 'ordinal', 'type')


@dataclass(frozen=True)
class ActionModeProfile:
    key: str
    action_display_mode: str
    object_reference_mode: Optional[str]
    system_prompt: str
    response_format_key: str        # 'planning' | 'command' | 'eqa'
    output_kind: str                # 'index' | 'command'
    needs_image_annotation: bool
    annotate_with_id: bool
    needs_catalog_text: bool


def _make_command_profile(ref: str) -> ActionModeProfile:
    sys_prompt = alfred_command_system_prompt.replace(
        '{reference_section}', get_command_reference_section(ref))
    # Resolve the per-mode Response Format fields BEFORE the shared fragments
    # ({cleaning_section}/{object_types_section}) are injected at import time,
    # so the literal {x},{y},{z} in those fragments are not mistaken for placeholders.
    sys_prompt = sys_prompt.replace(
        '{response_fields}', get_command_response_fields(ref)).replace(
        '{response_field_note}', get_command_response_note(ref))
    needs_img = ref in ('id', 'ordinal')
    return ActionModeProfile(
        key=f'command_{ref}',
        action_display_mode='command',
        object_reference_mode=ref,
        system_prompt=sys_prompt,
        # Per-reference-mode key (e.g. 'command_bbox') so the JSON Schema sent via
        # response_format only advertises target_bbox for bbox / target_point for
        # point (and neither for id/ordinal/type). See remote_model.GUIDE_REGISTRY.
        response_format_key=f'command_{ref}',
        output_kind='command',
        needs_image_annotation=needs_img,
        annotate_with_id=(ref == 'id'),
        needs_catalog_text=(ref == 'type'),
    )


_FULL = ActionModeProfile(
    key='full', action_display_mode='full', object_reference_mode=None,
    system_prompt=alfred_system_prompt, response_format_key='planning',
    output_kind='index', needs_image_annotation=False,
    annotate_with_id=False, needs_catalog_text=False)

_VISIBLE = ActionModeProfile(
    key='visible', action_display_mode='visible', object_reference_mode=None,
    system_prompt=alfred_system_prompt, response_format_key='planning',
    output_kind='index', needs_image_annotation=False,
    annotate_with_id=False, needs_catalog_text=False)

_COMMAND: Dict[str, ActionModeProfile] = {
    ref: _make_command_profile(ref) for ref in VALID_REFERENCE_MODES
}


def get_profile(action_display_mode: str,
                object_reference_mode: Optional[str] = None) -> ActionModeProfile:
    """Look up the profile for a (display_mode, reference_mode) combination."""
    if action_display_mode == 'full':
        return _FULL
    if action_display_mode == 'visible':
        return _VISIBLE
    if action_display_mode == 'command':
        if object_reference_mode not in VALID_REFERENCE_MODES:
            raise ValueError(
                f"action_display_mode='command' requires object_reference_mode in "
                f"{VALID_REFERENCE_MODES}, got {object_reference_mode!r}")
        return _COMMAND[object_reference_mode]
    raise ValueError(
        f"action_display_mode must be in {VALID_DISPLAY_MODES}, got {action_display_mode!r}")
