vlm_generation_guide={
    "type": "object",
    'properties': {
        "visual_state_description": {
            "type": "string",
            "description": "Brief description of what you see in the current observation image (2-3 sentences)",
        },
        "reasoning_and_reflection": {
            "type": "string",
            "description": "Your reasoning process: analyze current state, history, failures, and next step strategy (3-5 sentences)",
        },
        "language_plan": {
            "type": "string",
            "description": "Natural language description of your planned next action (1-2 sentences)",
        },
        "executable_plan": {
            "type": "array",
            "description": "A list containing exactly ONE action with action ID and name. Must not be empty.",
            "items": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "integer",
                        "description": "The action ID to select from the available actions (0 ~ max_id)",
                    },
                    "action_name": {
                        "type": "string",
                        "description": "The exact action name from the available actions list",
                    }
                },
                "required": ["action_id", "action_name"],
            },
            "minItems": 1,
            "maxItems": 1  # Force exactly one action
        }
    },
    "required": ["visual_state_description", "reasoning_and_reflection", "language_plan", "executable_plan"]
}

llm_generation_guide={
    "type": "object",
    'properties': {
        "reasoning_and_reflection": {
            "type": "string",
            "description": "summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task",
        },
        "language_plan": {
            "type": "string",
            "description": "The list of actions to achieve the user instruction. Each action is started by the step number and the action name",
        },
        "executable_plan": {
            "type": "array",
            "description": "A list of actions needed to achieve the user instruction, with each action having an action ID and a name. List length should not exceed 20.",
            "items": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "integer",
                        "description": "The action ID to select from the available actions given by the prompt",
                    },
                    "action_name": {
                        "type": "string",
                        "description": "The name of the action",
                    }
                },
                "required": ["action_id", "action_name"]
            }
        },
    },
    "required": ["reasoning_and_reflection", "language_plan", "executable_plan"]
}

eqa_generation_guide = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The option letter (A, B, C, or D) that is the correct answer",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation for why this answer is correct",
        }
    },
    "required": ["answer", "reasoning"]
}


# ----- command action-display mode guides (free-text action command output) -----
#
# The executable_plan item schema varies by object_reference_mode so the JSON
# Schema sent via response_format exposes ONLY the field the active mode needs:
#   'bbox'  -> adds target_bbox (and requires it)
#   'point' -> adds target_point (and requires it)
#   id/ordinal/type -> action only
# This keeps the schema consistent with the per-mode text in the system prompt's
# Response Format section, avoiding confusion from advertising unused fields.


def _bbox_field():
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "[x1,y1,x2,y2] bounding box of the target object, normalized to [0,1] (origin top-left, bottom-right (1,1)). REQUIRED when object_reference_mode is 'bbox'.",
    }


def _point_field():
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 2,
        "description": "[cx,cy] point on the target object, normalized to [0,1] (origin top-left, bottom-right (1,1)). REQUIRED when object_reference_mode is 'point'.",
    }


def get_command_generation_guide(object_reference_mode):
    """Build the VLM command-mode JSON schema for a given object_reference_mode.

    Only the field relevant to the active reference mode is declared in the
    executable_plan item schema (and required when applicable).
    """
    item_properties = {
        "action": {
            "type": "string",
            "description": "One action command following the action grammar, e.g. 'pick up mug_0', 'put down plate in sink', 'GoTo (1.5,0,2.3)'.",
        },
    }
    required = ["action"]
    if object_reference_mode == 'bbox':
        item_properties["target_bbox"] = _bbox_field()
        required.append("target_bbox")
    elif object_reference_mode == 'point':
        item_properties["target_point"] = _point_field()
        required.append("target_point")
    return {
        "type": "object",
        'properties': {
            "visual_state_description": {
                "type": "string",
                "description": "Brief description of what you see in the current observation image (2-3 sentences)",
            },
            "reasoning_and_reflection": {
                "type": "string",
                "description": "Your reasoning process: analyze current state, history, failures, and next step strategy (3-5 sentences)",
            },
            "language_plan": {
                "type": "string",
                "description": "Natural language description of your planned next action (1-2 sentences)",
            },
            "executable_plan": {
                "type": "array",
                "description": "A list containing exactly ONE action command string. Must not be empty.",
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": required,
                },
                "minItems": 1,
                "maxItems": 1,
            },
        },
        "required": ["visual_state_description", "reasoning_and_reflection", "language_plan", "executable_plan"],
    }


llm_command_generation_guide = {
    "type": "object",
    'properties': {
        "reasoning_and_reflection": {
            "type": "string",
            "description": "summarize the history of interactions and any available environmental feedback, and reason about the next action",
        },
        "language_plan": {
            "type": "string",
            "description": "Natural language description of your planned next action",
        },
        "executable_plan": {
            "type": "array",
            "description": "A list containing exactly ONE action command string.",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "One action command following the action grammar.",
                    },
                },
                "required": ["action"],
            },
            "minItems": 1,
            "maxItems": 1,
        },
    },
    "required": ["reasoning_and_reflection", "language_plan", "executable_plan"],
}

