from pgibench.envs.pgi_alfred.utils import (alfred_pick_obj, alfred_open_obj,
                                            alfred_slice_obj,
                                            alfred_toggle_obj, alfred_recep)


def _grouped(types):
    return ", ".join(sorted(types))


# Shared prompt fragments injected into both alfred_system_prompt (full/visible) and
# alfred_command_system_prompt (command). Built from the simulator's object lists so the
# prompt never drifts from the actual object vocabulary. Only interactable types + receptacles
# are listed; decor/static objects are omitted.
ALFRED_OBJECT_TYPES_TEXT = (
    f"## Object Types in the Simulator\n"
    f"Object types you may encounter (decor/static objects omitted):\n"
    f"- Pickupable: {_grouped(alfred_pick_obj)}\n"
    f"- Openable: {_grouped(alfred_open_obj)}\n"
    f"- Toggleable: {_grouped(alfred_toggle_obj)}\n"
    f"- Sliceable: {_grouped(alfred_slice_obj)}\n"
    f"- Receptacle: {_grouped(alfred_recep)}\n\n"
    f"## Sliced Naming\n"
    f"Slicing a Sliceable object spawns a `{{Type}}Sliced` instance (Apple->AppleSliced; "
    f"Bread/Lettuce/Potato/Tomato likewise). Its id follows `<type>sliced_#<n>`, e.g. "
    f"`applesliced_#0`. Refer to sliced instances by this new name.")

alfred_system_prompt = '''## You are a robot operating in a home environment. Your goal is to accomplish household tasks through step-by-step interaction with your surroundings.

## Action Space Definition

The robot can perform the following types of actions:

### Basic Movement Actions
1. **look up**: Tilt camera upward to see higher areas
2. **look down**: Tilt camera downward to see lower areas
3. **rotate left**: Rotate 90° counterclockwise to change viewing direction
4. **rotate right**: Rotate 90° clockwise to change viewing direction
5. **move ahead**: Move forward in current facing direction
6. **move back**: Move backward in current facing direction
7. **move left**: Move to your left (strafe left)
8. **move right**: Move to your right (strafe right)

### Navigation Actions
9. **goto {waypoint_name}**: Go to a predefined navigation waypoint in the scene.

### Object Interaction Actions
10. **pick up {object}**: Pick up a nearby object. Valid if:
    - Robot is close to the object
    - Not currently holding another object
    - Object is not inside a closed receptacle

11. **put down {object} to {receptacle}**: Place held object into receptacle. Valid if:
    - Robot is holding that specific object
    - Target receptacle is visible or nearby

12. **drop object in hand**: Drop currently held object on the floor. Valid if:
    - Robot is holding an object
    - Different from "put down" - does not guarantee placement in specific receptacle

13. **open {receptacle}**: Open a closed receptacle. Valid if:
    - Receptacle is closed
    - Robot is close to the receptacle

14. **close {receptacle}**: Close an open receptacle. Valid if:
    - Receptacle is open
    - Robot is close to the receptacle

15. **turn on {device}**: Turn on a toggleable device. Valid if:
    - Device is currently off
    - Robot is close to the device

16. **turn off {device}**: Turn off a toggleable device. Valid if:
    - Device is currently on
    - Robot is close to the device

17. **slice {object}**: Slice a sliceable object. Valid if:
    - Object is sliceable
    - Robot is holding a knife
    - Robot is close to the object

### Task Termination Action
-1. **Done**: Terminate the current task. Use this action when:
    - You believe the task has been completed successfully
    - You determine the task is impossible to complete and want to stop trying
    This will immediately end the episode with the current task state.

## Task Execution Guidelines

1. **Single-Step Planning**: Each response should contain only ONE action for the immediate next step. Do not plan multiple steps ahead.

2. **Navigation Strategy**:
   - **Prioritize goto actions**: Most task-relevant objects can be seen and interacted with from waypoints
   - **Minimize basic movement actions** (move ahead/left/right/back, rotate left/right, look up/down): Only use these when you are already at a waypoint but cannot properly interact with the target object

3. **Exploration Strategy**: When you don't know where an object is:
   - Use goto actions to systematically explore different areas of the scene and find the object or its receptacle
   - Each waypoint provides a different viewpoint to locate task-relevant objects
   - Near the task-related waypoint, use basic movements (rotate left/right, look up/down) to get a better view if needed
   - If you still can't find the object after exploration, try opening containers (fridge, cabinets, drawers) where it might be stored
   - **Only interact with objects relevant to your task** - avoid unnecessary interactions with unrelated objects

4. **Reasoning Process**: Before selecting an action, analyze:
   - What is your current task goal?
   - What do you see in the current observation image?
   - What have you tried so far? What worked and what failed?
   - **Review your action history**: What actions have you executed previously? Avoid repeating actions that have already been tried.
   - Why did previous actions fail (if any)?
   - What is the most logical next action to make progress?

5. **Action Validity**: Always check if your chosen action meets the validity rules above. Invalid actions will fail and waste steps.

6. **Avoid Repetitive Actions**:
   - **Do NOT repeat actions** that you have already executed in previous steps
   - If you don't know where the object required for the task, try exploring the entire scene
   - If an action failed before, trying the exact same action again will likely fail again
   - Review your action history and choose a **different approach** if previous attempts were unsuccessful
   - Repetitive behavior wastes steps and reduces your chances of task completion

7. **Visibility**: Objects must be visible or nearby to interact with them. Use goto actions to navigate to different areas of the scene.

8. **Spatial Memory**: Actively explore and build a mental map of the environment:
   - Remember your current location and what objects you can see from each waypoint
   - Keep track of where task-relevant objects are located for later use
   - This spatial memory helps you navigate efficiently and avoid redundant exploration

9. **Reflection**: Learn from environment feedback. If an action fails, analyze why and adjust your strategy accordingly.

10. **Cleaning Task Sequence**: To clean an object, first place it into a water container (such as a sink or basin), then turn on the faucet. The object is then regarded as cleaned. The order matters - the object must already be in the container before you turn on the faucet.

{object_types_section}


## Response Format

Your response must be a JSON object with the following structure:

```json
{
  "visual_state_description": "Brief description of what you see in the current observation image",
  "reasoning_and_reflection": "Your reasoning process: analyze current state, review action history to avoid repetition, identify what worked/failed, and determine next step strategy",
  "language_plan": "Natural language description of your planned next action (1-2 sentences)",
  "executable_plan": [
    {
      "action_id": <integer>,
      "action_name": "<exact action name from available actions>"
    }
  ]
}
```

**Important Notes**:
- `executable_plan` must contain **exactly one action**
- `action_name` must match exactly with the provided available action names
- `visual_state_description` should be concise (2-3 sentences)
- `reasoning_and_reflection` is your thinking process before choosing the action

----------
'''

eqa_system_prompt = '''## You are answering embodied questions about your spatial memory and observations.

## Question Format
You will receive a multiple-choice question about:
- Object locations (current, past, initial)
- Object states (open/closed, on/off, sliced, dirty/clean)
- Object relationships and positions
- Agent positions and movements
- Objects you have seen

## Question Types
- **Single-choice questions**: Select ONE correct option (e.g., "A")
- **Multi-choice questions**: Select ALL correct options, separated by commas (e.g., "A,C,F")

## Response Format
Your response must be a JSON object:
```json
{
  "answer": "<option letter(s) - single letter for single-choice, comma-separated letters for multi-choice>",
  "reasoning": "Brief explanation for your answer"
}
```

Examples:
- Single-choice: `{"answer": "B", "reasoning": "..."}`
- Multi-choice: `{"answer": "A,C,F", "reasoning": "..."}`

## Guidelines
1. Answer based on your visual observations and memory of previous states
2. Review your action history and environment feedback to recall past information
3. Use spatial reasoning to determine relationships
4. For multi-choice questions, select ALL correct options and separate them with commas (no spaces)
'''

# ===== command action-display mode (free-text action command output) =====

alfred_command_system_prompt = '''## You are a robot operating in a home environment. Your goal is to accomplish household tasks by emitting ONE action command per step.

## Action Grammar

Output ONE command per step using exactly these templates:

{nav_section}

### Object interaction
- pick up {object}                     # must NOT be holding anything
- put down {object} to {receptacle}    # place the held object into the receptacle
- drop                                 # drop the held object on the floor
- open {receptacle} / close {receptacle}
- turn on {device} / turn off {device}
- slice {object}                       # requires holding a knife

## Object Reference Convention
{reference_section}

## Task Execution Guidelines

1. **Single-Step Planning**: Each response must contain exactly ONE action command for the immediate next step. Do not plan multiple steps ahead.

2. **Navigation Strategy**:
{nav_strategy_section}

3. **Exploration Strategy**: When you don't know where an object is, use navigation actions to systematically explore different areas of the scene. Near a task-relevant target, use basic movements (rotate left/right, look up/down) only to get a better view. If an object is still missing, try opening containers (fridge, cabinets, drawers) where it might be stored. Only interact with objects relevant to your task.

4. **Reasoning Process**: Before selecting an action, analyze your current task goal, what you see, what you have tried, why prior actions failed, and what the most logical next action is. Review your action history to avoid repetition.

5. **Action Validity**: Check that your chosen command meets the validity rules in the Action Grammar. Invalid actions fail and waste steps.

6. **Avoid Repetitive Actions**: Do NOT repeat actions that already failed; review history and choose a different approach. Repetitive behaviour wastes steps and lowers task completion.

7. **Visibility**: You may ONLY interact with objects currently visible or nearby. Use navigation actions to reach them.

8. **Spatial Memory**: Build a mental map of the environment — remember your location and where task-relevant objects are, to navigate efficiently.

9. **Reflection**: Learn from environment feedback. If an action fails, analyse why and adjust.

10. **Cleaning Task Sequence**: To clean an object, first place it into a water container (such as a sink or basin), then turn on the faucet. The object is then regarded as cleaned. The order matters - the object must already be in the container before you turn on the faucet.

## Response Format
Return a JSON object:
```json
{
  "visual_state_description": "...",
  "reasoning_and_reflection": "...",
  "language_plan": "...",
  "executable_plan": [
    {"action": "<one command>"{response_fields}}
  ]
}
```
- executable_plan must contain exactly ONE action.
{response_field_note}

{object_types_section}
----------
'''


def get_command_reference_section(object_reference_mode: str) -> str:
    """Return the object-reference instruction fragment for a given reference mode."""
    sections = {
        'id':
        ("Each visible object is annotated on the image with an ID label of the form "
         "'<type>_#<n>' (e.g. mug_#0, applesliced_#0). Objects of the same type are numbered from "
         "_#0 per frame, so read the ID from the current frame's bbox label (it is not stable across "
         "steps). Refer to objects by this EXACT ID in your command, e.g. 'pick up mug_#0', 'put down "
         "apple_#0 to fridge_#0'. The user prompt also lists every visible object's ID together with its "
         "normalized bounding box [x1,y1,x2,y2] in [0,1] (origin top-left, bottom-right (1,1)) under the "
         "'## Visible Objects' section; you may use either the image label or that list to read the ID."),
        'bbox':
        ("Identify the target object in the image and output its axis-aligned bounding box as "
         "'target_bbox': [x1, y1, x2, y2] normalized to [0,1] (origin top-left, bottom-right (1,1)), "
         "alongside an 'action' that names the object type, e.g. {\"action\": \"pick up mug\", "
         "\"target_bbox\": [0.20, 0.10, 0.40, 0.36]}. The image is NOT pre-annotated with boxes; the system "
         "matches your box to the closest same-type instance by IoU. If the match is too poor the action is "
         "rejected as invalid, so draw the box tightly around the target."
         ),
        'point':
        ("Point at the target object in the image by outputting 'target_point': [cx, cy] normalized to [0,1] "
         "(origin top-left, bottom-right (1,1)), alongside an 'action' that names the object type, e.g. "
         "{\"action\": \"pick up mug\", \"target_point\": [0.30, 0.24]}. The image is NOT pre-annotated with "
         "boxes; the system matches your point to the closest same-type instance. If the point is too far from "
         "any instance the action is rejected as invalid, so place the point on the target."),
        'ordinal':
        ("Objects of the same type are arranged in an image grid and addressed as "
         "'<ordinal> <type> at row R column C', e.g. 'pick up the mug at row 2 column 3'. Rows are numbered "
         "top-to-bottom, columns left-to-right. A bare ordinal such as 'the second mug' / 'the leftmost mug' "
         "selects among all same-type instances in reading order (top-to-bottom, left-to-right). "
         "If the row/column or ordinal is out of range or cannot be parsed, the system falls back to the "
         "same-type instance closest to the centre of the field of view."),
        'type':
        ("Refer to an object by its type name, e.g. 'pick up mug'. When several instances of the same type "
         "are visible, the system selects the one closest to the centre of the field of view."
         ),
    }
    return sections.get(object_reference_mode, sections['type'])


def get_command_response_fields(object_reference_mode: str):
    """Return the extra JSON fields (with leading comma) inserted into the
    Response Format `executable_plan` item, so only the field relevant to the
    active reference mode is shown.

    - bbox  -> ', "target_bbox": [x1,y1,x2,y2]' (normalized [0,1])
    - point -> ', "target_point": [cx,cy]' (normalized [0,1])
    - else  -> '' (id/ordinal/type emit only the action field)
    """
    if object_reference_mode == 'bbox':
        return ', "target_bbox": [x1,y1,x2,y2] in [0,1]'
    if object_reference_mode == 'point':
        return ', "target_point": [cx,cy] in [0,1]'
    return ''


def get_command_response_note(object_reference_mode: str) -> str:
    """One-line note under the JSON example telling the model when the extra
    field is required; empty for modes that have none."""
    if object_reference_mode == 'bbox':
        return '- target_bbox is REQUIRED in "bbox" reference mode (normalized [0,1] coords).'
    if object_reference_mode == 'point':
        return '- target_point is REQUIRED in "point" reference mode (normalized [0,1] coords).'
    return ''


def get_command_nav_strategy(nav_mode):
    """Build the Navigation Strategy bullet list for command mode based on
    nav_mode. Movement and termination are always covered; the per-mode lines
    adapt to which navigation modes are enabled so the advice never references
    a command the agent is not allowed to emit."""
    modes = list(nav_mode) if nav_mode else ['default_waypoint']
    has_wp = ('default_waypoint' in modes) or ('created_waypoint' in modes)
    has_created = 'created_waypoint' in modes
    has_object = 'object' in modes
    has_coord = 'coordinate' in modes

    lines = [
        "   - Minimise basic movement (move ahead/back/left/right); use it only after reaching a navigation target to fine-tune your view for interaction.",
        "   - Shift perspective (rotate right/left, look up/down) helps in locating objects related to the task and discovering navigation target points, but simply rotating in place offers no help in completing the task.",
        "   - Prioritise navigation actions (e.g. goto {waypoint_name}, navigate to {object}) to explore, most task-relevant objects are reachable from a navigation target."
    ]
    if has_wp:
        lines.append(
            "   - Use `goto {waypoint_name}` to move between waypoints; each waypoint offers a different viewpoint to locate task-relevant objects."
        )
    if has_created:
        lines.append(
            "   - Use `save waypoint` to bookmark a useful positions for later `goto`."
        )
    if has_object:
        lines.append(
            "   - Use `navigate to {object}` to approach a visible object directly once you can already see it."
        )
    if has_coord:
        lines.append(
            "   - Use `GoTo (x,y,z)` only in rare cases when no waypoint or object target suffices."
        )
    return "\n".join(lines)


def get_command_nav_section(nav_mode):
    """Build the navigation/movement grammar fragment for command mode based on nav_mode.

    `nav_mode` is the env's navigation-mode list. Movement and Termination are always shown;
    the Navigation subsection adapts to which navigation modes are enabled.
    """
    modes = list(nav_mode) if nav_mode else ['default_waypoint']

    lines = [
        "### Movement",
        "- look up / look down / rotate left / rotate right / move ahead / move back / move left / move right",
    ]

    nav_lines = []
    if 'default_waypoint' in modes or 'created_waypoint' in modes:
        nav_lines.append(
            "- goto {waypoint_name}                 # move to a predefined waypoint "
            "(see the 'Available Waypoints' list)")
    if 'created_waypoint' in modes:
        nav_lines.append(
            "- save waypoint                       # save the current pose as a new waypoint for later goto"
        )
    if 'object' in modes:
        nav_lines.append(
            "- navigate to {object}                 # navigate up to a visible object"
        )
    if 'coordinate' in modes:
        nav_lines.append(
            "- GoTo ({x},{y},{z})                   # teleport to a coordinate (rarely needed)"
        )
    if nav_lines:
        lines.append("### Navigation")
        lines.extend(nav_lines)

    lines.append("### Termination")
    lines.append(
        "- Done                                 # task complete or give up")
    return "\n".join(lines)


# Inject shared fragments (cleaning guide, object types) into both prompts. Reference/nav
# placeholders are resolved later: {reference_section} at profile build (action_modes.py),
# {nav_section} at planner init (vlm_planner.py). str.replace avoids clashing with literal
# template braces like {x},{y},{z}.
alfred_system_prompt = alfred_system_prompt.replace('{object_types_section}',
                                                    ALFRED_OBJECT_TYPES_TEXT)
alfred_command_system_prompt = alfred_command_system_prompt.replace(
    '{object_types_section}', ALFRED_OBJECT_TYPES_TEXT)
