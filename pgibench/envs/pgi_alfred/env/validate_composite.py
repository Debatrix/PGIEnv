#!/usr/bin/env python3
"""
Static validation for composite task YAML configs.

Checks YAML schema, object/receptacle/action legality, room compatibility, and
goal-state conflicts BEFORE any trajectory matching or simulator run. This is the
fast, deterministic gate in stage 2 of the compose-alfred-task skill.

Run:
    python -m pgibench.envs.pgi_alfred.env.validate_composite --config <yaml>
"""

import argparse
from typing import Dict, List, Any, Tuple

from pgibench.envs.pgi_alfred.gen import constants as C


# Required pddl params per task type. Mirrors ALFRED task definitions.
TASK_TYPE_PARAMS = {
    'pick_and_place_simple': ['object', 'parent'],
    'pick_two_obj_and_place': ['object', 'parent'],
    'look_at_obj_in_light': ['object', 'toggle'],
    'pick_clean_then_place_in_recep': ['object', 'parent'],
    'pick_heat_then_place_in_recep': ['object', 'parent'],
    'pick_cool_then_place_in_recep': ['object', 'parent'],
    'pick_slice_then_place_in_recep': ['object', 'parent'],
    'pick_and_place_with_movable_recep': ['object', 'parent', 'movable_recep'],
    # Atomic subtask types (compose freely with the ALFRED types above).
    'toggle_object_on': ['toggle'],
    'toggle_object_off': ['toggle'],
    'open_receptacle': ['parent'],
    'close_receptacle': ['parent'],
    'slice_in_place': ['object'],
}

# Rooms where each task type is valid.
TASK_VALID_ROOMS = {
    "pick_and_place_simple": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
    "pick_two_obj_and_place": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
    "look_at_obj_in_light": {"LivingRoom", "Bedroom"},
    "pick_clean_then_place_in_recep": {"Kitchen", "Bathroom"},
    "pick_heat_then_place_in_recep": {"Kitchen"},
    "pick_cool_then_place_in_recep": {"Kitchen"},
    "pick_slice_then_place_in_recep": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
    "pick_and_place_with_movable_recep": {"Kitchen", "LivingRoom", "Bedroom"},
    # Atomic subtask types are valid in every room.
    "toggle_object_on": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
    "toggle_object_off": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
    "open_receptacle": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
    "close_receptacle": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
    "slice_in_place": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
}

# Scene number ranges per room.
ROOM_SCENES = {
    "Kitchen": list(range(1, 31)),
    "LivingRoom": list(range(201, 231)),
    "Bedroom": list(range(301, 331)),
    "Bathroom": list(range(401, 431)),
}

# Action type -> set of objects that action applies to (from gen/constants.py).
ACTION_OBJECT_SETS = {
    'heat': C.VAL_ACTION_OBJECTS['Heatable'],
    'cool': C.VAL_ACTION_OBJECTS['Coolable'],
    'clean': C.VAL_ACTION_OBJECTS['Cleanable'],
    'slice': C.VAL_ACTION_OBJECTS['Sliceable'],
}
TASK_TO_ACTION = {
    'pick_heat_then_place_in_recep': 'heat',
    'pick_cool_then_place_in_recep': 'cool',
    'pick_clean_then_place_in_recep': 'clean',
}

VALID_OBJECTS = set(C.OBJ_PARENTS.keys())
TOGGLEABLE = C.VAL_ACTION_OBJECTS['Toggleable']
MOVABLE_RECEPTACLES = set(C.MOVABLE_RECEPTACLES)
OPENABLE = getattr(C, 'OPENABLE_CLASS_SET', set(getattr(C, 'OPENABLE_CLASS_LIST', [])))


def _obj_in_recep(obj: str, recep: str) -> bool:
    """Whether `obj` can be placed into `recep` per VAL_RECEPTACLE_OBJECTS."""
    objs = C.VAL_RECEPTACLE_OBJECTS.get(recep)
    return bool(objs) and obj in objs


def validate_subtask(st: Dict[str, Any], room_type: str, idx: int) -> Tuple[List[str], List[str]]:
    """Validate one subtask. Returns (errors, warnings)."""
    errors, warnings = [], []
    prefix = f"subtask[{idx}]"

    task_type = st.get('task_type')
    if task_type not in TASK_TYPE_PARAMS:
        errors.append(f"{prefix}: invalid task_type '{task_type}'. "
                      f"Must be one of {list(TASK_TYPE_PARAMS)}")
        return errors, warnings

    if room_type not in TASK_VALID_ROOMS.get(task_type, set()):
        errors.append(f"{prefix}: '{task_type}' is not valid in room '{room_type}'. "
                      f"Valid rooms: {sorted(TASK_VALID_ROOMS[task_type])}")
        return errors, warnings

    # Required params present.
    for param in TASK_TYPE_PARAMS[task_type]:
        if not st.get(param):
            errors.append(f"{prefix}: missing required param '{param}' for '{task_type}'")

    obj = st.get('object', '')
    if obj and obj not in VALID_OBJECTS:
        errors.append(f"{prefix}: object '{obj}' is not a valid ALFRED object type")

    # sliced requires a sliceable object.
    if st.get('sliced') and obj and obj not in ACTION_OBJECT_SETS['slice']:
        errors.append(f"{prefix}: sliced=True but '{obj}' is not sliceable "
                      f"(sliceable: {sorted(ACTION_OBJECT_SETS['slice'])})")

    # Action-object compatibility (heat/cool/clean).
    action_key = TASK_TO_ACTION.get(task_type)
    if action_key and obj and obj not in ACTION_OBJECT_SETS[action_key]:
        errors.append(f"{prefix}: '{obj}' is not {action_key}able for '{task_type}'")

    # toggle-based tasks (look_at_obj_in_light, toggle_object_on/off): toggle must be a lamp.
    if task_type in ('look_at_obj_in_light', 'toggle_object_on', 'toggle_object_off'):
        toggle = st.get('toggle', '')
        if toggle and toggle not in TOGGLEABLE:
            errors.append(f"{prefix}: toggle '{toggle}' is not a valid toggleable "
                          f"(toggleable: {sorted(TOGGLEABLE)})")

    # parent receptacle compatibility (all tasks that place into a parent).
    parent = st.get('parent', '')
    if parent and task_type != 'look_at_obj_in_light':
        if parent not in C.VAL_RECEPTACLE_OBJECTS:
            errors.append(f"{prefix}: parent '{parent}' is not a valid receptacle")
        elif obj and not _obj_in_recep(obj, parent):
            warnings.append(f"{prefix}: object '{obj}' is not listed as placeable "
                            f"into receptacle '{parent}' (may still be placeable in sim)")

    # movable receptacle.
    mrecep = st.get('movable_recep', '')
    if mrecep and mrecep not in MOVABLE_RECEPTACLES:
        errors.append(f"{prefix}: movable_recep '{mrecep}' is not in "
                      f"MOVABLE_RECEPTACLES {sorted(MOVABLE_RECEPTACLES)}")

    # open/close_receptacle: parent must be an openable receptacle.
    if task_type in ('open_receptacle', 'close_receptacle') and parent and parent not in OPENABLE:
        errors.append(f"{prefix}: parent '{parent}' is not openable "
                      f"(openable: {sorted(OPENABLE)})")

    # slice_in_place / pick_slice_then_place_in_recep: object must be sliceable.
    if task_type in ('slice_in_place', 'pick_slice_then_place_in_recep') \
            and obj and obj not in ACTION_OBJECT_SETS['slice']:
        errors.append(f"{prefix}: object '{obj}' is not sliceable "
                      f"(sliceable: {sorted(ACTION_OBJECT_SETS['slice'])})")

    return errors, warnings


def _goal_conflicts(subtasks: List[Dict[str, Any]]) -> List[str]:
    """Warn if the same object is targeted by multiple place subtasks with conflicting
    final receptacles. CompositeTask.goal_satisfied requires ALL subtasks satisfied in the
    FINAL state, so a conflict makes the task unsatisfiable."""
    warnings = []
    final_recep_of = {}
    for i, st in enumerate(subtasks):
        obj = st.get('object')
        parent = st.get('parent')
        if not obj or not parent:
            continue
        prev = final_recep_of.get(obj)
        if prev is not None and prev != parent:
            warnings.append(
                f"goal conflict: object '{obj}' placed into '{prev}' (subtask) "
                f"and '{parent}' (subtask[{i}]) — both must hold in final state, "
                f"task is likely unsatisfiable")
        else:
            final_recep_of[obj] = parent
    return warnings


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Validate a composite task config dict. Returns (ok, errors, warnings)."""
    errors, warnings = [], []

    for field in ('name', 'scene', 'instruction', 'subtasks'):
        if field not in config:
            errors.append(f"missing required field: {field}")

    if errors:
        return False, errors, warnings

    scene = config['scene']
    room_type = scene.get('room_type')
    if room_type not in ROOM_SCENES:
        errors.append(f"invalid room_type '{room_type}'. "
                      f"Must be one of {list(ROOM_SCENES)}")

    scene_num = scene.get('scene_num')
    if scene_num is not None and room_type in ROOM_SCENES:
        if scene_num not in ROOM_SCENES[room_type]:
            errors.append(f"scene_num {scene_num} not in {room_type} range "
                          f"{ROOM_SCENES[room_type][0]}-{ROOM_SCENES[room_type][-1]}")

    subtasks = config['subtasks']
    if not subtasks:
        errors.append("at least one subtask is required")

    if errors:
        return False, errors, warnings

    for i, st in enumerate(subtasks):
        e, w = validate_subtask(st, room_type, i)
        errors += e
        warnings += w

    if not config.get('instruction'):
        errors.append("instruction must be a non-empty template string")

    warnings += _goal_conflicts(subtasks)

    return (len(errors) == 0), errors, warnings


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Validate composite task YAML')
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    ok, errors, warnings = validate_config(config)

    name = config.get('name', args.config)
    if errors:
        print(f"[FAIL] {name}")
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print(f"[PASS] {name}")
    for w in warnings:
        print(f"  WARN:  {w}")

    return 0 if ok else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
