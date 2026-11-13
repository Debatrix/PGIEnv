"""
Object Relative Position Question Generators

Generates single-choice questions about the relative position (cardinal direction)
of one object to another at different temporal points.
"""

import random
from typing import Dict, List, Optional, Any
from types import ModuleType

from pgibench.envs.pgi_alfred.eqa.thor_tracker import ThorTracker
from pgibench.envs.pgi_alfred.eqa.utils import (
    get_options_list, has_sequential_number, format_question_metadata,
    create_question_dict, get_object_loc_desc, calculate_relative_direction,
    is_top_level_object)

# Define angle ranges for each direction (0 = North, clockwise)
direction_angle_ranges = {
    "North": "[315, 360) + [0, 45)",
    "East": "[45, 135)",
    "South": "[135, 225)",
    "West": "[225, 315)"
}


# Decorator to mark functions as EQA generators
def eqa_generator(question_id: str, question_type: str, templates: List[str],
                  description: str):
    """Decorator to mark a function as an EQA question generator."""

    def decorator(func):
        func.is_eqa_generator = True
        func.question_id = question_id
        func.question_type = question_type
        func.templates = templates
        func.description = description
        return func

    return decorator


def _get_valid_objects(tracker: ThorTracker, timestep: int) -> List[tuple]:
    """
    Get objects valid for relative position questions.

    Args:
        tracker: ThorTracker instance
        timestep: Timestep to query

    Returns:
        List of (object_id, object_type, locDesc) tuples
    """
    valid_objects = []
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if state and is_top_level_object(
                state) and obj_id in tracker.historically_seen_objects:
            obj_loc_desc = get_object_loc_desc(state)
            if obj_loc_desc and not has_sequential_number(obj_loc_desc):
                valid_objects.append((obj_id, state.object_type, obj_loc_desc))

    return valid_objects


@eqa_generator(
    question_id="obj_rel_pos_initial",
    question_type="single_choice",
    templates=[
        "At the start, what was the position of {obj_B} relative to {obj_A}?",
        "Initially, was {obj_B} to the east, south, west, or north of {obj_A}?",
        "When the task began, in which direction of {obj_A} was {obj_B}?",
        "What was the relative direction of {obj_B} to {obj_A} at the initial moment?"
    ],
    description="Relative position between two objects at t=0.")
def generate_obj_rel_pos_initial(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate relative position question at t=0."""
    target_timestep = 0
    current_timestep = tracker.current_timestep

    # Get valid objects
    valid_objects = _get_valid_objects(tracker, target_timestep)

    if len(valid_objects) < 2:
        return None

    # Select obj_A randomly
    obj_a_id, obj_a_type, obj_a_loc_desc = rng.choice(valid_objects)

    # Filter objects where locDesc doesn't contain obj_a's objectType
    filtered_objects = [
        (oid, otype, locdesc) for oid, otype, locdesc in valid_objects
        if obj_a_type.lower() not in locdesc.lower() and oid != obj_a_id
    ]

    if not filtered_objects:
        return None

    # Select obj_B
    obj_b_id, obj_b_type, obj_b_loc_desc = rng.choice(filtered_objects)

    # Get positions
    obj_a_state = tracker.get_object_state_at_step(obj_a_id, target_timestep)
    obj_b_state = tracker.get_object_state_at_step(obj_b_id, target_timestep)

    if not obj_a_state or not obj_b_state:
        return None

    # Calculate direction and angle
    direction, angle_deg = calculate_relative_direction(
        obj_a_state.position, obj_b_state.position)

    # Create options (4 cardinal directions)
    all_directions = ["East", "South", "West", "North"]
    rng.shuffle(all_directions)
    options = get_options_list(all_directions)

    # Find correct answer label
    correct_answer = None
    for label, dir_name in options.items():
        if dir_name == direction:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_rel_pos_initial.templates)
    question_text = template.format(obj_B=obj_b_loc_desc, obj_A=obj_a_loc_desc)

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "reference_object": {
            "objectId": obj_a_id,
            "position": obj_a_state.position
        },
        "target_object": {
            "objectId": obj_b_id,
            "position": obj_b_state.position
        },
        "relative_info": {
            "from_objA_to_objB_angle": angle_deg,
            "calculated_direction": direction
        },
        "options_info": {}
    }

    for label, dir_name in options.items():
        evidence["options_info"][label] = {
            "direction": dir_name,
            "angle_range": direction_angle_ranges[dir_name],
            "is_correct": (dir_name == direction)
        }

    # Create metadata
    metadata = format_question_metadata(question_id="obj_rel_pos_initial",
                                        question_type="single_choice",
                                        question_timestep=target_timestep,
                                        current_timestep=current_timestep,
                                        template_used=template)

    return create_question_dict(question_id="",
                                question_text=question_text,
                                options=options,
                                correct_answer=correct_answer,
                                metadata=metadata,
                                evidence=evidence)


@eqa_generator(
    question_id="obj_rel_pos_current",
    question_type="single_choice",
    templates=[
        "Currently, what is the position of {obj_B} relative to {obj_A}?",
        "Is {obj_B} to the east, south, west, or north of {obj_A} right now?",
        "At this moment, from the perspective of {obj_A}, where is {obj_B}?",
        "In which direction of {obj_A} is {obj_B} currently located?"
    ],
    description="Relative position between two objects at current moment.")
def generate_obj_rel_pos_current(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate relative position question at current moment."""
    target_timestep = tracker.current_timestep
    current_timestep = tracker.current_timestep

    # Get valid objects
    valid_objects = _get_valid_objects(tracker, target_timestep)

    if len(valid_objects) < 2:
        return None

    # Select obj_A randomly
    obj_a_id, obj_a_type, obj_a_loc_desc = rng.choice(valid_objects)

    # Filter objects where locDesc doesn't contain obj_a's objectType
    filtered_objects = [
        (oid, otype, locdesc) for oid, otype, locdesc in valid_objects
        if obj_a_type.lower() not in locdesc.lower() and oid != obj_a_id
    ]

    if not filtered_objects:
        return None

    # Select obj_B
    obj_b_id, obj_b_type, obj_b_loc_desc = rng.choice(filtered_objects)

    # Get positions
    obj_a_state = tracker.get_object_state_at_step(obj_a_id, target_timestep)
    obj_b_state = tracker.get_object_state_at_step(obj_b_id, target_timestep)

    if not obj_a_state or not obj_b_state:
        return None

    # Calculate direction and angle
    direction, angle_deg = calculate_relative_direction(
        obj_a_state.position, obj_b_state.position)

    # Create options
    all_directions = ["East", "South", "West", "North"]
    rng.shuffle(all_directions)
    options = get_options_list(all_directions)

    # Find correct answer label
    correct_answer = None
    for label, dir_name in options.items():
        if dir_name == direction:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_rel_pos_current.templates)
    question_text = template.format(obj_B=obj_b_loc_desc, obj_A=obj_a_loc_desc)

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "reference_object": {
            "objectId": obj_a_id,
            "position": obj_a_state.position
        },
        "target_object": {
            "objectId": obj_b_id,
            "position": obj_b_state.position
        },
        "relative_info": {
            "from_objA_to_objB_angle": angle_deg,
            "calculated_direction": direction
        },
        "options_info": {}
    }

    for label, dir_name in options.items():
        evidence["options_info"][label] = {
            "direction": dir_name,
            "angle_range": direction_angle_ranges[dir_name],
            "is_correct": (dir_name == direction)
        }

    # Create metadata
    metadata = format_question_metadata(question_id="obj_rel_pos_current",
                                        question_type="single_choice",
                                        question_timestep=target_timestep,
                                        current_timestep=current_timestep,
                                        template_used=template)

    return create_question_dict(question_id="",
                                question_text=question_text,
                                options=options,
                                correct_answer=correct_answer,
                                metadata=metadata,
                                evidence=evidence)


@eqa_generator(
    question_id="obj_rel_pos_n_steps_ago",
    question_type="single_choice",
    templates=[
        "{n} steps ago, what was the position of {obj_B} relative to {obj_A}?",
        "From the perspective of {obj_A}, where was {obj_B} {n} steps back?",
        "In which direction of {obj_A} was {obj_B} located {n} steps ago?",
        "What was the relative direction of {obj_B} to {obj_A} {n} steps before now?"
    ],
    description="Relative position between two objects n steps ago.")
def generate_obj_rel_pos_n_steps_ago(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate relative position question n steps ago."""
    current_timestep = tracker.current_timestep

    # Need at least 2 timesteps of history
    if current_timestep < 2:
        return None

    # Randomly select n (1 to current_timestep - 1)
    n = rng.randint(1, current_timestep - 1)
    target_timestep = current_timestep - n

    # Get valid objects
    valid_objects = _get_valid_objects(tracker, target_timestep)

    if len(valid_objects) < 2:
        return None

    # Select obj_A randomly
    obj_a_id, obj_a_type, obj_a_loc_desc = rng.choice(valid_objects)

    # Filter objects where locDesc doesn't contain obj_a's objectType
    filtered_objects = [
        (oid, otype, locdesc) for oid, otype, locdesc in valid_objects
        if obj_a_type.lower() not in locdesc.lower() and oid != obj_a_id
    ]

    if not filtered_objects:
        return None

    # Select obj_B
    obj_b_id, obj_b_type, obj_b_loc_desc = rng.choice(filtered_objects)

    # Get positions
    obj_a_state = tracker.get_object_state_at_step(obj_a_id, target_timestep)
    obj_b_state = tracker.get_object_state_at_step(obj_b_id, target_timestep)

    if not obj_a_state or not obj_b_state:
        return None

    # Calculate direction and angle
    direction, angle_deg = calculate_relative_direction(
        obj_a_state.position, obj_b_state.position)

    # Create options
    all_directions = ["East", "South", "West", "North"]
    rng.shuffle(all_directions)
    options = get_options_list(all_directions)

    # Find correct answer label
    correct_answer = None
    for label, dir_name in options.items():
        if dir_name == direction:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_rel_pos_n_steps_ago.templates)
    question_text = template.format(n=n,
                                    obj_B=obj_b_loc_desc,
                                    obj_A=obj_a_loc_desc)

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "reference_object": {
            "objectId": obj_a_id,
            "position": obj_a_state.position
        },
        "target_object": {
            "objectId": obj_b_id,
            "position": obj_b_state.position
        },
        "relative_info": {
            "from_objA_to_objB_angle": angle_deg,
            "calculated_direction": direction
        },
        "options_info": {}
    }

    for label, dir_name in options.items():
        evidence["options_info"][label] = {
            "direction": dir_name,
            "angle_range": direction_angle_ranges[dir_name],
            "is_correct": (dir_name == direction)
        }

    # Create metadata
    metadata = format_question_metadata(question_id="obj_rel_pos_n_steps_ago",
                                        question_type="single_choice",
                                        question_timestep=target_timestep,
                                        current_timestep=current_timestep,
                                        template_used=template)

    return create_question_dict(question_id="",
                                question_text=question_text,
                                options=options,
                                correct_answer=correct_answer,
                                metadata=metadata,
                                evidence=evidence)


@eqa_generator(
    question_id="obj_rel_pos_random_t",
    question_type="single_choice",
    templates=[
        "At time {t}, what was the position of {obj_B} relative to {obj_A}?",
        "At timestamp {t}, was {obj_B} to the east, south, west, or north of {obj_A}?",
        "From the perspective of {obj_A}, where was {obj_B} at time {t}?",
        "In which direction of {obj_A} was {obj_B} located at time {t}?"
    ],
    description="Relative position between two objects at random time t.")
def generate_obj_rel_pos_random_t(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate relative position question at random time t."""
    current_timestep = tracker.current_timestep

    # Randomly select t from valid timesteps
    t = rng.randint(0, current_timestep)
    target_timestep = t

    # Get valid objects
    valid_objects = _get_valid_objects(tracker, target_timestep)

    if len(valid_objects) < 2:
        return None

    # Select obj_A randomly
    obj_a_id, obj_a_type, obj_a_loc_desc = rng.choice(valid_objects)

    # Filter objects where locDesc doesn't contain obj_a's objectType
    filtered_objects = [
        (oid, otype, locdesc) for oid, otype, locdesc in valid_objects
        if obj_a_type.lower() not in locdesc.lower() and oid != obj_a_id
    ]

    if not filtered_objects:
        return None

    # Select obj_B
    obj_b_id, obj_b_type, obj_b_loc_desc = rng.choice(filtered_objects)

    # Get positions
    obj_a_state = tracker.get_object_state_at_step(obj_a_id, target_timestep)
    obj_b_state = tracker.get_object_state_at_step(obj_b_id, target_timestep)

    if not obj_a_state or not obj_b_state:
        return None

    # Calculate direction and angle
    direction, angle_deg = calculate_relative_direction(
        obj_a_state.position, obj_b_state.position)

    # Create options
    all_directions = ["East", "South", "West", "North"]
    rng.shuffle(all_directions)
    options = get_options_list(all_directions)

    # Find correct answer label
    correct_answer = None
    for label, dir_name in options.items():
        if dir_name == direction:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_rel_pos_random_t.templates)
    question_text = template.format(t=t,
                                    obj_B=obj_b_loc_desc,
                                    obj_A=obj_a_loc_desc)

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "reference_object": {
            "objectId": obj_a_id,
            "position": obj_a_state.position
        },
        "target_object": {
            "objectId": obj_b_id,
            "position": obj_b_state.position
        },
        "relative_info": {
            "from_objA_to_objB_angle": angle_deg,
            "calculated_direction": direction
        },
        "options_info": {}
    }

    for label, dir_name in options.items():
        evidence["options_info"][label] = {
            "direction": dir_name,
            "angle_range": direction_angle_ranges[dir_name],
            "is_correct": (dir_name == direction)
        }

    # Create metadata
    metadata = format_question_metadata(question_id="obj_rel_pos_random_t",
                                        question_type="single_choice",
                                        question_timestep=target_timestep,
                                        current_timestep=current_timestep,
                                        template_used=template)

    return create_question_dict(question_id="",
                                question_text=question_text,
                                options=options,
                                correct_answer=correct_answer,
                                metadata=metadata,
                                evidence=evidence)
