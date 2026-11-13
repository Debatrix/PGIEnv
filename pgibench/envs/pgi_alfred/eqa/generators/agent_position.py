"""
Agent Spatiotemporal Position Question Generators

Generates questions about agent position, object visibility, and
spatial relationships between agent and objects at different temporal points.
"""

import random
from typing import Dict, List, Optional, Any
from types import ModuleType

from pgibench.envs.pgi_alfred.eqa.thor_tracker import ThorTracker
from pgibench.envs.pgi_alfred.eqa.utils import (
    get_options_list, select_distractors, has_sequential_number,
    format_question_metadata, create_question_dict, get_object_loc_desc,
    calculate_relative_direction, get_objects_in_direction,
    get_visible_objects_at_step, is_top_level_object)

# Define angle ranges for each direction (0° = North, clockwise)
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


@eqa_generator(
    question_id="obj_rel_agent_initial",
    question_type="single_choice",
    templates=[
        "Initially, was the {obj} to the east, south, west, or north of the agent?",
        "At the start, what was the direction of the {obj} relative to the agent?",
        "From the agent's starting position, where was the {obj} located?",
        "Initially, the {obj} was positioned in which direction relative to the agent?"
    ],
    description="Object direction relative to agent at t=0.")
def generate_obj_rel_agent_initial(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate object direction relative to agent question at t=0."""
    target_timestep = 0
    current_timestep = tracker.current_timestep

    # Get agent position at target timestep
    agent_state = tracker.get_agent_state_at_step(target_timestep)
    if not agent_state:
        return None

    agent_pos = agent_state.position

    # Get valid objects (exclude 'Top' type)
    valid_objects = []
    for obj_id in tracker.all_object_ids:
        if obj_id in tracker.historically_seen_objects:
            state = tracker.get_object_state_at_step(obj_id, target_timestep)
            if state and is_top_level_object(state):
                obj_loc_desc = get_object_loc_desc(state)
                if obj_loc_desc and not has_sequential_number(obj_loc_desc):
                    valid_objects.append(
                        (obj_id, obj_loc_desc, state.position))

    if not valid_objects:
        return None

    # Select random object
    obj_id, obj_loc_desc, obj_pos = rng.choice(valid_objects)

    # Calculate direction and angle
    direction, angle_deg = calculate_relative_direction(agent_pos, obj_pos)

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

    if correct_answer is None:
        return None

    # Select random template and format question
    template = rng.choice(generate_obj_rel_agent_initial.templates)
    question_text = template.format(obj=obj_loc_desc)

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "agent": {
            "position": agent_pos
        },
        "object": {
            "objectId": obj_id,
            "position": obj_pos
        },
        "relative_info": {
            "from_agent_to_object_angle": angle_deg,
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
    metadata = format_question_metadata(question_id="obj_rel_agent_initial",
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
    question_id="obj_rel_agent_current",
    question_type="single_choice",
    templates=[
        "Currently, is the {obj} to the east, south, west, or north of the agent?",
        "At this moment, what is the direction of the {obj} relative to the agent?",
        "From the agent's current position, where is the {obj} located?",
        "Right now, in which direction of the agent is the {obj}?",
        "The {obj} is positioned in which direction relative to the agent?"
    ],
    description="Object direction relative to agent now.")
def generate_obj_rel_agent_current(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate object direction relative to agent question at current moment."""
    target_timestep = tracker.current_timestep
    current_timestep = tracker.current_timestep

    # Get agent position at target timestep
    agent_state = tracker.get_agent_state_at_step(target_timestep)
    if not agent_state:
        return None

    agent_pos = agent_state.position

    # Get valid objects
    valid_objects = []
    for obj_id in tracker.all_object_ids:
        if obj_id in tracker.historically_seen_objects:
            state = tracker.get_object_state_at_step(obj_id, target_timestep)
            if state and is_top_level_object(state):
                obj_loc_desc = get_object_loc_desc(state)
                if obj_loc_desc and not has_sequential_number(obj_loc_desc):
                    valid_objects.append(
                        (obj_id, obj_loc_desc, state.position))

    if not valid_objects:
        return None

    # Select random object
    obj_id, obj_loc_desc, obj_pos = rng.choice(valid_objects)

    # Calculate direction and angle
    direction, angle_deg = calculate_relative_direction(agent_pos, obj_pos)

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

    if correct_answer is None:
        return None

    # Select random template and format question
    template = rng.choice(generate_obj_rel_agent_current.templates)
    question_text = template.format(obj=obj_loc_desc)

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "agent": {
            "position": agent_pos
        },
        "object": {
            "objectId": obj_id,
            "position": obj_pos
        },
        "relative_info": {
            "from_agent_to_object_angle": angle_deg,
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
    metadata = format_question_metadata(question_id="obj_rel_agent_current",
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
    question_id="obj_rel_agent_n_steps_ago",
    question_type="single_choice",
    templates=[
        "{n} steps ago, was the {obj} to the east, south, west, or north of the agent?",
        "From the agent's position {n} steps back, where was the {obj} located?",
        "{n} steps ago, the {obj} was positioned in which direction relative to the agent?",
        "What was the direction of the {obj} relative to the agent {n} steps before now?"
    ],
    description="Object direction relative to agent n steps ago.")
def generate_obj_rel_agent_n_steps_ago(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate object direction relative to agent question n steps ago."""
    current_timestep = tracker.current_timestep

    # Need at least 2 timesteps of history
    if current_timestep < 2:
        return None

    # Randomly select n (1 to current_timestep - 1)
    n = rng.randint(1, current_timestep - 1)
    target_timestep = current_timestep - n

    # Get agent position at target timestep
    agent_state = tracker.get_agent_state_at_step(target_timestep)
    if not agent_state:
        return None

    agent_pos = agent_state.position

    # Get valid objects
    valid_objects = []
    for obj_id in tracker.all_object_ids:
        if obj_id in tracker.historically_seen_objects:
            state = tracker.get_object_state_at_step(obj_id, target_timestep)
            if state and is_top_level_object(state):
                obj_loc_desc = get_object_loc_desc(state)
                if obj_loc_desc and not has_sequential_number(obj_loc_desc):
                    valid_objects.append(
                        (obj_id, obj_loc_desc, state.position))

    if not valid_objects:
        return None

    # Select random object
    obj_id, obj_loc_desc, obj_pos = rng.choice(valid_objects)

    # Calculate direction and angle
    direction, angle_deg = calculate_relative_direction(agent_pos, obj_pos)

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

    if correct_answer is None:
        return None

    # Select random template and format question
    template = rng.choice(generate_obj_rel_agent_n_steps_ago.templates)
    question_text = template.format(n=n, obj=obj_loc_desc)

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "agent": {
            "position": agent_pos
        },
        "object": {
            "objectId": obj_id,
            "position": obj_pos
        },
        "relative_info": {
            "from_agent_to_object_angle": angle_deg,
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
    metadata = format_question_metadata(
        question_id="obj_rel_agent_n_steps_ago",
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
    question_id="obj_rel_agent_random_t",
    question_type="single_choice",
    templates=[
        "At time {t}, was the {obj} to the east, south, west, or north of the agent?",
        "At step {t}, what was the direction of the {obj} relative to the agent?",
        "From the agent's position at time {t}, where was the {obj} located?",
        "At time {t}, in which direction of the agent was the {obj}?",
        "At time {t}, the {obj} was positioned in which direction relative to the agent?"
    ],
    description="Object direction relative to agent at random time t.")
def generate_obj_rel_agent_random_t(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate object direction relative to agent question at random time t."""
    current_timestep = tracker.current_timestep

    # Randomly select t from valid timesteps
    t = rng.randint(0, current_timestep)
    target_timestep = t

    # Get agent position at target timestep
    agent_state = tracker.get_agent_state_at_step(target_timestep)
    if not agent_state:
        return None

    agent_pos = agent_state.position

    # Get valid objects
    valid_objects = []
    for obj_id in tracker.all_object_ids:
        if obj_id in tracker.historically_seen_objects:
            state = tracker.get_object_state_at_step(obj_id, target_timestep)
            if state and is_top_level_object(state):
                obj_loc_desc = get_object_loc_desc(state)
                if obj_loc_desc and not has_sequential_number(obj_loc_desc):
                    valid_objects.append(
                        (obj_id, obj_loc_desc, state.position))

    if not valid_objects:
        return None

    # Select random object
    obj_id, obj_loc_desc, obj_pos = rng.choice(valid_objects)

    # Calculate direction and angle
    direction, angle_deg = calculate_relative_direction(agent_pos, obj_pos)

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

    if correct_answer is None:
        return None

    # Select random template and format question
    template = rng.choice(generate_obj_rel_agent_random_t.templates)
    question_text = template.format(t=t, obj=obj_loc_desc)

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "agent": {
            "position": agent_pos
        },
        "object": {
            "objectId": obj_id,
            "position": obj_pos
        },
        "relative_info": {
            "from_agent_to_object_angle": angle_deg,
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
    metadata = format_question_metadata(question_id="obj_rel_agent_random_t",
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
    question_id="agent_rel_objs_initial",
    question_type="multi_choice",
    templates=[
        "Initially, which objects were located to the {D} of the agent?",
        "At the start, select all objects that are on the {D} side of the agent.",
        "Which objects were to the {D} of the agent initially?",
        "At the beginning, identify the objects located to the {D} of the agent."
    ],
    description="Objects located on the D side of agent at t=0.")
def generate_agent_rel_objs_initial(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate multi-choice question about objects in cardinal direction from agent at t=0."""
    target_timestep = 0
    current_timestep = tracker.current_timestep

    # Get agent position at target timestep
    agent_state = tracker.get_agent_state_at_step(target_timestep)
    if not agent_state:
        return None

    agent_pos = agent_state.position

    # Randomly select cardinal direction
    direction = rng.choice(["East", "South", "West", "North"])

    # Get objects in that direction
    objects_in_direction, distractor_pool = get_objects_in_direction(
        tracker, agent_pos, direction, target_timestep)

    if len(objects_in_direction) < 1:
        return None

    # Select X1 correct options (1-3 objects)
    num_correct = min(rng.randint(1, min(3, len(objects_in_direction))),
                      len(objects_in_direction))
    correct_objects = rng.sample(objects_in_direction, num_correct)

    # Get all object IDs NOT in that direction for distractors
    if not distractor_pool:
        return None

    # Select X2 incorrect options (2-3 objects)
    num_incorrect = min(rng.randint(2, 3), len(distractor_pool))
    if num_incorrect < 2:
        return None

    incorrect_objects = rng.sample(distractor_pool, num_incorrect)

    # Combine and shuffle options
    all_obj_options = [
        (obj_id, obj_type) for obj_id, obj_type, _ in correct_objects
    ] + [(obj_id, obj_type) for obj_id, obj_type, _ in incorrect_objects]
    rng.shuffle(all_obj_options)

    # Use objectType for options
    options = get_options_list([obj_type for _, obj_type in all_obj_options])

    # Find all correct answer labels
    correct_labels = []
    correct_types = {obj_type for _, obj_type, _ in correct_objects}
    for label, obj_type in options.items():
        if obj_type in correct_types:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)

    # Select random template and format question
    template = rng.choice(generate_agent_rel_objs_initial.templates)
    question_text = template.format(D=direction.lower())

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "agent": {
            "position": agent_pos,
            "direction_filter": direction
        },
        "options_info": {}
    }

    _inv_options = {v: k for k, v in options.items()}
    for obj_id, obj_type in all_obj_options:
        is_correct = (obj_type in correct_types)
        label = _inv_options[obj_type]
        obj_state = tracker.get_object_state_at_step(obj_id, target_timestep)
        obj_pos = obj_state.position
        calc_direction, angle_deg = calculate_relative_direction(
            agent_pos, obj_pos)

        evidence["options_info"][label] = {
            "is_correct": is_correct,
            "objectType": obj_type,
            "objectId": obj_id,
            "position": obj_pos,
            "from_agent_to_object_angle": angle_deg,
            "calculated_direction": calc_direction,
            "angle_range": direction_angle_ranges[calc_direction]
        }

    # Create metadata
    metadata = format_question_metadata(question_id="agent_rel_objs_initial",
                                        question_type="multi_choice",
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
    question_id="agent_rel_objs_current",
    question_type="multi_choice",
    templates=[
        "Currently, which objects are located to the {D} of the agent?",
        "At this moment, select all objects that are on the {D} side of the agent.",
        "Right now, which objects are positioned to the {D} of the agent?",
        "Which objects are to the {D} of the agent currently?",
        "Identify the objects located to the {D} of the agent right now."
    ],
    description="Objects located on the D side of agent now.")
def generate_agent_rel_objs_current(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate multi-choice question about objects in cardinal direction from agent now."""
    target_timestep = tracker.current_timestep
    current_timestep = tracker.current_timestep

    # Get agent position at target timestep
    agent_state = tracker.get_agent_state_at_step(target_timestep)
    if not agent_state:
        return None

    agent_pos = agent_state.position

    # Randomly select cardinal direction
    direction = rng.choice(["East", "South", "West", "North"])

    # Get objects in that direction
    objects_in_direction, distractor_pool = get_objects_in_direction(
        tracker, agent_pos, direction, target_timestep)

    if len(objects_in_direction) < 1:
        return None

    # Select X1 correct options (1-3 objects)
    num_correct = min(rng.randint(1, min(3, len(objects_in_direction))),
                      len(objects_in_direction))
    correct_objects = rng.sample(objects_in_direction, num_correct)

    if not distractor_pool:
        return None

    # Select X2 incorrect options (2-3 objects)
    num_incorrect = min(rng.randint(2, 3), len(distractor_pool))
    if num_incorrect < 2:
        return None

    incorrect_objects = rng.sample(distractor_pool, num_incorrect)

    # Combine and shuffle options
    all_obj_options = [
        (obj_id, obj_type) for obj_id, obj_type, _ in correct_objects
    ] + [(obj_id, obj_type) for obj_id, obj_type, _ in incorrect_objects]
    rng.shuffle(all_obj_options)

    # Use objectType for options
    options = get_options_list([obj_type for _, obj_type in all_obj_options])

    # Find all correct answer labels
    correct_labels = []
    correct_types = {obj_type for _, obj_type, _ in correct_objects}
    for label, obj_type in options.items():
        if obj_type in correct_types:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)

    # Select random template and format question
    template = rng.choice(generate_agent_rel_objs_current.templates)
    question_text = template.format(D=direction.lower())

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "agent": {
            "position": agent_pos,
            "direction_filter": direction
        },
        "options_info": {}
    }

    _inv_options = {v: k for k, v in options.items()}
    for obj_id, obj_type in all_obj_options:
        is_correct = (obj_type in correct_types)
        label = _inv_options[obj_type]
        obj_state = tracker.get_object_state_at_step(obj_id, target_timestep)
        obj_pos = obj_state.position
        calc_direction, angle_deg = calculate_relative_direction(
            agent_pos, obj_pos)

        evidence["options_info"][label] = {
            "is_correct": is_correct,
            "objectType": obj_type,
            "objectId": obj_id,
            "position": obj_pos,
            "from_agent_to_object_angle": angle_deg,
            "calculated_direction": calc_direction,
            "angle_range": direction_angle_ranges[calc_direction]
        }

    # Create metadata
    metadata = format_question_metadata(question_id="agent_rel_objs_current",
                                        question_type="multi_choice",
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
    question_id="agent_rel_objs_n_steps_ago",
    question_type="multi_choice",
    templates=[
        "{n} steps ago, which objects were located to the {D} of the agent?",
        "{n} steps back, which objects were positioned to the {D} of the agent?",
        "Which objects were to the {D} of the agent {n} steps ago?",
    ],
    description="Objects located on the D side of agent n steps ago.")
def generate_agent_rel_objs_n_steps_ago(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate multi-choice question about objects in cardinal direction from agent n steps ago."""
    current_timestep = tracker.current_timestep

    # Need at least 2 timesteps of history
    if current_timestep < 2:
        return None

    # Randomly select n (1 to current_timestep - 1)
    n = rng.randint(1, current_timestep - 1)
    target_timestep = current_timestep - n

    # Get agent position at target timestep
    agent_state = tracker.get_agent_state_at_step(target_timestep)
    if not agent_state:
        return None

    agent_pos = agent_state.position
    direction = rng.choice(["East", "South", "West", "North"])
    objects_in_direction, distractor_pool = get_objects_in_direction(
        tracker, agent_pos, direction, target_timestep)

    if len(objects_in_direction) < 1:
        return None

    num_correct = min(rng.randint(1, min(3, len(objects_in_direction))),
                      len(objects_in_direction))
    correct_objects = rng.sample(objects_in_direction, num_correct)

    if not distractor_pool:
        return None

    num_incorrect = min(rng.randint(2, 3), len(distractor_pool))
    if num_incorrect < 2:
        return None

    incorrect_objects = rng.sample(distractor_pool, num_incorrect)
    all_obj_options = [
        (obj_id, obj_type) for obj_id, obj_type, _ in correct_objects
    ] + [(obj_id, obj_type) for obj_id, obj_type, _ in incorrect_objects]
    rng.shuffle(all_obj_options)
    options = get_options_list([obj_type for _, obj_type in all_obj_options])

    correct_labels = []
    correct_types = {obj_type for _, obj_type, _ in correct_objects}
    for label, obj_type in options.items():
        if obj_type in correct_types:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)
    template = rng.choice(generate_agent_rel_objs_n_steps_ago.templates)
    question_text = template.format(n=n, D=direction.lower())

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "agent": {
            "position": agent_pos,
            "direction_filter": direction
        },
        "options_info": {}
    }

    _inv_options = {v: k for k, v in options.items()}
    for obj_id, obj_type in all_obj_options:
        is_correct = (obj_type in correct_types)
        label = _inv_options[obj_type]
        obj_state = tracker.get_object_state_at_step(obj_id, target_timestep)
        obj_pos = obj_state.position
        calc_direction, angle_deg = calculate_relative_direction(
            agent_pos, obj_pos)

        evidence["options_info"][label] = {
            "is_correct": is_correct,
            "objectType": obj_type,
            "objectId": obj_id,
            "position": obj_pos,
            "from_agent_to_object_angle": angle_deg,
            "calculated_direction": calc_direction,
            "angle_range": direction_angle_ranges[calc_direction]
        }

    metadata = format_question_metadata(
        question_id="agent_rel_objs_n_steps_ago",
        question_type="multi_choice",
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
    question_id="agent_rel_objs_random_t",
    question_type="multi_choice",
    templates=[
        "At time {t}, which objects were located to the {D} of the agent?",
        "At step {t}, select all objects that are on the {D} side of the agent.",
        "At time {t}, which objects were positioned to the {D} of the agent?",
        "Which objects were to the {D} of the agent at time {t}?",
        "At time {t}, identify the objects located to the {D} of the agent."
    ],
    description="Objects located on the D side of agent at random time t.")
def generate_agent_rel_objs_random_t(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate multi-choice question about objects in cardinal direction from agent at random time t."""
    current_timestep = tracker.current_timestep
    t = rng.randint(0, current_timestep)
    target_timestep = t

    agent_state = tracker.get_agent_state_at_step(target_timestep)
    if not agent_state:
        return None

    agent_pos = agent_state.position
    direction = rng.choice(["East", "South", "West", "North"])
    objects_in_direction, distractor_pool = get_objects_in_direction(
        tracker, agent_pos, direction, target_timestep)

    if len(objects_in_direction) < 1:
        return None

    num_correct = min(rng.randint(1, min(3, len(objects_in_direction))),
                      len(objects_in_direction))
    correct_objects = rng.sample(objects_in_direction, num_correct)

    if not distractor_pool:
        return None

    num_incorrect = min(rng.randint(2, 3), len(distractor_pool))
    if num_incorrect < 2:
        return None

    incorrect_objects = rng.sample(distractor_pool, num_incorrect)
    all_obj_options = [
        (obj_id, obj_type) for obj_id, obj_type, _ in correct_objects
    ] + [(obj_id, obj_type) for obj_id, obj_type, _ in incorrect_objects]
    rng.shuffle(all_obj_options)
    options = get_options_list([obj_type for _, obj_type in all_obj_options])

    correct_labels = []
    correct_types = {obj_type for _, obj_type, _ in correct_objects}
    for label, obj_type in options.items():
        if obj_type in correct_types:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)
    template = rng.choice(generate_agent_rel_objs_random_t.templates)
    question_text = template.format(t=t, D=direction.lower())

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "agent": {
            "position": agent_pos,
            "direction_filter": direction
        },
        "options_info": {}
    }

    _inv_options = {v: k for k, v in options.items()}
    for obj_id, obj_type in all_obj_options:
        is_correct = (obj_type in correct_types)
        label = _inv_options[obj_type]
        obj_state = tracker.get_object_state_at_step(obj_id, target_timestep)
        obj_pos = obj_state.position
        calc_direction, angle_deg = calculate_relative_direction(
            agent_pos, obj_pos)

        evidence["options_info"][label] = {
            "is_correct": is_correct,
            "objectType": obj_type,
            "objectId": obj_id,
            "position": obj_pos,
            "from_agent_to_object_angle": angle_deg,
            "calculated_direction": calc_direction,
            "angle_range": direction_angle_ranges[calc_direction]
        }

    metadata = format_question_metadata(question_id="agent_rel_objs_random_t",
                                        question_type="multi_choice",
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
    question_id="agent_vis_initial",
    question_type="multi_choice",
    templates=[
        "Initially, which object are visible to the agent?",
        "At the start, which of these objects are within the agent's field of view?",
        "Which objects are visible to the agent at the start?",
        "Identify the objects that are visible to the agent initially."
    ],
    description="Objects visible to agent initially.")
def generate_agent_vis_initial(tracker: ThorTracker, constants: ModuleType,
                               timestep: int,
                               rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate multi-choice question about objects visible to agent now."""
    target_timestep = 0
    current_timestep = tracker.current_timestep

    # Get all visible objects at target timestep
    visible_objects = get_visible_objects_at_step(tracker, target_timestep)

    if not visible_objects:
        return None

    # Get unique object types from visible objects
    visible_types = set()
    visible_type_to_ids = {}
    for obj_id, obj_type in visible_objects:
        visible_types.add(obj_type)
        if obj_type not in visible_type_to_ids:
            visible_type_to_ids[obj_type] = []
        visible_type_to_ids[obj_type].append(obj_id)

    visible_types = list(visible_types)

    # Get all object types in scene
    all_types = set()
    for obj_id in tracker.all_object_ids:
        if obj_id in tracker.historically_seen_objects:
            state = tracker.get_object_state_at_step(obj_id, target_timestep)
            if state:
                all_types.add(state.object_type)

    # Select X1 correct options (1-3 visible object types)
    num_correct = min(rng.randint(1, min(3, len(visible_types))),
                      len(visible_types))
    correct_types = rng.sample(visible_types, num_correct)

    # Select X2 incorrect options (2-3 types NOT visible)
    not_visible_types = list(all_types - set(visible_types))
    if not not_visible_types:
        # Fallback: use visible types but won't match
        not_visible_types = []

    if len(not_visible_types) < 2:
        # Need at least 2 distractors
        return None

    num_incorrect = min(rng.randint(2, 3), len(not_visible_types))
    incorrect_types = rng.sample(not_visible_types, num_incorrect)

    # Combine and shuffle options
    all_type_options = correct_types + incorrect_types
    rng.shuffle(all_type_options)
    options = get_options_list(all_type_options)

    # Find all correct answer labels
    correct_labels = []
    for label, obj_type in options.items():
        if obj_type in correct_types:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)

    # Select random template and format question
    template = rng.choice(generate_agent_vis_initial.templates)
    question_text = template

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        'visible_objects': [x[0] for x in visible_objects],
        "options_info": {}
    }

    # Build options_info
    for label, obj_type in options.items():
        is_correct = (obj_type in correct_types)

        evidence["options_info"][label] = {
            "objectType": obj_type,
            "is_correct": is_correct,
        }

    # Create metadata
    metadata = format_question_metadata(question_id="agent_vis_initial",
                                        question_type="multi_choice",
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
    question_id="agent_vis_n_steps_ago",
    question_type="multi_choice",
    templates=[
        "{n} steps ago, which object types were visible to the agent?",
        "{n} steps back, which of these objects were within the agent's field of view?",
        "Which objects were visible to the agent {n} steps ago?",
        "Identify the objects that were visible to the agent {n} steps back."
    ],
    description="Objects visible to agent n steps ago.")
def generate_agent_vis_n_steps_ago(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate multi-choice question about objects visible to agent n steps ago."""
    current_timestep = tracker.current_timestep

    # Need at least 2 timesteps of history
    if current_timestep < 2:
        return None

    # Randomly select n (1 to current_timestep - 1)
    n = rng.randint(1, current_timestep - 1)
    target_timestep = current_timestep - n

    # Get all visible objects at target timestep
    visible_objects = get_visible_objects_at_step(tracker, target_timestep)

    if not visible_objects:
        return None

    # Get unique object types from visible objects
    visible_types = set()
    visible_type_to_ids = {}
    for obj_id, obj_type in visible_objects:
        visible_types.add(obj_type)
        if obj_type not in visible_type_to_ids:
            visible_type_to_ids[obj_type] = []
        visible_type_to_ids[obj_type].append(obj_id)

    visible_types = list(visible_types)

    # Get all object types in scene
    all_types = set()
    for obj_id in tracker.all_object_ids:
        if obj_id in tracker.historically_seen_objects:
            state = tracker.get_object_state_at_step(obj_id, target_timestep)
            if state:
                all_types.add(state.object_type)

    # Select X1 correct options (1-3 visible object types)
    num_correct = min(rng.randint(1, min(3, len(visible_types))),
                      len(visible_types))
    correct_types = rng.sample(visible_types, num_correct)

    # Select X2 incorrect options (2-3 types NOT visible)
    not_visible_types = list(all_types - set(visible_types))
    if not not_visible_types:
        # Fallback: use visible types but won't match
        not_visible_types = []

    if len(not_visible_types) < 2:
        # Need at least 2 distractors
        return None

    num_incorrect = min(rng.randint(2, 3), len(not_visible_types))
    incorrect_types = rng.sample(not_visible_types, num_incorrect)

    # Combine and shuffle options
    all_type_options = correct_types + incorrect_types
    rng.shuffle(all_type_options)
    options = get_options_list(all_type_options)

    # Find all correct answer labels
    correct_labels = []
    for label, obj_type in options.items():
        if obj_type in correct_types:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)

    # Select random template and format question
    template = rng.choice(generate_agent_vis_n_steps_ago.templates)
    question_text = template.format(n=n)

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        'visible_objects': [x[0] for x in visible_objects],
        "options_info": {}
    }

    # Build options_info
    for label, obj_type in options.items():
        is_correct = (obj_type in correct_types)

        evidence["options_info"][label] = {
            "objectType": obj_type,
            "is_correct": is_correct,
        }

    # Create metadata
    metadata = format_question_metadata(question_id="agent_vis_n_steps_ago",
                                        question_type="multi_choice",
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
    question_id="agent_vis_random_t",
    question_type="multi_choice",
    templates=[
        "At time {t}, which object types were visible to the agent?",
        "At step {t}, select all objects that the agent could see.",
        "At time {t}, which of these objects were within the agent's field of view?",
        "Which objects were visible to the agent at time {t}?",
        "At time {t}, identify the objects that were visible to the agent."
    ],
    description="Objects visible to agent at random time t.")
def generate_agent_vis_random_t(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate multi-choice question about objects visible to agent at random time t."""
    current_timestep = tracker.current_timestep

    # Randomly select t from valid timesteps
    t = rng.randint(0, current_timestep)
    target_timestep = t

    # Get all visible objects at target timestep
    visible_objects = get_visible_objects_at_step(tracker, target_timestep)

    if not visible_objects:
        return None

    # Get unique object types from visible objects
    visible_types = set()
    visible_type_to_ids = {}
    for obj_id, obj_type in visible_objects:
        visible_types.add(obj_type)
        if obj_type not in visible_type_to_ids:
            visible_type_to_ids[obj_type] = []
        visible_type_to_ids[obj_type].append(obj_id)

    visible_types = list(visible_types)

    # Get all object types in scene
    all_types = set()
    for obj_id in tracker.all_object_ids:
        if obj_id in tracker.historically_seen_objects:
            state = tracker.get_object_state_at_step(obj_id, target_timestep)
            if state:
                all_types.add(state.object_type)

    # Select X1 correct options (1-3 visible object types)
    num_correct = min(rng.randint(1, min(3, len(visible_types))),
                      len(visible_types))
    correct_types = rng.sample(visible_types, num_correct)

    # Select X2 incorrect options (2-3 types NOT visible)
    not_visible_types = list(all_types - set(visible_types))
    if not not_visible_types:
        # Fallback: use visible types but won't match
        not_visible_types = []

    if len(not_visible_types) < 2:
        # Need at least 2 distractors
        return None

    num_incorrect = min(rng.randint(2, 3), len(not_visible_types))
    incorrect_types = rng.sample(not_visible_types, num_incorrect)

    # Combine and shuffle options
    all_type_options = correct_types + incorrect_types
    rng.shuffle(all_type_options)
    options = get_options_list(all_type_options)

    # Find all correct answer labels
    correct_labels = []
    for label, obj_type in options.items():
        if obj_type in correct_types:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)

    # Select random template and format question
    template = rng.choice(generate_agent_vis_random_t.templates)
    question_text = template.format(t=t)

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        'visible_objects': [x[0] for x in visible_objects],
        "options_info": {}
    }

    # Build options_info
    for label, obj_type in options.items():
        is_correct = (obj_type in correct_types)

        evidence["options_info"][label] = {
            "objectType": obj_type,
            "is_correct": is_correct,
        }

    # Create metadata
    metadata = format_question_metadata(question_id="agent_vis_random_t",
                                        question_type="multi_choice",
                                        question_timestep=target_timestep,
                                        current_timestep=current_timestep,
                                        template_used=template)

    return create_question_dict(question_id="",
                                question_text=question_text,
                                options=options,
                                correct_answer=correct_answer,
                                metadata=metadata,
                                evidence=evidence)
