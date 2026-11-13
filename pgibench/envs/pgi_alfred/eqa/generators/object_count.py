"""
Object Count Question Generators

Generates questions about object counts in containers and scenes
at different temporal points.
"""

import random
from typing import Dict, List, Optional, Any
from types import ModuleType

from pgibench.envs.pgi_alfred.eqa.thor_tracker import ThorTracker
from pgibench.envs.pgi_alfred.eqa.utils import (
    get_options_list, has_sequential_number, format_question_metadata,
    create_question_dict, get_object_loc_desc, get_objects_in_container,
    get_objects_by_type, generate_count_distractors)


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


def _get_containers(tracker: ThorTracker, timestep: int) -> List[tuple]:
    """
    Get all containers at a given timestep where all objects inside have been seen.

    Args:
        tracker: ThorTracker instance
        timestep: Timestep to query

    Returns:
        List of (container_id, container_locDesc, objects_inside) tuples
        where all objects inside the container have been seen by the agent
    """
    containers = []
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if state and state.receptacle_object_ids:
            # Check if all objects inside the container have been seen
            objects_inside = state.receptacle_object_ids
            all_objects_seen = all(obj_id in tracker.historically_seen_objects
                                   for obj_id in objects_inside)

            # Only include containers where all contents have been seen
            if all_objects_seen:
                container_loc_desc = get_object_loc_desc(state)
                if container_loc_desc and not has_sequential_number(
                        container_loc_desc):
                    containers.append(
                        (obj_id, container_loc_desc, objects_inside))

    return containers


def _get_all_seen_object_types_in_scene(tracker: ThorTracker,
                                        timestep: int) -> List[str]:
    """
    Get all unique object types in the scene where ALL instances have been seen.

    Args:
        tracker: ThorTracker instance
        timestep: Timestep to query

    Returns:
        List of unique objectTypes where all instances have been seen
    """
    # Get historically seen objects at the timestep
    obs = tracker.observations.get(timestep)
    if not obs:
        return []

    seen_objects = obs.historically_seen_objects

    # Group objects by type and check if all instances of each type are seen
    type_to_objects = {}
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if state:
            obj_type = state.object_type
            if obj_type not in type_to_objects:
                type_to_objects[obj_type] = []
            type_to_objects[obj_type].append(obj_id)

    # Filter to only types where all instances have been seen
    seen_types = []
    for obj_type, obj_ids in type_to_objects.items():
        if all(obj_id in seen_objects for obj_id in obj_ids):
            seen_types.append(obj_type)

    return seen_types


@eqa_generator(
    question_id="obj_count_in_container_initial",
    question_type="single_choice",
    templates=[
        "How many objects were inside the {RE_obj} at the start?",
        "When the task began, what was the count of objects in the {RE_obj}?",
        "Initially, how many items were contained within the {RE_obj}?",
        "What was the number of objects in the {RE_obj} at the beginning?"
    ],
    description="Object count in container at t=0.")
def generate_obj_count_in_container_initial(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate count in container question at t=0."""
    target_timestep = 0
    current_timestep = tracker.current_timestep

    # Get containers
    containers = _get_containers(tracker, target_timestep)

    if not containers:
        return None

    # Select random container
    container_id, container_loc_desc, objects_inside = rng.choice(containers)

    # Get count
    actual_count = len(objects_inside)

    # Generate distractors
    num_distractors = min(
        3, len(generate_count_distractors(actual_count, 3, rng)))
    if num_distractors < 1:
        # Try with fewer distractors
        num_distractors = min(
            2, len(generate_count_distractors(actual_count, 2, rng)))
    if num_distractors < 1:
        return None

    distractor_counts = generate_count_distractors(actual_count,
                                                   num_distractors, rng)

    # Ensure unique options
    all_counts = [actual_count] + distractor_counts
    all_counts = list(set(all_counts))  # Remove duplicates
    rng.shuffle(all_counts)
    options = get_options_list(all_counts)

    # Find correct answer label
    correct_answer = None
    for label, count in options.items():
        if count == actual_count:
            correct_answer = label
            break

    if correct_answer is None:
        return None

    # Select random template and format question
    template = rng.choice(generate_obj_count_in_container_initial.templates)
    question_text = template.format(RE_obj=container_loc_desc)

    # Collect evidence
    container_state = tracker.get_object_state_at_step(container_id,
                                                       target_timestep)

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "container": {
            "objectId": container_id,
            "receptacleObjectIds": objects_inside  # Proves count
        },
    }

    # Create metadata
    metadata = format_question_metadata(
        question_id="obj_count_in_container_initial",
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
    question_id="obj_count_in_container_current",
    question_type="single_choice",
    templates=[
        "How many objects are inside the {RE_obj} right now?",
        "At this moment, what is the count of objects in the {RE_obj}?",
        "Currently, how many items are contained within the {RE_obj}?",
        "What is the number of objects in the {RE_obj} at the current time?"
    ],
    description="Object count in container now.")
def generate_obj_count_in_container_current(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate count in container question at current moment."""
    target_timestep = tracker.current_timestep
    current_timestep = tracker.current_timestep

    # Get containers
    containers = _get_containers(tracker, target_timestep)

    if not containers:
        return None

    # Select random container
    container_id, container_loc_desc, objects_inside = rng.choice(containers)

    # Get count
    actual_count = len(objects_inside)

    # Generate distractors
    num_distractors = min(
        3, len(generate_count_distractors(actual_count, 3, rng)))
    if num_distractors < 1:
        num_distractors = min(
            2, len(generate_count_distractors(actual_count, 2, rng)))
    if num_distractors < 1:
        return None

    distractor_counts = generate_count_distractors(actual_count,
                                                   num_distractors, rng)

    # Ensure unique options
    all_counts = [actual_count] + distractor_counts
    all_counts = list(set(all_counts))
    rng.shuffle(all_counts)
    options = get_options_list(all_counts)

    # Find correct answer label
    correct_answer = None
    for label, count in options.items():
        if count == actual_count:
            correct_answer = label
            break

    if correct_answer is None:
        return None

    # Select random template and format question
    template = rng.choice(generate_obj_count_in_container_current.templates)
    question_text = template.format(RE_obj=container_loc_desc)

    # Collect evidence
    container_state = tracker.get_object_state_at_step(container_id,
                                                       target_timestep)

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "container": {
            "objectId": container_id,
            "receptacleObjectIds": objects_inside
        },
    }

    # Create metadata
    metadata = format_question_metadata(
        question_id="obj_count_in_container_current",
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
    question_id="obj_count_in_container_n_steps_ago",
    question_type="single_choice",
    templates=[
        "How many objects were inside the {RE_obj} {n} steps ago?",
        "{n} steps ago, how many items were contained within the {RE_obj}?",
        "What was the number of objects in the {RE_obj} {n} steps before now?"
    ],
    description="Object count in container n steps ago.")
def generate_obj_count_in_container_n_steps_ago(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate count in container question n steps ago."""
    current_timestep = tracker.current_timestep

    # Need at least 2 timesteps of history
    if current_timestep < 2:
        return None

    # Randomly select n
    n = rng.randint(1, current_timestep - 1)
    target_timestep = current_timestep - n

    # Get containers
    containers = _get_containers(tracker, target_timestep)

    if not containers:
        return None

    # Select random container
    container_id, container_loc_desc, objects_inside = rng.choice(containers)

    # Get count
    actual_count = len(objects_inside)

    # Generate distractors
    num_distractors = min(
        3, len(generate_count_distractors(actual_count, 3, rng)))
    if num_distractors < 1:
        num_distractors = min(
            2, len(generate_count_distractors(actual_count, 2, rng)))
    if num_distractors < 1:
        return None

    distractor_counts = generate_count_distractors(actual_count,
                                                   num_distractors, rng)

    # Ensure unique options
    all_counts = [actual_count] + distractor_counts
    all_counts = list(set(all_counts))
    rng.shuffle(all_counts)
    options = get_options_list(all_counts)

    # Find correct answer label
    correct_answer = None
    for label, count in options.items():
        if count == actual_count:
            correct_answer = label
            break

    if correct_answer is None:
        return None

    # Select random template and format question
    template = rng.choice(
        generate_obj_count_in_container_n_steps_ago.templates)
    question_text = template.format(n=n, RE_obj=container_loc_desc)

    # Collect evidence
    container_state = tracker.get_object_state_at_step(container_id,
                                                       target_timestep)

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "container": {
            "objectId": container_id,
            "receptacleObjectIds": objects_inside
        },
    }

    # Create metadata
    metadata = format_question_metadata(
        question_id="obj_count_in_container_n_steps_ago",
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
    question_id="obj_count_in_container_random_t",
    question_type="single_choice",
    templates=[
        "At time {t}, how many objects were inside the {RE_obj}?",
        "At time {t}, what was the count of objects in the {RE_obj}?",
        "At time {t}, how many items were contained within the {RE_obj}?",
        "What was the number of objects in the {RE_obj} at time {t}?"
    ],
    description="Object count in container at random time t.")
def generate_obj_count_in_container_random_t(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate count in container question at random time t."""
    current_timestep = tracker.current_timestep

    # Randomly select t
    t = rng.randint(0, current_timestep)
    target_timestep = t

    # Get containers
    containers = _get_containers(tracker, target_timestep)

    if not containers:
        return None

    # Select random container
    container_id, container_loc_desc, objects_inside = rng.choice(containers)

    # Get count
    actual_count = len(objects_inside)

    # Generate distractors
    num_distractors = min(
        3, len(generate_count_distractors(actual_count, 3, rng)))
    if num_distractors < 1:
        num_distractors = min(
            2, len(generate_count_distractors(actual_count, 2, rng)))
    if num_distractors < 1:
        return None

    distractor_counts = generate_count_distractors(actual_count,
                                                   num_distractors, rng)

    # Ensure unique options
    all_counts = [actual_count] + distractor_counts
    all_counts = list(set(all_counts))
    rng.shuffle(all_counts)
    options = get_options_list(all_counts)

    # Find correct answer label
    correct_answer = None
    for label, count in options.items():
        if count == actual_count:
            correct_answer = label
            break

    if correct_answer is None:
        return None

    # Select random template and format question
    template = rng.choice(generate_obj_count_in_container_random_t.templates)
    question_text = template.format(t=t, RE_obj=container_loc_desc)

    # Collect evidence
    container_state = tracker.get_object_state_at_step(container_id,
                                                       target_timestep)

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "container": {
            "objectId": container_id,
            "receptacleObjectIds": objects_inside
        },
    }

    # Create metadata
    metadata = format_question_metadata(
        question_id="obj_count_in_container_random_t",
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
    question_id="obj_count_in_scene_current",
    question_type="single_choice",
    templates=[
        "How many {obj} are there in the scene right now?",
        "At this moment, what is the total number of {obj} present?",
        "Currently, count the number of {obj} in the environment.",
        "What is the number of {obj} present at the current time?",
        "How many {obj} can be found in the scene now?"
    ],
    description="Total count of object type now.")
def generate_obj_count_in_scene_current(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate scene count question at current moment."""
    target_timestep = tracker.current_timestep
    current_timestep = tracker.current_timestep

    # Get all object types where ALL instances have been seen
    object_types = _get_all_seen_object_types_in_scene(tracker,
                                                       current_timestep)

    if not object_types:
        return None

    # Select random object type
    obj_type = rng.choice(object_types)

    # Get count of objects of this type
    objects_of_type = get_objects_by_type(tracker, obj_type, target_timestep)
    actual_count = len(objects_of_type)

    # Generate distractors
    num_distractors = min(
        3, len(generate_count_distractors(actual_count, 3, rng)))
    if num_distractors < 1:
        num_distractors = min(
            2, len(generate_count_distractors(actual_count, 2, rng)))
    if num_distractors < 1:
        return None

    distractor_counts = generate_count_distractors(actual_count,
                                                   num_distractors, rng)

    # Ensure unique options
    all_counts = [actual_count] + distractor_counts
    all_counts = list(set(all_counts))
    rng.shuffle(all_counts)
    options = get_options_list(all_counts)

    # Find correct answer label
    correct_answer = None
    for label, count in options.items():
        if count == actual_count:
            correct_answer = label
            break

    if correct_answer is None:
        return None

    # Select random template and format question
    template = rng.choice(generate_obj_count_in_scene_current.templates)
    question_text = template.format(obj=obj_type.lower())

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "object_type": obj_type,
        "all_objects_of_type": objects_of_type,  # Proves count
    }

    # Create metadata
    metadata = format_question_metadata(
        question_id="obj_count_in_scene_current",
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
