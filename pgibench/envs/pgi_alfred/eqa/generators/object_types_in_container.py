"""
Object Types in Container Question Generators

Generates multi-choice questions about which object types are inside
a specific container at different temporal points.
"""

import random
from typing import Dict, List, Optional, Any
from types import ModuleType

from pgibench.envs.pgi_alfred.eqa.thor_tracker import ThorTracker
from pgibench.envs.pgi_alfred.eqa.utils import (
    get_options_list, select_distractors, has_sequential_number,
    format_question_metadata, create_question_dict, get_object_loc_desc,
    get_objects_in_container)


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


def _get_containers_with_objects(tracker: ThorTracker,
                                 timestep: int) -> List[tuple]:
    """
    Get containers that have objects inside at a given timestep,
    where all objects inside have been seen.

    Args:
        tracker: ThorTracker instance
        timestep: Timestep to query

    Returns:
        List of (container_id, container_locDesc, object_ids_inside) tuples
        where all objects inside the container have been seen by the agent
    """
    # Get historically seen objects at the timestep
    seen_objects = tracker.historically_seen_objects

    containers = []
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if state and state.receptacle_object_ids:
            # Check if all objects inside the container have been seen
            objects_inside = state.receptacle_object_ids
            all_objects_seen = all(obj_id in seen_objects
                                   for obj_id in objects_inside)

            # Only include containers where all contents have been seen
            if all_objects_seen:
                container_loc_desc = get_object_loc_desc(state)
                if container_loc_desc and not has_sequential_number(
                        container_loc_desc):
                    containers.append(
                        (obj_id, container_loc_desc, objects_inside))

    return containers


def _get_unique_object_types(tracker: ThorTracker, object_ids: List[str],
                             timestep: int) -> List[str]:
    """
    Get unique object types from a list of object IDs.

    Args:
        tracker: ThorTracker instance
        object_ids: List of object IDs
        timestep: Timestep to query

    Returns:
        List of unique objectTypes
    """
    obj_types = set()
    for obj_id in tracker.all_object_ids:
        if obj_id in object_ids:
            state = tracker.get_object_state_at_step(obj_id, timestep)
            if state:
                obj_types.add(state.object_type)
    return list(obj_types)


def _get_all_object_types_in_scene(tracker: ThorTracker,
                                   timestep: int) -> List[str]:
    """
    Get all unique object types in the scene at a timestep.

    Args:
        tracker: ThorTracker instance
        timestep: Timestep to query

    Returns:
        List of unique objectTypes
    """
    object_types = set()
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if state and state.pickupable:
            object_types.add(state.object_type)

    return list(object_types)


@eqa_generator(
    question_id="obj_types_in_container_initial",
    question_type="multi_choice",
    templates=[
        "Initially, which types of objects were inside the {RE_obj}?",
        "At the start, what objects could be found in the {RE_obj}?",
        "Select all object types present in the {RE_obj} at time 0.",
        "Which items did the {RE_obj} contain at the beginning?"
    ],
    description="Object types inside a container at the start.")
def generate_obj_types_in_container_initial(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate object types in container question at t=0."""
    target_timestep = 0
    current_timestep = tracker.current_timestep

    # Get containers with objects
    containers = _get_containers_with_objects(tracker, target_timestep)

    if not containers:
        return None

    # Select random container
    container_id, container_loc_desc, objects_inside = rng.choice(containers)

    # Get unique object types inside container
    object_types_inside = _get_unique_object_types(tracker, objects_inside,
                                                   target_timestep)

    if len(object_types_inside) < 1:
        return None

    # Select X1 correct options (subset of types inside)
    num_correct = min(rng.randint(1, min(3, len(object_types_inside))),
                      len(object_types_inside))
    correct_types = rng.sample(object_types_inside, num_correct)

    # Get all object types in scene as distractor pool
    all_types = _get_all_object_types_in_scene(tracker, target_timestep)

    # Remove correct types from distractors
    distractor_pool = [t for t in all_types if t not in object_types_inside]

    # Select X2 incorrect options
    num_incorrect = min(rng.randint(2, 3), len(distractor_pool))

    if num_incorrect < 2:
        return None

    incorrect_types = select_distractors(correct_types, distractor_pool,
                                         num_incorrect, rng)

    # Combine and shuffle options
    all_types_options = correct_types + incorrect_types
    rng.shuffle(all_types_options)
    options = get_options_list(all_types_options)

    # Find all correct answer labels
    correct_labels = []
    for label, obj_type in options.items():
        if obj_type in correct_types:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)

    # Select random template and format question
    template = rng.choice(generate_obj_types_in_container_initial.templates)
    question_text = template.format(RE_obj=container_loc_desc)

    # Collect evidence
    container_state = tracker.get_object_state_at_step(container_id,
                                                       target_timestep)

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "container": {
            "objectId": container_state.object_id,
            "receptacleObjectIds": container_state.receptacle_object_ids
        },
        "options_info": {}
    }

    # Build options_info for all options
    for label, obj_type in options.items():
        is_correct = (obj_type in correct_types)

        # Find matching object IDs for this type
        matching_ids = []
        if is_correct:
            for obj_id in objects_inside:
                obj_state = tracker.get_object_state_at_step(
                    obj_id, target_timestep)
                if obj_state and obj_state.object_type == obj_type:
                    matching_ids.append(obj_id)

        evidence["options_info"][label] = {
            "objectType": obj_type,
            "is_correct": is_correct,
            "matching_object_ids": matching_ids
        }

    # Create metadata
    metadata = format_question_metadata(
        question_id="obj_types_in_container_initial",
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
    question_id="obj_types_in_container_current",
    question_type="multi_choice",
    templates=[
        "Which types of objects are currently inside the {RE_obj}?",
        "Right now, what objects can be found in the {RE_obj}?",
        "Select all object types present in the {RE_obj} at this moment.",
        "Which items does the {RE_obj} contain currently?"
    ],
    description="Object types inside a container now.")
def generate_obj_types_in_container_current(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate object types in container question at current moment."""
    target_timestep = tracker.current_timestep
    current_timestep = tracker.current_timestep

    # Get containers with objects
    containers = _get_containers_with_objects(tracker, target_timestep)

    if not containers:
        return None

    # Select random container
    container_id, container_loc_desc, objects_inside = rng.choice(containers)

    # Get unique object types inside container
    object_types_inside = _get_unique_object_types(tracker, objects_inside,
                                                   target_timestep)

    if len(object_types_inside) < 1:
        return None

    # Select X1 correct options
    num_correct = min(rng.randint(1, min(3, len(object_types_inside))),
                      len(object_types_inside))
    correct_types = rng.sample(object_types_inside, num_correct)

    # Get all object types in scene
    all_types = _get_all_object_types_in_scene(tracker, target_timestep)

    # Remove correct types from distractors
    distractor_pool = [t for t in all_types if t not in object_types_inside]

    # Select X2 incorrect options
    num_incorrect = min(rng.randint(2, 3), len(distractor_pool))

    if num_incorrect < 2:
        return None

    incorrect_types = select_distractors(correct_types, distractor_pool,
                                         num_incorrect, rng)

    # Combine and shuffle options
    all_types_options = correct_types + incorrect_types
    rng.shuffle(all_types_options)
    options = get_options_list(all_types_options)

    # Find all correct answer labels
    correct_labels = []
    for label, obj_type in options.items():
        if obj_type in correct_types:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)

    # Select random template and format question
    template = rng.choice(generate_obj_types_in_container_current.templates)
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
        "options_info": {}
    }

    # Build options_info
    for label, obj_type in options.items():
        is_correct = (obj_type in correct_types)

        matching_ids = []
        if is_correct:
            for obj_id in objects_inside:
                obj_state = tracker.get_object_state_at_step(
                    obj_id, target_timestep)
                if obj_state and obj_state.object_type == obj_type:
                    matching_ids.append(obj_id)

        evidence["options_info"][label] = {
            "objectType": obj_type,
            "is_correct": is_correct,
            "matching_object_ids": matching_ids
        }

    # Create metadata
    metadata = format_question_metadata(
        question_id="obj_types_in_container_current",
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
    question_id="obj_types_in_container_n_steps_ago",
    question_type="multi_choice",
    templates=[
        "{n} steps ago, which types of objects were inside the {RE_obj}?",
        "{n} steps ago, what objects could be found in the {RE_obj}?",
        "Select all object types present in the {RE_obj} {n} steps back.",
        "Which items did the {RE_obj} contain {n} steps ago?"
    ],
    description="Object types inside a container n steps ago.")
def generate_obj_types_in_container_n_steps_ago(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate object types in container question n steps ago."""
    current_timestep = tracker.current_timestep

    # Need at least 2 timesteps of history
    if current_timestep < 2:
        return None

    # Randomly select n
    n = rng.randint(1, current_timestep - 1)
    target_timestep = current_timestep - n

    # Get containers with objects
    containers = _get_containers_with_objects(tracker, target_timestep)

    if not containers:
        return None

    # Select random container
    container_id, container_loc_desc, objects_inside = rng.choice(containers)

    # Get unique object types inside container
    object_types_inside = _get_unique_object_types(tracker, objects_inside,
                                                   target_timestep)

    if len(object_types_inside) < 1:
        return None

    # Select X1 correct options
    num_correct = min(rng.randint(1, min(3, len(object_types_inside))),
                      len(object_types_inside))
    correct_types = rng.sample(object_types_inside, num_correct)

    # Get all object types in scene
    all_types = _get_all_object_types_in_scene(tracker, target_timestep)

    # Remove correct types from distractors
    distractor_pool = [t for t in all_types if t not in object_types_inside]

    # Select X2 incorrect options
    num_incorrect = min(rng.randint(2, 3), len(distractor_pool))

    if num_incorrect < 2:
        return None

    incorrect_types = select_distractors(correct_types, distractor_pool,
                                         num_incorrect, rng)

    # Combine and shuffle options
    all_types_options = correct_types + incorrect_types
    rng.shuffle(all_types_options)
    options = get_options_list(all_types_options)

    # Find all correct answer labels
    correct_labels = []
    for label, obj_type in options.items():
        if obj_type in correct_types:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)

    # Select random template and format question
    template = rng.choice(
        generate_obj_types_in_container_n_steps_ago.templates)
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
        "options_info": {}
    }

    # Build options_info
    for label, obj_type in options.items():
        is_correct = (obj_type in correct_types)

        matching_ids = []
        if is_correct:
            for obj_id in objects_inside:
                obj_state = tracker.get_object_state_at_step(
                    obj_id, target_timestep)
                if obj_state and obj_state.object_type == obj_type:
                    matching_ids.append(obj_id)

        evidence["options_info"][label] = {
            "objectType": obj_type,
            "is_correct": is_correct,
            "matching_object_ids": matching_ids
        }

    # Create metadata
    metadata = format_question_metadata(
        question_id="obj_types_in_container_n_steps_ago",
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
    question_id="obj_types_in_container_random_t",
    question_type="multi_choice",
    templates=[
        "At time {t}, which types of objects were inside the {RE_obj}?",
        "At time {t}, what objects could be found in the {RE_obj}?",
        "Select all object types present in the {RE_obj} at time {t}.",
        "Which items did the {RE_obj} contain at time {t}?"
    ],
    description="Object types inside a container at random time t.")
def generate_obj_types_in_container_random_t(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate object types in container question at random time t."""
    current_timestep = tracker.current_timestep

    # Randomly select t
    t = rng.randint(0, current_timestep)
    target_timestep = t

    # Get containers with objects
    containers = _get_containers_with_objects(tracker, target_timestep)

    if not containers:
        return None

    # Select random container
    container_id, container_loc_desc, objects_inside = rng.choice(containers)

    # Get unique object types inside container
    object_types_inside = _get_unique_object_types(tracker, objects_inside,
                                                   target_timestep)

    if len(object_types_inside) < 1:
        return None

    # Select X1 correct options
    num_correct = min(rng.randint(1, min(3, len(object_types_inside))),
                      len(object_types_inside))
    correct_types = rng.sample(object_types_inside, num_correct)

    # Get all object types in scene
    all_types = _get_all_object_types_in_scene(tracker, target_timestep)

    # Remove correct types from distractors
    distractor_pool = [t for t in all_types if t not in object_types_inside]

    # Select X2 incorrect options
    num_incorrect = min(rng.randint(2, 3), len(distractor_pool))

    if num_incorrect < 2:
        return None

    incorrect_types = select_distractors(correct_types, distractor_pool,
                                         num_incorrect, rng)

    # Combine and shuffle options
    all_types_options = correct_types + incorrect_types
    rng.shuffle(all_types_options)
    options = get_options_list(all_types_options)

    # Find all correct answer labels
    correct_labels = []
    for label, obj_type in options.items():
        if obj_type in correct_types:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)

    # Select random template and format question
    template = rng.choice(generate_obj_types_in_container_random_t.templates)
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
        "options_info": {}
    }

    # Build options_info
    for label, obj_type in options.items():
        is_correct = (obj_type in correct_types)

        matching_ids = []
        if is_correct:
            for obj_id in objects_inside:
                obj_state = tracker.get_object_state_at_step(
                    obj_id, target_timestep)
                if obj_state and obj_state.object_type == obj_type:
                    matching_ids.append(obj_id)

        evidence["options_info"][label] = {
            "objectType": obj_type,
            "is_correct": is_correct,
            "matching_object_ids": matching_ids
        }

    # Create metadata
    metadata = format_question_metadata(
        question_id="obj_types_in_container_random_t",
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
