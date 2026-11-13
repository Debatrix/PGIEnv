"""
Object Attributes and State Question Generators

Generates questions about object properties (OPEN/CLOSED, ON/OFF, etc.)
at different temporal points.
"""

import random
from typing import Dict, List, Optional, Any
from types import ModuleType

from pgibench.envs.pgi_alfred.eqa.thor_tracker import ThorTracker
from pgibench.envs.pgi_alfred.eqa.utils import (
    get_options_list, select_distractors, has_sequential_number,
    format_question_metadata, create_question_dict, get_object_loc_desc,
    get_object_property_state, get_seen_objects)


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


def get_attribute_name_for_state(state_value: str) -> str:
    """
    Map property state value to actual ObjectState attribute name.

    Args:
        state_value: State value like 'OPEN', 'ON', 'SLICED', 'DIRTY'

    Returns:
        Actual ObjectState attribute name (snake_case)
    """
    mapping = {
        'OPEN': 'is_open',
        'CLOSED': 'is_open',
        'ON': 'is_toggled_on',
        'OFF': 'is_toggled_on',
        'SLICED': 'is_sliced',
        'INTACT': 'is_sliced',
        'DIRTY': 'is_dirty',
        'CLEAN': 'is_dirty'
    }
    return mapping.get(state_value, state_value.lower())


@eqa_generator(question_id="obj_property_initial",
               question_type="single_choice",
               templates=[
                   "Initially, what was the state of {obj}?",
                   "What was the status of {obj} at the beginning?",
                   "Describe the state of {obj} at the initial moment."
               ],
               description="Object property state at t=0.")
def generate_obj_property_initial(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate property state question at t=0."""
    target_timestep = 0
    current_timestep = tracker.current_timestep

    # Find objects with relevant properties
    candidates = []
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, target_timestep)
        if state and obj_id in tracker.historically_seen_objects:
            prop = get_object_property_state(state)
            if prop:
                obj_loc_desc = get_object_loc_desc(state)
                if obj_loc_desc and not has_sequential_number(obj_loc_desc):
                    candidates.append((obj_id, obj_loc_desc, prop))

    if not candidates:
        return None

    # Select one object
    obj_id, obj_loc_desc, prop_info = rng.choice(candidates)

    # Generate options based on property type
    correct_state = prop_info['state']

    if prop_info['property_type'] == 'container':
        distractors = ['CLOSED'] if correct_state == 'OPEN' else ['OPEN']
    elif prop_info['property_type'] == 'toggleable':
        distractors = ['OFF'] if correct_state == 'ON' else ['ON']
    elif prop_info['property_type'] == 'sliceable':
        distractors = ['INTACT'] if correct_state == 'SLICED' else ['SLICED']
    elif prop_info['property_type'] == 'dirtyable':
        distractors = ['CLEAN'] if correct_state == 'DIRTY' else ['DIRTY']
    else:
        return None

    # Create options
    all_options = [correct_state] + distractors
    rng.shuffle(all_options)
    options = get_options_list(all_options)

    # Find correct answer label
    correct_answer = None
    for label, state_name in options.items():
        if state_name == correct_state:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_property_initial.templates)
    question_text = template.format(obj=obj_loc_desc)

    # Collect evidence
    obj_state = tracker.get_object_state_at_step(obj_id, target_timestep)

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "object": {
            "objectId":
            obj_id,
            get_attribute_name_for_state(prop_info['state']):
            getattr(obj_state,
                    get_attribute_name_for_state(prop_info['state']), False)
        },
    }

    # Create metadata
    metadata = format_question_metadata(question_id="obj_property_initial",
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


@eqa_generator(question_id="obj_property_current",
               question_type="single_choice",
               templates=[
                   "What is the state of {obj} right now?",
                   "What is the current status of {obj}?",
                   "Describe the state of {obj} at the current time."
               ],
               description="Object property state now.")
def generate_obj_property_current(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate property state question at current moment."""
    target_timestep = tracker.current_timestep
    current_timestep = tracker.current_timestep

    # Find objects with relevant properties
    candidates = []
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, target_timestep)
        if state and obj_id in tracker.historically_seen_objects:
            prop = get_object_property_state(state)
            if prop:
                obj_loc_desc = get_object_loc_desc(state)
                if obj_loc_desc and not has_sequential_number(obj_loc_desc):
                    candidates.append((obj_id, obj_loc_desc, prop))

    if not candidates:
        return None

    # Select one object
    obj_id, obj_loc_desc, prop_info = rng.choice(candidates)

    # Generate options
    correct_state = prop_info['state']

    if prop_info['property_type'] == 'container':
        distractors = ['CLOSED'] if correct_state == 'OPEN' else ['OPEN']
    elif prop_info['property_type'] == 'toggleable':
        distractors = ['OFF'] if correct_state == 'ON' else ['ON']
    elif prop_info['property_type'] == 'sliceable':
        distractors = ['INTACT'] if correct_state == 'SLICED' else ['SLICED']
    elif prop_info['property_type'] == 'dirtyable':
        distractors = ['CLEAN'] if correct_state == 'DIRTY' else ['DIRTY']
    else:
        return None

    # Create options
    all_options = [correct_state] + distractors
    rng.shuffle(all_options)
    options = get_options_list(all_options)

    # Find correct answer label
    correct_answer = None
    for label, state_name in options.items():
        if state_name == correct_state:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_property_current.templates)
    question_text = template.format(obj=obj_loc_desc)

    # Collect evidence
    obj_state = tracker.get_object_state_at_step(obj_id, target_timestep)

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "object": {
            "objectId":
            obj_id,
            get_attribute_name_for_state(prop_info['state']):
            getattr(obj_state,
                    get_attribute_name_for_state(prop_info['state']), False)
        },
    }

    # Create metadata
    metadata = format_question_metadata(question_id="obj_property_current",
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


@eqa_generator(question_id="obj_property_n_steps_ago",
               question_type="single_choice",
               templates=[
                   "What was the state of {obj} {n} steps ago?",
                   "{n} steps ago, what was the status of {obj}",
                   "Describe the state of {obj} {n} steps before now."
               ],
               description="Object property state n steps ago.")
def generate_obj_property_n_steps_ago(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate property state question n steps ago."""
    current_timestep = tracker.current_timestep

    # Need at least 2 timesteps of history
    if current_timestep < 2:
        return None

    # Randomly select n
    n = rng.randint(1, current_timestep - 1)
    target_timestep = current_timestep - n

    # Find objects with relevant properties
    candidates = []
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, target_timestep)
        if state and obj_id in tracker.historically_seen_objects:
            prop = get_object_property_state(state)
            if prop:
                obj_loc_desc = get_object_loc_desc(state)
                if obj_loc_desc and not has_sequential_number(obj_loc_desc):
                    candidates.append((obj_id, obj_loc_desc, prop))

    if not candidates:
        return None

    # Select one object
    obj_id, obj_loc_desc, prop_info = rng.choice(candidates)

    # Generate options
    correct_state = prop_info['state']

    if prop_info['property_type'] == 'container':
        distractors = ['CLOSED'] if correct_state == 'OPEN' else ['OPEN']
    elif prop_info['property_type'] == 'toggleable':
        distractors = ['OFF'] if correct_state == 'ON' else ['ON']
    elif prop_info['property_type'] == 'sliceable':
        distractors = ['INTACT'] if correct_state == 'SLICED' else ['SLICED']
    elif prop_info['property_type'] == 'dirtyable':
        distractors = ['CLEAN'] if correct_state == 'DIRTY' else ['DIRTY']
    else:
        return None

    # Create options
    all_options = [correct_state] + distractors
    rng.shuffle(all_options)
    options = get_options_list(all_options)

    # Find correct answer label
    correct_answer = None
    for label, state_name in options.items():
        if state_name == correct_state:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_property_n_steps_ago.templates)
    question_text = template.format(n=n, obj=obj_loc_desc)

    # Collect evidence
    obj_state = tracker.get_object_state_at_step(obj_id, target_timestep)

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "object": {
            "objectId":
            obj_id,
            get_attribute_name_for_state(prop_info['state']):
            getattr(obj_state,
                    get_attribute_name_for_state(prop_info['state']), False)
        },
    }

    # Create metadata
    metadata = format_question_metadata(question_id="obj_property_n_steps_ago",
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


@eqa_generator(question_id="obj_property_random_t",
               question_type="single_choice",
               templates=[
                   "At time {t}, what was the state of {obj}?",
                   "What was the status of {obj} at step {t}?",
                   "Describe the state of {obj} at time {t}."
               ],
               description="Object property state at random time t.")
def generate_obj_property_random_t(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate property state question at random time t."""
    current_timestep = tracker.current_timestep

    # Randomly select t
    t = rng.randint(0, current_timestep)
    target_timestep = t

    # Find objects with relevant properties
    candidates = []
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, target_timestep)
        if state and obj_id in tracker.historically_seen_objects:
            prop = get_object_property_state(state)
            if prop:
                obj_loc_desc = get_object_loc_desc(state)
                if obj_loc_desc and not has_sequential_number(obj_loc_desc):
                    candidates.append((obj_id, obj_loc_desc, prop))

    if not candidates:
        return None

    # Select one object
    obj_id, obj_loc_desc, prop_info = rng.choice(candidates)

    # Generate options
    correct_state = prop_info['state']

    if prop_info['property_type'] == 'container':
        distractors = ['CLOSED'] if correct_state == 'OPEN' else ['OPEN']
    elif prop_info['property_type'] == 'toggleable':
        distractors = ['OFF'] if correct_state == 'ON' else ['ON']
    elif prop_info['property_type'] == 'sliceable':
        distractors = ['INTACT'] if correct_state == 'SLICED' else ['SLICED']
    elif prop_info['property_type'] == 'dirtyable':
        distractors = ['CLEAN'] if correct_state == 'DIRTY' else ['DIRTY']
    else:
        return None

    # Create options
    all_options = [correct_state] + distractors
    rng.shuffle(all_options)
    options = get_options_list(all_options)

    # Find correct answer label
    correct_answer = None
    for label, state_name in options.items():
        if state_name == correct_state:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_property_random_t.templates)
    question_text = template.format(t=t, obj=obj_loc_desc)

    # Collect evidence
    obj_state = tracker.get_object_state_at_step(obj_id, target_timestep)

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "object": {
            "objectId":
            obj_id,
            get_attribute_name_for_state(prop_info['state']):
            getattr(obj_state,
                    get_attribute_name_for_state(prop_info['state']), False)
        },
    }

    # Create metadata
    metadata = format_question_metadata(question_id="obj_property_random_t",
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
    question_id="obj_existence_seen",
    question_type="multi_choice",
    templates=[
        "Select all types of objects that you have seen so far.",
        "Which of these object types have you observed during this episode?",
        "Identify all object types that have been visible to you.",
        "Choose all object types that you have encountered."
    ],
    description="Object types seen by agent (multi-choice).")
def generate_obj_existence_seen(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """Generate multi-choice question about object types the agent has seen."""
    current_timestep = tracker.current_timestep
    target_timestep = current_timestep

    # Get objects that have been seen by the agent
    seen_object_ids = get_seen_objects(tracker, current_timestep)

    if not seen_object_ids:
        return None

    # Get unique object types from seen objects
    seen_types = set()
    seen_type_to_ids = {}  # Map type to example object IDs
    for obj_id in seen_object_ids:
        state = tracker.get_object_state_at_step(obj_id, current_timestep)
        if state:
            obj_type = state.object_type
            seen_types.add(obj_type)
            if obj_type not in seen_type_to_ids:
                seen_type_to_ids[obj_type] = []
            seen_type_to_ids[obj_type].append(obj_id)

    seen_types = list(seen_types)

    # Get all object types currently in scene
    current_scene_types = set()
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, current_timestep)
        if state:
            current_scene_types.add(state.object_type)

    # Get all object types from constants.py (all possible object types)
    all_possible_types = constants.OBJECTS

    # Find types that are NOT in current scene (for distractors)
    distractor_pool = [
        t for t in all_possible_types if t not in current_scene_types
    ]

    if not distractor_pool:
        return None

    # Select X1 correct options (1-3 seen object types)
    num_correct = min(rng.randint(1, min(3, len(seen_types))), len(seen_types))
    correct_types = rng.sample(seen_types, num_correct)

    # Select X2 incorrect options (2-3 types not in scene)
    num_incorrect = min(rng.randint(2, 3), len(distractor_pool))

    if num_incorrect < 2:
        return None

    incorrect_types = select_distractors(correct_types, distractor_pool,
                                         num_incorrect, rng)

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
    template = rng.choice(generate_obj_existence_seen.templates)
    question_text = template

    # Collect evidence
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "seen_object_ids": seen_object_ids,  # All objects seen by agent
        "options_info": {}
    }

    # Build options_info
    for label, obj_type in options.items():
        is_correct = (obj_type in correct_types)
        example_ids = seen_type_to_ids.get(obj_type, [])

        evidence["options_info"][label] = {
            "objectType": obj_type,
            "objectId": example_ids,
            "is_correct": is_correct,
        }

    # Create metadata
    metadata = format_question_metadata(question_id="obj_existence_seen",
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
