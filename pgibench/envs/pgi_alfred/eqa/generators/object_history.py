"""
Object Location History Question Generator

Generates multi-choice questions about where an object has been placed
throughout the episode (cumulative history).
"""

import random
from typing import Dict, List, Optional, Any
from types import ModuleType

from pgibench.envs.pgi_alfred.eqa.thor_tracker import ThorTracker
from pgibench.envs.pgi_alfred.eqa.utils import (
    get_options_list, select_distractors, has_sequential_number,
    format_question_metadata, create_question_dict, get_object_loc_desc,
    get_object_history)


# Decorator to mark functions as EQA generators
def eqa_generator(question_id: str, question_type: str, templates: List[str],
                  description: str,
                  held_templates: Optional[List[str]] = None):
    """Decorator to mark a function as an EQA question generator."""

    def decorator(func):
        func.is_eqa_generator = True
        func.question_id = question_id
        func.question_type = question_type
        func.templates = templates
        func.held_templates = held_templates or []
        func.description = description
        return func

    return decorator


def _get_all_parent_receptacles(tracker: ThorTracker,
                                timestep: int) -> List[str]:
    """
    Get all parent receptacle locDescs at a given timestep.

    Args:
        tracker: ThorTracker instance
        timestep: Timestep to query

    Returns:
        List of parent receptacle locDescs
    """
    loc2id = {}
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if state and state.receptacle and obj_id in tracker.historically_seen_objects:
            loc_desc = get_object_loc_desc(state)
            loc2id[loc_desc] = obj_id
    return loc2id


def _format_ordinal(n: int) -> str:
    """Format a number as ordinal string (1st, 2nd, 3rd, etc.)."""
    if 11 <= n % 100 <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


@eqa_generator(
    question_id="obj_loc_history",
    question_type="multi_choice",
    templates=[
        "Select all locations where the {obj} has been placed so far.",
        "Throughout the episode, where did the {obj} reside?",
        "Which of these locations have held the {obj} at some point?"
    ],
    held_templates=[
        "Select all locations where the {ordinal} {obj} you picked up has been placed so far.",
        "Throughout the episode, where did the {ordinal} {obj} you picked up reside?",
        "Which of these locations have held the {ordinal} {obj} you picked up at some point?"
    ],
    description="Locations where the object has been (Cumulative History).")
def generate_obj_loc_history(tracker: ThorTracker, constants: ModuleType,
                             timestep: int,
                             rng: random.Random) -> Optional[Dict[str, Any]]:
    """
    Generate multi-choice question about object location history.

    Args:
        tracker: ThorTracker instance
        constants: AlfredHV constants module
        timestep: Current timestep
        rng: Random instance

    Returns:
        Question dictionary or None if generation failed
    """
    current_timestep = tracker.current_timestep

    # Only consider objects the agent has seen and can be unambiguously referenced
    objects_with_history = []
    for obj_id in tracker.historically_seen_objects:
        obj_state = tracker.get_object_state_at_step(obj_id, 0)
        if obj_state and obj_state.parent_receptacles:
            # Get unique parent history
            unique_parents = get_object_history(tracker, obj_id,
                                                current_timestep)

            # Need at least 2 unique locations to make an interesting question
            if len(unique_parents) >= 2:
                obj_name = obj_state.object_type

                # Object must be uniquely identifiable:
                # either the only one of its type, or was held by the agent
                is_unique = tracker.is_unique_object_type(obj_id)
                was_held = tracker.was_object_ever_held(obj_id)

                if is_unique:
                    objects_with_history.append(
                        (obj_id, obj_name, unique_parents, 'unique', None))
                elif was_held:
                    ordinal = tracker.get_hold_ordinal(obj_id)
                    if ordinal is not None:
                        objects_with_history.append(
                            (obj_id, obj_name, unique_parents, 'held',
                             ordinal))

    if not objects_with_history:
        return None

    # Select random object that has moved
    obj_id, obj_type, history, ref_type, ordinal = rng.choice(
        objects_with_history)

    # Get parent locDescs for history
    correct_parent_loc_descs = set()
    for parent_id, t in history:
        parent_state = tracker.get_object_state_at_step(
            parent_id, current_timestep)
        if parent_state:
            parent_loc_desc = get_object_loc_desc(parent_state)
            if parent_loc_desc and not has_sequential_number(parent_loc_desc):
                correct_parent_loc_descs.add(parent_loc_desc)
    correct_parent_loc_descs = list(correct_parent_loc_descs)

    if len(correct_parent_loc_descs) < 1:
        return None

    # Select X1 correct options (subset of history)
    num_correct = min(rng.randint(1, min(3, len(correct_parent_loc_descs))),
                      len(correct_parent_loc_descs))
    correct_options = rng.sample(correct_parent_loc_descs, num_correct)

    # Get all possible parent receptacles as distractors
    loc2id = _get_all_parent_receptacles(tracker, current_timestep)

    # Remove correct parents from distractors
    distractor_pool = [
        p for p in loc2id.keys() if p not in correct_parent_loc_descs
    ]

    # Select X2 incorrect options
    num_incorrect = min(rng.randint(2, 3), len(distractor_pool))

    if num_incorrect < 2:
        return None

    incorrect_options = select_distractors(correct_options, distractor_pool,
                                           num_incorrect, rng)

    # Combine and shuffle options
    all_options = correct_options + incorrect_options
    rng.shuffle(all_options)
    options = get_options_list(all_options)

    # Find all correct answer labels
    correct_labels = []
    for label, text in options.items():
        if text in correct_options:
            correct_labels.append(label)

    correct_answer = ",".join(correct_labels)

    # Select template based on reference type and format question
    if ref_type == 'held' and generate_obj_loc_history.held_templates:
        template = rng.choice(generate_obj_loc_history.held_templates)
        ordinal_str = _format_ordinal(ordinal)
        question_text = template.format(obj=obj_type.lower(),
                                         ordinal=ordinal_str)
    else:
        template = rng.choice(generate_obj_loc_history.templates)
        question_text = template.format(obj=obj_type.lower())

    # Collect evidence
    evidence = {
        "question_timestep": current_timestep,
        "current_timestep": current_timestep,
        "query_object": {
            "objectId": obj_id,
            "parentReceptacles": [loc2id.get(x,x) for x in correct_parent_loc_descs],
            "refType": ref_type,
            "holdOrdinal": ordinal
        },
        "options_info": {}
    }

    # Build options_info for all options
    for label, option_text in options.items():
        is_correct = (option_text in correct_options)

        # Find object ID for this option's locDesc
        option_obj_id = loc2id[option_text]
        proof = []
        for t in range(current_timestep):
            option_state = tracker.get_object_state_at_step(option_obj_id, t)
            if obj_id in option_state.receptacle_object_ids:
                proof.append((t, option_state.receptacle_object_ids))
        evidence["options_info"][label] = {
            "option_text": option_text,
            "is_correct": is_correct,
            "objectId": option_obj_id,
            "receptacleObjectIds": proof
        }

    # Create metadata
    metadata = format_question_metadata(question_id="obj_loc_history",
                                        question_type="multi_choice",
                                        question_timestep=current_timestep,
                                        current_timestep=current_timestep,
                                        template_used=template)

    return create_question_dict(question_id="",
                                question_text=question_text,
                                options=options,
                                correct_answer=correct_answer,
                                metadata=metadata,
                                evidence=evidence)
