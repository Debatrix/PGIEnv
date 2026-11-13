"""
Object Location Question Generators

Generates questions about object locations at different temporal points:
- Initial moment (t=0)
- Current moment
- N steps ago
- Random time t
"""

import random
from typing import Dict, List, Optional, Any
from types import ModuleType

from pgibench.envs.pgi_alfred.eqa.thor_tracker import ThorTracker
from pgibench.envs.pgi_alfred.eqa.utils import (
    get_options_list, select_distractors, has_sequential_number,
    format_question_metadata, create_question_dict, get_object_loc_desc)


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
    parents = set()
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if state and state.receptacle and obj_id in tracker.historically_seen_objects:
            # obj_name = state.object_type
            # parents.add(obj_name)
            loc_desc = get_object_loc_desc(state)
            parents.add(loc_desc)
    return list(parents)


def _get_objects_with_parents(tracker: ThorTracker,
                              timestep: int) -> List[tuple]:
    """
    Get objects that have parent receptacles at a given timestep.

    Args:
        tracker: ThorTracker instance
        timestep: Timestep to query

    Returns:
        List of (object_id, object_type, locDesc, parent_id, parent_locDesc) tuples
    """
    objects_with_parents = []
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if state and state.parent_receptacles and obj_id in tracker.historically_seen_objects:
            # Use objectType for object name
            obj_name = state.object_type
            obj_loc_desc = get_object_loc_desc(state)

            # Get parent locDesc
            parent_id = state.parent_receptacles[0]  # Use first parent
            parent_state = tracker.get_object_state_at_step(
                parent_id, timestep)
            if parent_state:
                parent_loc_desc = get_object_loc_desc(parent_state)
                if parent_loc_desc and not has_sequential_number(
                        parent_loc_desc
                ) and parent_id in tracker.historically_seen_objects:
                    objects_with_parents.append(
                        (obj_id, obj_name, obj_loc_desc, parent_id,
                         parent_loc_desc))

    return objects_with_parents


@eqa_generator(question_id="obj_loc_initial",
               question_type="single_choice",
               templates=[
                   "Where was the {obj} at the start of the task?",
                   "Initially, which place contained the {obj}?",
                   "When the episode began, where could you find the {obj}?",
                   "What was the initial location of the {obj}?"
               ],
               description="Object location at the initial moment (t=0).")
def generate_obj_loc_initial(tracker: ThorTracker, constants: ModuleType,
                             timestep: int,
                             rng: random.Random) -> Optional[Dict[str, Any]]:
    """
    Generate object location at initial moment question.

    Variables:
        <obj>: objectType
        <RE_obj_t>: locDesc (at t=0)
        <RE_obj_f>: list of locDesc (distractors)
        <X>: int (number of distractors)

    Args:
        tracker: ThorTracker instance
        constants: Constants module
        timestep: Current timestep (ignored, uses t=0)
        rng: Random instance for reproducibility

    Returns:
        Question dictionary or None if generation failed
    """
    # Get objects with parents at t=0
    objects_with_parents = _get_objects_with_parents(tracker, 0)

    if not objects_with_parents:
        return None

    # Select random object
    obj_id, obj_type, _, parent_id, parent_loc_desc = rng.choice(
        objects_with_parents)

    # Get all possible parent receptacles as distractors
    all_parents = _get_all_parent_receptacles(tracker, 0)

    # Remove correct parent from distractors
    distractor_pool = [p for p in all_parents if p != parent_loc_desc]

    # Determine number of distractors (at least 3, at most available)
    num_distractors = min(rng.randint(3, 5), len(distractor_pool))

    if num_distractors < 3:
        return None

    # Select distractors
    distractors = select_distractors([parent_loc_desc], distractor_pool,
                                     num_distractors, rng)

    # Shuffle and create options
    all_options = [parent_loc_desc] + distractors
    rng.shuffle(all_options)
    options = get_options_list(all_options)

    # Find correct answer label
    for label, text in options.items():
        if text == parent_loc_desc:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_loc_initial.templates)
    question_text = template.format(obj=obj_type.lower())

    # Collect evidence - only relevant information
    query_obj_state = tracker.get_object_state_at_step(obj_id, 0)

    evidence = {
        "question_timestep": 0,
        "current_timestep": tracker.current_timestep,
        "query_object": {
            "objectId":
            obj_id,
            "parentReceptacles":
            query_obj_state.parent_receptacles if query_obj_state else []
        },
        "options_info": {}
    }

    # Build options_info for all options
    for label, option_text in options.items():
        is_correct = (label == correct_answer)

        option_obj_id = None
        for test_obj_id in tracker.all_object_ids:
            test_state = tracker.get_object_state_at_step(test_obj_id, 0)
            if test_state:
                test_loc_desc = get_object_loc_desc(test_state)
                if test_loc_desc == option_text:
                    option_obj_id = test_obj_id
                    break

        # Prove this distractor doesn't contain the query object
        option_state = tracker.get_object_state_at_step(option_obj_id, 0)
        evidence["options_info"][label] = {
            "option_text": option_text,
            "is_correct": is_correct,
            "objectId": option_obj_id,
            "receptacleObjectIds": option_state.receptacle_object_ids
        }

    # Create metadata (simplified structure)
    metadata = format_question_metadata(
        question_id="obj_loc_initial",
        question_type="single_choice",
        question_timestep=0,
        current_timestep=tracker.current_timestep,
        template_used=template)

    return create_question_dict(
        question_id="",  # Will be set by EQAGenerator
        question_text=question_text,
        options=options,
        correct_answer=correct_answer,
        metadata=metadata,
        evidence=evidence)


@eqa_generator(question_id="obj_loc_current",
               question_type="single_choice",
               templates=[
                   "Where is the {obj} right now?",
                   "Currently, where is the {obj} in?",
                   "At this moment, where would you find the {obj}?",
                   "What is the location of the {obj} at the current time?"
               ],
               description="Object location at the current moment.")
def generate_obj_loc_current(tracker: ThorTracker, constants: ModuleType,
                             timestep: int,
                             rng: random.Random) -> Optional[Dict[str, Any]]:
    """
    Generate object location at current moment question.

    Variables:
        <obj>: objectType
        <RE_obj_t>: locDesc (at current)
        <RE_obj_f>: list of locDesc (distractors)
        <X>: int

    Args:
        tracker: ThorTracker instance
        constants: Constants module
        timestep: Current timestep
        rng: Random instance

    Returns:
        Question dictionary or None if generation failed
    """
    # Use current timestep from tracker
    current_timestep = tracker.current_timestep

    # Get objects with parents at current timestep
    objects_with_parents = _get_objects_with_parents(tracker, current_timestep)

    if not objects_with_parents:
        return None

    # Select random object
    obj_id, obj_type, _, parent_id, parent_loc_desc = rng.choice(
        objects_with_parents)

    # Get all possible parent receptacles as distractors
    all_parents = _get_all_parent_receptacles(tracker, current_timestep)

    # Remove correct parent from distractors
    distractor_pool = [p for p in all_parents if p != parent_loc_desc]

    # Determine number of distractors
    num_distractors = min(rng.randint(3, 5), len(distractor_pool))

    if num_distractors < 3:
        return None

    # Select distractors
    distractors = select_distractors([parent_loc_desc], distractor_pool,
                                     num_distractors, rng)

    # Shuffle and create options
    all_options = [parent_loc_desc] + distractors
    rng.shuffle(all_options)
    options = get_options_list(all_options)

    # Find correct answer label
    for label, text in options.items():
        if text == parent_loc_desc:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_loc_current.templates)
    question_text = template.format(obj=obj_type.lower())

    # Collect evidence - only relevant information
    query_obj_state = tracker.get_object_state_at_step(obj_id,
                                                       current_timestep)
    parent_obj_state = tracker.get_object_state_at_step(
        parent_id, current_timestep)

    evidence = {
        "question_timestep": current_timestep,
        "current_timestep": current_timestep,
        "query_object": {
            "objectId":
            obj_id,
            "parentReceptacles":
            query_obj_state.parent_receptacles if query_obj_state else []
        },
        "options_info": {}
    }

    # Build options_info for all options
    for label, option_text in options.items():
        is_correct = (label == correct_answer)
        option_obj_id = None
        for test_obj_id in tracker.all_object_ids:
            test_state = tracker.get_object_state_at_step(
                test_obj_id, current_timestep)
            if test_state:
                test_loc_desc = get_object_loc_desc(test_state)
                if test_loc_desc == option_text:
                    option_obj_id = test_obj_id
                    break

        option_state = tracker.get_object_state_at_step(
            option_obj_id, current_timestep)
        evidence["options_info"][label] = {
            "option_text": option_text,
            "is_correct": is_correct,
            "objectId": option_obj_id,
            "receptacleObjectIds": option_state.receptacle_object_ids
        }

    # Create metadata (simplified structure)
    metadata = format_question_metadata(question_id="obj_loc_current",
                                        question_type="single_choice",
                                        question_timestep=current_timestep,
                                        current_timestep=current_timestep,
                                        template_used=template)

    return create_question_dict(question_id="",
                                question_text=question_text,
                                options=options,
                                correct_answer=correct_answer,
                                metadata=metadata,
                                evidence=evidence)


@eqa_generator(
    question_id="obj_loc_n_steps_ago",
    question_type="single_choice",
    templates=[
        "Where was the {obj} {n} steps ago?",
        "At {n} steps ago, where was the {obj} located?",
        "What was the position of the {obj} {n} steps back?",
    ],
    description="Object location N steps before the current moment.")
def generate_obj_loc_n_steps_ago(
        tracker: ThorTracker, constants: ModuleType, timestep: int,
        rng: random.Random) -> Optional[Dict[str, Any]]:
    """
    Generate object location N steps ago question.

    Variables:
        <n>: int
        <obj>: objectType
        <RE_obj_t>: locDesc (at t-n)
        <RE_obj_f>: list of locDesc (distractors)
        <X>: int

    Args:
        tracker: ThorTracker instance
        constants: Constants module
        timestep: Current timestep
        rng: Random instance

    Returns:
        Question dictionary or None if generation failed
    """
    current_timestep = tracker.current_timestep

    # Need at least 2 timesteps of history
    if current_timestep < 2:
        return None

    # Randomly select n (1 to current_timestep - 1)
    n = rng.randint(1, current_timestep - 1)
    target_timestep = current_timestep - n

    # Get objects with parents at target timestep
    objects_with_parents = _get_objects_with_parents(tracker, target_timestep)

    if not objects_with_parents:
        return None

    # Select random object
    obj_id, obj_type, _, parent_id, parent_loc_desc = rng.choice(
        objects_with_parents)

    # Get all possible parent receptacles as distractors
    all_parents = _get_all_parent_receptacles(tracker, target_timestep)

    # Remove correct parent from distractors
    distractor_pool = [p for p in all_parents if p != parent_loc_desc]

    # Determine number of distractors
    num_distractors = min(rng.randint(3, 5), len(distractor_pool))

    if num_distractors < 3:
        return None

    # Select distractors
    distractors = select_distractors([parent_loc_desc], distractor_pool,
                                     num_distractors, rng)

    # Shuffle and create options
    all_options = [parent_loc_desc] + distractors
    rng.shuffle(all_options)
    options = get_options_list(all_options)

    # Find correct answer label
    for label, text in options.items():
        if text == parent_loc_desc:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_loc_n_steps_ago.templates)
    question_text = template.format(obj=obj_type.lower(), n=n)

    # Collect evidence - only relevant information
    query_obj_state = tracker.get_object_state_at_step(obj_id, target_timestep)
    parent_obj_state = tracker.get_object_state_at_step(
        parent_id, target_timestep)

    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "query_object": {
            "objectId":
            obj_id,
            "parentReceptacles":
            query_obj_state.parent_receptacles if query_obj_state else []
        },
        "options_info": {}
    }

    # Build options_info for all options
    for label, option_text in options.items():
        is_correct = (label == correct_answer)

        option_obj_id = None
        for test_obj_id in tracker.all_object_ids:
            test_state = tracker.get_object_state_at_step(
                test_obj_id, target_timestep)
            if test_state:
                test_loc_desc = get_object_loc_desc(test_state)
                if test_loc_desc == option_text:
                    option_obj_id = test_obj_id
                    break

        option_state = tracker.get_object_state_at_step(
            option_obj_id, target_timestep)
        evidence["options_info"][label] = {
            "option_text": option_text,
            "is_correct": is_correct,
            "objectId": option_obj_id,
            "receptacleObjectIds": option_state.receptacle_object_ids
        }

    # Create metadata (simplified structure)
    metadata = format_question_metadata(question_id="obj_loc_n_steps_ago",
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
    question_id="obj_loc_random_t",
    question_type="single_choice",
    templates=[
        "Where was the {obj} at time {t}?",
        "Which place contained the {obj} at time {t}?",
        "At time {t}, where could you find the {obj}?",
        "What was the location of the {obj} at step {t}?"
    ],
    description="Object location at a random time t (0 <= t <= current).")
def generate_obj_loc_random_t(tracker: ThorTracker, constants: ModuleType,
                              timestep: int,
                              rng: random.Random) -> Optional[Dict[str, Any]]:
    """
    Generate object location at random time t question.

    Variables:
        <t>: int
        <obj>: objectType
        <RE_obj_t>: locDesc (at t)
        <RE_obj_f>: list of locDesc (distractors)
        <X>: int

    Args:
        tracker: ThorTracker instance
        constants: Constants module
        timestep: Current timestep (not used, selects random t)
        rng: Random instance

    Returns:
        Question dictionary or None if generation failed
    """
    current_timestep = tracker.current_timestep

    # Randomly select t from valid timesteps
    t = rng.randint(0, current_timestep)

    # Get objects with parents at timestep t
    objects_with_parents = _get_objects_with_parents(tracker, t)

    if not objects_with_parents:
        return None

    # Select random object
    obj_id, obj_type, _, parent_id, parent_loc_desc = rng.choice(
        objects_with_parents)

    # Get all possible parent receptacles as distractors
    all_parents = _get_all_parent_receptacles(tracker, t)

    # Remove correct parent from distractors
    distractor_pool = [p for p in all_parents if p != parent_loc_desc]

    # Determine number of distractors
    num_distractors = min(rng.randint(3, 5), len(distractor_pool))

    if num_distractors < 3:
        return None

    # Select distractors
    distractors = select_distractors([parent_loc_desc], distractor_pool,
                                     num_distractors, rng)

    # Shuffle and create options
    all_options = [parent_loc_desc] + distractors
    rng.shuffle(all_options)
    options = get_options_list(all_options)

    # Find correct answer label
    for label, text in options.items():
        if text == parent_loc_desc:
            correct_answer = label
            break

    # Select random template and format question
    template = rng.choice(generate_obj_loc_random_t.templates)
    question_text = template.format(obj=obj_type.lower(), t=t)

    # Collect evidence - only relevant information
    query_obj_state = tracker.get_object_state_at_step(obj_id, t)

    evidence = {
        "question_timestep": t,
        "current_timestep": current_timestep,
        "query_object": {
            "objectId":
            obj_id,
            "parentReceptacles":
            query_obj_state.parent_receptacles if query_obj_state else []
        },
        "options_info": {}
    }

    # Build options_info for all options
    for label, option_text in options.items():
        is_correct = (label == correct_answer)

        option_obj_id = None
        for test_obj_id in tracker.all_object_ids:
            test_state = tracker.get_object_state_at_step(test_obj_id, t)
            if test_state:
                test_loc_desc = get_object_loc_desc(test_state)
                if test_loc_desc == option_text:
                    option_obj_id = test_obj_id
                    break

        option_state = tracker.get_object_state_at_step(option_obj_id, t)
        evidence["options_info"][label] = {
            "option_text": option_text,
            "is_correct": is_correct,
            "objectId": option_obj_id,
            "receptacleObjectIds": option_state.receptacle_object_ids
        }

    # Create metadata (simplified structure)
    metadata = format_question_metadata(question_id="obj_loc_random_t",
                                        question_type="single_choice",
                                        question_timestep=t,
                                        current_timestep=current_timestep,
                                        template_used=template)

    return create_question_dict(question_id="",
                                question_text=question_text,
                                options=options,
                                correct_answer=correct_answer,
                                metadata=metadata,
                                evidence=evidence)
