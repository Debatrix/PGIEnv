"""
<Topic> Question Generator

Generates <single_choice|multi_choice> questions about <what>.
<Temporal variant note: initial / current / N-steps-ago / random-T / history>.

Stage 1 of compose-eqa-generator: copy this file, fill in the blanks, then
register the same entry in benchmark_design.yaml.
"""

import random
from typing import Dict, List, Optional, Any, Tuple
from types import ModuleType

from pgibench.envs.pgi_alfred.eqa.thor_tracker import ThorTracker
from pgibench.envs.pgi_alfred.eqa.utils import (
    get_options_list, select_distractors, has_sequential_number,
    format_question_metadata, create_question_dict, get_object_loc_desc,
    # add more from utils.py as needed:
    # get_object_history, get_objects_in_container, get_visible_objects_at_step,
    # calculate_relative_direction, get_object_property_state,
    # generate_count_distractors, get_seen_objects, is_top_level_object,
    # is_immovable_object, get_objects_by_type, get_objects_in_direction,
)


# --------------------------------------------------------------------------
# Decorator — copy-pasted per file by design (no shared module).
# Keep the signature identical to other generators; _discover_generators reads
# these attributes.
# --------------------------------------------------------------------------
def eqa_generator(question_id: str,
                  question_type: str,
                  templates: List[str],
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


# --------------------------------------------------------------------------
# Private helpers — prefix with `_`. Reuse within file; across files, copy.
# --------------------------------------------------------------------------
def _get_candidates(tracker: ThorTracker, timestep: int) -> List[Tuple]:
    """
    Collect candidate (obj_id, obj_name, ...) tuples that satisfy this
    generator's preconditions.

    Filter rules of thumb:
    - Iterate `tracker.historically_seen_objects` (agent has observed them).
    - For locDesc fields, drop entries where `has_sequential_number(loc_desc)`.
    - For uniqueness, gate on `tracker.is_unique_object_type(obj_id)` OR
      `tracker.was_object_ever_held(obj_id)` (then reference by hold-ordinal
      via `tracker.get_hold_ordinal(obj_id)`).
    """
    candidates = []
    for obj_id in tracker.historically_seen_objects:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if not state:
            continue
        # ... your filters here ...
        candidates.append((obj_id, state.object_type))
    return candidates


def _build_distractor_pool(tracker: ThorTracker,
                           timestep: int,
                           exclude: List[str]) -> List[str]:
    """
    Build a distractor pool from the scene's actual locDescs / types.
    `exclude` is the list of correct option strings — distractors must not
    overlap. Return [] if pool is too small; the main function then returns None.
    """
    pool = set()
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if not state:
            continue
        loc = get_object_loc_desc(state)
        if loc and loc not in exclude and not has_sequential_number(loc):
            pool.add(loc)
    return list(pool)


# --------------------------------------------------------------------------
# Main generator — one decorated function per question_id.
# For multiple temporal variants of the same concept, copy this whole
# @eqa_generator + def block, change question_id suffix + target_timestep.
#
# Function-name convention: real generators in this codebase use unique
# `generate_<question_id>(...)` names (see object_count.py,
# agent_position.py). The registration key is the decorator's `question_id`,
# NOT the function name — but matching the file's existing style is correct.
# When appending to an existing file, copy that file's naming pattern.
# --------------------------------------------------------------------------
@eqa_generator(
    question_id="<concept>_<variant>",  # e.g. "obj_loc_initial"
    question_type="single_choice",  # or "multi_choice"
    templates=[
        "Where was the {obj} at the start of the task?",
        "Initially, which place contained the {obj}?",
    ],
    # held_templates: only set when the question can reference a held object by
    # ordinal ("the second apple you picked up"). Otherwise drop the arg.
    description="One-line description; mirrored in benchmark_design.yaml.",
)
def generate_<concept>_<variant>(tracker: ThorTracker,
                                 constants: ModuleType,
                                 timestep: int,
                                 rng: random.Random) -> Optional[Dict[str, Any]]:
    """
    Generate one question. Contract:
    - Signature is fixed; caller uses kwargs (tracker=, constants=, ...).
    - Return None when data is insufficient (pool empty, too few candidates).
      Never fabricate options to "make it work".
    - evidence must trace back to tracker state so a grader can verify.
    """
    current_timestep = tracker.current_timestep
    # Pick target_timestep based on variant:
    #   initial → 0
    #   current → current_timestep
    #   n_steps_ago → max(0, current_timestep - n)
    #   random_t   → rng.randint(0, current_timestep)
    target_timestep = 0

    candidates = _get_candidates(tracker, target_timestep)
    if not candidates:
        return None

    obj_id, obj_name = rng.choice(candidates)

    # --- correct option ---
    obj_state = tracker.get_object_state_at_step(obj_id, target_timestep)
    if not obj_state:
        return None
    correct_loc = get_object_loc_desc(obj_state)
    if not correct_loc or has_sequential_number(correct_loc):
        return None

    # --- distractor pool ---
    pool = _build_distractor_pool(tracker, target_timestep, exclude=[correct_loc])
    num_distractors = min(3, len(pool))  # single_choice wants >= 3
    if num_distractors < 3:
        return None
    distractors = select_distractors([correct_loc], pool, num_distractors, rng)

    # --- assemble options ---
    all_options = [correct_loc] + distractors
    rng.shuffle(all_options)
    options = get_options_list(all_options)

    correct_answer = next(label for label, text in options.items()
                          if text == correct_loc)

    # --- question text ---
    template = rng.choice(generate.templates)
    question_text = template.format(obj=obj_name.lower())

    # --- evidence (for grading) ---
    evidence = {
        "question_timestep": target_timestep,
        "current_timestep": current_timestep,
        "query_object": {
            "objectId": obj_id,
            "objectType": obj_name,
            "parentReceptacles": obj_state.parent_receptacles,
        },
        # For multi_choice: include "options_info" dict keyed by label with
        # per-option proof (see references/examples.md pattern 2).
    }

    metadata = format_question_metadata(
        question_id="<concept>_<variant>",
        question_type="single_choice",
        question_timestep=target_timestep,
        current_timestep=current_timestep,
        template_used=template,
    )

    return create_question_dict(question_id="",
                                question_text=question_text,
                                options=options,
                                correct_answer=correct_answer,
                                metadata=metadata,
                                evidence=evidence)
