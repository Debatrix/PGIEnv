"""
Utility functions for EQA generation.
"""

import re
import math
import string
from typing import List, Dict, Any, Optional
import random


def get_options_list(items: List[str],
                     labels: str = "ABCDEFG") -> Dict[str, str]:
    """
    Convert a list of items to a dictionary with letter labels.

    Args:
        items: List of option strings
        labels: String of label characters to use (default: "ABCDEFG")

    Returns:
        Dictionary mapping labels to items, e.g., {"A": "option1", "B": "option2", ...}
    """
    return {label: item for label, item in zip(labels, items)}


def select_distractors(correct_items: List[str], all_items: List[str],
                       count: int, rng: random.Random) -> List[str]:
    """
    Select random distractor items from a list, excluding correct items.

    Args:
        correct_items: Items that should NOT be selected
        all_items: Pool of items to select from
        count: Number of distractors to select
        rng: Random instance for reproducibility

    Returns:
        List of selected distractor items
    """
    # Filter out correct items
    distractor_pool = [item for item in all_items if item not in correct_items]

    # Select requested number of distractors (or all if not enough)
    num_to_select = min(count, len(distractor_pool))
    return rng.sample(distractor_pool, num_to_select)


def filter_sequential_numbers(loc_descs: List[str]) -> List[str]:
    """
    Filter out locDescs that contain sequential numbers.

    Based on benchmark_design.yaml requirement:
    "Do not select objects with locDesc containing sequential numbers."

    Args:
        loc_descs: List of location descriptions

    Returns:
        Filtered list without sequential numbers
    """
    filtered = []
    for desc in loc_descs:
        # Check if description contains sequential numbers like "1 2", "1,2", etc.
        # Simple heuristic: if it has multiple numbers, skip it
        numbers = re.findall(r'\d+', desc)
        if len(numbers) <= 1:
            filtered.append(desc)
    return filtered


def format_question_metadata(question_id: str, question_type: str,
                             question_timestep: int, current_timestep: int,
                             template_used: str) -> Dict[str, Any]:
    """
    Create standardized metadata dictionary for a question.

    Args:
        question_id: Unique identifier for the question type
        question_type: "single_choice" or "multi_choice"
        question_timestep: Timestep this question references (the time being asked about)
        current_timestep: Current timestep when the question is generated
        template_used: The template string used for the question

    Returns:
        Metadata dictionary
    """
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question_timestep": question_timestep,
        "current_timestep": current_timestep,
        "template_used": template_used
    }


def create_question_dict(
        question_id: str,
        question_text: str,
        options: Dict[str, str],
        correct_answer: str,
        metadata: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a complete question dictionary with all required fields.

    Args:
        question_id: Unique identifier (e.g., "obj_loc_initial_001")
        question_text: The actual question text
        options: Dictionary of labeled options {"A": "...", "B": "...", ...}
        correct_answer: The correct option label(s), e.g., "A" or "A,B,C"
        metadata: Question metadata dictionary
        evidence: Optional evidence dictionary with tracker data

    Returns:
        Complete question dictionary
    """
    result = {
        "id": question_id,
        "question_text": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "metadata": metadata
    }

    if evidence is not None:
        result["evidence"] = evidence

    return result


def validate_question_dict(question: Dict[str, Any]) -> bool:
    """
    Validate that a question dictionary has all required fields.

    Args:
        question: Question dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    required_keys = {
        "id", "question_text", "options", "correct_answer", "metadata"
    }

    # Check all required keys exist
    if not required_keys.issubset(question.keys()):
        return False

    # Validate options are non-empty
    if not question["options"]:
        return False

    # Get question_type from metadata
    question_type = question["metadata"].get("question_type")

    # Validate correct_answer is a valid option label
    if question_type == "single_choice":
        if question["correct_answer"] not in question["options"]:
            return False
    elif question_type == "multi_choice":
        # For multi-choice, correct_answer can be like "A,B,C"
        answers = question["correct_answer"].split(",")
        if not all(a.strip() in question["options"] for a in answers):
            return False

    # Validate metadata has required fields
    metadata_required = {
        "question_timestep", "current_timestep", "question_id",
        "question_type", "template_used"
    }
    if not metadata_required.issubset(question["metadata"].keys()):
        return False

    # Validate evidence if present
    if "evidence" in question:
        evidence = question["evidence"]
        evidence_required = {
            "question_timestep",
            # "current_timestep",
            # "options_info",
        }
        if not evidence_required.issubset(evidence.keys()):
            return False

        # # Validate options_info has entries for all option labels
        # if not set(question["options"].keys()).issubset(
        #         evidence["options_info"].keys()):
        #     return False

    return True


def get_object_loc_desc(object_state) -> Optional[str]:
    """
    Safely get locDesc from an ObjectState.

    Args:
        object_state: ObjectState instance

    Returns:
        locDesc string or None if not available
    """
    if object_state is None:
        return None
    return object_state.loc_desc if object_state.loc_desc else None


def has_sequential_number(text: str) -> bool:
    """
    Check if text contains sequential numbers or ordinal indicators.

    Checks for:
    - Sequential numeric patterns like "1 2", "1,2", etc.
    - Ordinal words like 'first', 'second', 'third', etc.
    - Ordinal suffixes like '1st', '2nd', '3rd', '4th', etc.

    Args:
        text: String to check

    Returns:
        True if sequential numbers or ordinals detected, False otherwise
    """
    text_lower = text.lower()

    # Check for ordinal words
    ordinal_words = {
        'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
        'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth',
        'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth',
        'nineteenth', 'twentieth'
    }

    for ordinal in ordinal_words:
        if ordinal in text_lower:
            return True

    # Check for ordinal suffixes (1st, 2nd, 3rd, 4th, 21st, etc.)
    # Pattern: digit + (st|nd|rd|th)
    if re.search(r'\d+(st|nd|rd|th)\b', text_lower):
        return True

    # Check for sequential numeric patterns
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        # Check if any two consecutive numbers are sequential
        nums = [int(n) for n in numbers]
        for i in range(len(nums) - 1):
            if abs(nums[i] - nums[i + 1]) == 1:
                return True

    return False


def shuffle_options(options: Dict[str, str], rng: random.Random) -> tuple:
    """
    Shuffle options while tracking correct answer.

    Args:
        options: Dictionary of options {"A": "...", "B": "...", ...}
        rng: Random instance

    Returns:
        Tuple of (shuffled_options, new_correct_label)
    """
    items = list(options.items())
    rng.shuffle(items)

    labels = "ABCDEFG"
    shuffled = {}
    label_mapping = {}

    for i, (old_label, text) in enumerate(items):
        new_label = labels[i]
        shuffled[new_label] = text
        label_mapping[old_label] = new_label

    return shuffled, label_mapping


def get_object_history(tracker, object_id: str,
                       current_timestep: int) -> List[Dict[str, Any]]:
    """
    Get complete history of where an object has been placed.

    Args:
        tracker: ThorTracker instance
        object_id: Object ID to track
        current_timestep: Current timestep

    Returns:
        List of unique parent IDs from object's history
    """
    unique_parents = []

    for t in range(current_timestep + 1):
        obj_state = tracker.get_object_state_at_step(object_id, t)
        if obj_state and obj_state.parent_receptacles:
            current_parent_id = obj_state.parent_receptacles[0]

            # Only add if not already in list (unique parents)
            if current_parent_id not in unique_parents:
                unique_parents.append((current_parent_id, t))

    return unique_parents


def calculate_relative_direction(obj_a_pos: Dict[str, float],
                                 obj_b_pos: Dict[str, float]) -> tuple:
    """
    Calculate cardinal direction from obj_a to obj_b.

    Coordinate system definition:
    - Y-axis: Vertical (up/down)
    - XZ-plane: Horizontal directions
      - +Z direction = North
      - +X direction = East
      - -Z direction = South
      - -X direction = West

    Args:
        obj_a_pos: Position dict with 'x', 'y', 'z' keys
        obj_b_pos: Position dict with 'x', 'y', 'z' keys

    Returns:
        Tuple of (direction: str, angle_degrees: float)
        direction: One of "East", "South", "West", "North"
        angle_degrees: Angle in degrees [0, 360) from obj_a to obj_b
                      (0° = North, clockwise)
    """
    dx = obj_b_pos['x'] - obj_a_pos['x']  # East-West (+X=East)
    dz = obj_b_pos['z'] - obj_a_pos['z']  # North-South (+Z=North)

    # arctan2(dz, dx) returns: 0°=+X (East), 90°=+Z (North)
    angle_rad = math.atan2(dz, dx)
    angle_deg = math.degrees(angle_rad)

    # Convert to North=0° reference system (clockwise)
    # 90° (North) -> 0°, 0° (East) -> 270°
    angle_deg = (90 - angle_deg) % 360

    # Convert to cardinal direction (simplified 4-direction version)
    # Each direction covers 90 degrees
    # North: [315°, 360°) ∪ [0°, 45°)
    if 315 <= angle_deg or angle_deg < 45:
        direction = "North"
    # East: [45°, 135°)
    elif 45 <= angle_deg < 135:
        direction = "East"
    # South: [135°, 225°)
    elif 135 <= angle_deg < 225:
        direction = "South"
    # West: [225°, 315°)
    else:  # 225 <= angle_deg < 315
        direction = "West"

    return direction, angle_deg


def get_objects_in_container(tracker, container_id: str,
                             timestep: int) -> List[str]:
    """
    Get all object IDs inside a container at a timestep.

    Args:
        tracker: ThorTracker instance
        container_id: Container object ID
        timestep: Timestep to query

    Returns:
        List of object IDs inside the container
    """
    container_state = tracker.get_object_state_at_step(container_id, timestep)
    if container_state and container_state.receptacle_object_ids:
        return container_state.receptacle_object_ids
    return []


def is_top_level_object(obj_state: Any) -> bool:
    """
    Check if an object is a top-level object (not in any container).

    Top-level objects are those that are not contained in any receptacle/container.
    These are the objects that can be seen and referenced in position questions.

    Args:
        obj_state: ObjectState instance

    Returns:
        True if object is not in any container (top-level), False otherwise
    """
    # Check if object has any parent receptacles
    # If parent_receptacles is empty, the object is top-level

    return (obj_state.parent_receptacles is None) or (len(
        obj_state.parent_receptacles) == 0)


def get_seen_objects(tracker, current_timestep: int) -> List[str]:
    """
    Get objects that the agent has seen at some point.

    Args:
        tracker: ThorTracker instance
        current_timestep: Current timestep

    Returns:
        List of object IDs that have been visible
    """
    seen_objects = set()
    for t in range(current_timestep + 1):
        for obj_id in tracker.all_object_ids:
            state = tracker.get_object_state_at_step(obj_id, t)
            if state and state.is_visible:
                seen_objects.add(obj_id)
    return list(seen_objects)


def get_object_property_state(obj_state: Any) -> Optional[Dict[str, Any]]:
    """
    Get relevant property state for an object.

    Args:
        obj_state: ObjectState instance

    Returns:
        Dictionary with property info or None if no relevant property
    """
    # Check boolean properties in priority order
    if obj_state.openable and hasattr(obj_state, 'is_open'):
        return {
            'property_type': 'container',
            'state': 'OPEN' if obj_state.is_open else 'CLOSED',
            'openable': True
        }
    elif obj_state.toggleable and hasattr(obj_state, 'is_toggled_on'):
        return {
            'property_type': 'toggleable',
            'state': 'ON' if obj_state.is_toggled_on else 'OFF',
            'toggleable': True
        }
    elif hasattr(obj_state, 'is_sliced') and obj_state.sliceable:
        return {
            'property_type': 'sliceable',
            'state': 'SLICED' if obj_state.is_sliced else 'INTACT',
            'sliceable': True
        }
    elif hasattr(obj_state, 'is_dirty') and obj_state.dirtyable:
        return {
            'property_type': 'dirtyable',
            'state': 'DIRTY' if obj_state.is_dirty else 'CLEAN',
            'dirtyable': True
        }

    return None


def get_objects_by_type(tracker, object_type: str, timestep: int) -> List[str]:
    """
    Get all object IDs of a given type at a timestep.

    Args:
        tracker: ThorTracker instance
        object_type: Object type to match
        timestep: Timestep to query

    Returns:
        List of object IDs matching the type
    """
    matching_ids = []
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if state and state.object_type == object_type:
            matching_ids.append(obj_id)
    return matching_ids


def generate_count_distractors(actual_count: int, num_distractors: int,
                               rng: random.Random) -> List[int]:
    """
    Generate distractor counts near the actual count.

    Args:
        actual_count: The correct count
        num_distractors: Number of distractors to generate
        rng: Random instance

    Returns:
        List of distractor counts (non-negative, distinct from actual)
    """
    candidates = []

    # Generate candidates near actual count
    for offset in range(-3, 4):  # -3 to +3
        if offset != 0:
            candidate = actual_count + offset
            if candidate >= 0:  # Non-negative only
                candidates.append(candidate)

    # Shuffle and select required number
    rng.shuffle(candidates)
    return candidates[:num_distractors]


def get_objects_in_direction(tracker, agent_pos: Dict[str, float],
                             direction: str, timestep: int) -> List[tuple]:
    """
    Get objects located in a cardinal direction from agent position.

    Args:
        tracker: ThorTracker instance
        agent_pos: Agent position {'x': float, 'y': float, 'z': float}
        direction: Cardinal direction ("East", "South", "West", "North")
        timestep: Timestep to query

    Returns:
        List of (object_id, object_type, position) tuples in that direction
    """
    objects_in_direction, objects_not_in_direction = [], []
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if state and obj_id in tracker.historically_seen_objects:
            obj_loc_desc = get_object_loc_desc(state)
            if obj_loc_desc and not has_sequential_number(obj_loc_desc):
                obj_dir = calculate_relative_direction(agent_pos,
                                                       state.position)
                if obj_dir[0] == direction:
                    objects_in_direction.append(
                        (obj_id, state.object_type, state.position))
                else:
                    objects_not_in_direction.append(
                        (obj_id, state.object_type, state.position))
    return objects_in_direction, objects_not_in_direction


def get_visible_objects_at_step(tracker, timestep: int) -> List[str]:
    """
    Get all object IDs visible at a specific timestep.

    Args:
        tracker: ThorTracker instance
        timestep: Timestep to query

    Returns:
        List of object IDs that are visible at this timestep
    """
    visible_objects = []
    for obj_id in tracker.all_object_ids:
        state = tracker.get_object_state_at_step(obj_id, timestep)
        if state and state.is_visible:
            visible_objects.append((obj_id, state.object_type))
    return visible_objects


def is_immovable_object(obj_state) -> bool:
    """
    Check if an object is immovable (not pickupable).

    Args:
        obj_state: ObjectState instance

    Returns:
        True if object cannot be picked up, False otherwise
    """
    # Objects that are NOT pickupable are immovable
    return not obj_state.pickupable if hasattr(obj_state,
                                               'pickupable') else False
