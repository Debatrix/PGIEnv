#!/usr/bin/env python3
import os
import json
import logging
from glob import glob
import math
from tqdm import tqdm
import numpy as np
from PIL import Image

from pgibench.envs.pgi_alfred.thor_connector import ThorConnector
from pgibench.envs.pgi_alfred.gen import constants

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('running/path_point_generation.log'),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

DISTANCE_FROM_REFERENCE = 1.0
HORIZONTAL_ANGLES = [0, 90, 180, 270, 45, 135, 225, 315]  #
VERTICAL_ANGLES = [0, 30, -30]
FACING_ANGLE_THRESHOLD = 45
MIN_COVERAGE_TARGET = 95.0
OUTPUT_DIR = 'pgibench/envs/pgi_alfred/gen/layouts'
REPORT_DIR = 'running'
STATE_RECORD_DIR = 'running/replay_batch'

# Object overlap filtering settings
OVERLAP_DISTANCE_THRESHOLD = 0.25  # Distance threshold (meters) for considering objects as overlapping


def get_visible_objects_with_pixel_counts(env_connector,
                                          use_inst_seg=True,
                                          min_pixel_ratio=0.00005,
                                          margin=10):
    """
    Get visible objects and their total pixel counts based on instance segmentation frame.
    This is a specialized version for path_point_calculator that returns both visible objects
    and total pixel counts for better path point quality assessment.

    Args:
        env_connector: ThorConnector instance
        use_inst_seg: Whether to use instance segmentation (default: True)
        min_pixel_ratio: Minimum pixel ratio threshold (default: 0.00005, ~0.005% of image)
                       Only objects with pixel count ratio >= this threshold are considered visible
        margin: Margin to crop from edges (default: 10 pixels)

    Returns:
        tuple: (list of visible object IDs, total visible pixels)
               - visible_objects: List of object IDs that are visible
               - total_visible_pixels: Sum of all pixels for visible objects
    """
    visible_objects = []
    total_visible_pixels = 0

    if hasattr(env_connector.last_event,
               'instance_segmentation_frame') and use_inst_seg:
        seg_frame = np.array(
            env_connector.last_event.instance_segmentation_frame)
        seg_frame = seg_frame[margin:-margin, margin:-margin, :]

        h, w = seg_frame.shape[:2]
        total_pixels = h * w
        min_pixel_threshold = max(5, int(total_pixels * min_pixel_ratio))

        # Get all valid object IDs from metadata
        valid_object_ids = {
            obj['objectId']
            for obj in env_connector.last_event.metadata['objects']
        }

        # Build reverse mapping: color -> object_id, but only for valid objects
        color_to_valid_obj_id = {
            color: obj_id
            for color, obj_id in
            env_connector.last_event.color_to_object_id.items()
            if obj_id in valid_object_ids
        }

        # Use numpy vectorized operations for faster counting
        # Reshape frame to (h*w, 3) for efficient processing
        seg_frame_reshaped = seg_frame.reshape(-1, 3)

        # Convert each row (RGB) to tuple for dictionary lookup
        # This is faster than iterating through pixels
        unique_colors, counts = np.unique(seg_frame_reshaped,
                                          axis=0,
                                          return_counts=True)

        # Count pixels only for valid objects
        object_pixel_counts = {}
        for color, count in zip(unique_colors, counts):
            color_tuple = tuple(color)
            if color_tuple in color_to_valid_obj_id:
                obj_id = color_to_valid_obj_id[color_tuple]
                object_pixel_counts[obj_id] = object_pixel_counts.get(
                    obj_id, 0) + count

        # Filter objects by pixel count threshold and calculate total
        for obj_id, pixel_count in object_pixel_counts.items():
            if pixel_count >= min_pixel_threshold:
                visible_objects.append(obj_id)
                total_visible_pixels += pixel_count
    else:
        # Fallback to simple visibility check (no pixel counts available)
        for obj in env_connector.last_event.metadata['objects']:
            if obj['visible']:
                visible_objects.append(obj['objectId'])
        # When not using instance segmentation, we can't calculate pixel counts
        # Return 0 as placeholder
        total_visible_pixels = 0

    return visible_objects, total_visible_pixels


def load_state_records_for_scene(scene_num):
    """
    Load all state_record.json files for a given scene from running/replay_batch/.

    Args:
        scene_num: Scene number (e.g., 1, 2, 301)

    Returns:
        list: List of tuples (trajectory_id, state) for all steps in all trajectories for this scene
    """
    all_states = []

    # Find all state_record.json files
    for trial_dir in glob(os.path.join(STATE_RECORD_DIR, 'trial_*')):
        state_file = os.path.join(trial_dir, 'state_record.json')
        if not os.path.exists(state_file):
            continue

        try:
            with open(state_file, 'r') as f:
                record = json.load(f)

            # Check if this record is for the target scene
            if record.get('scene_num') != scene_num:
                continue

            trajectory_id = record['trajectory_id']

            # Extract all states from this trajectory
            for state in record.get('states', []):
                all_states.append((trajectory_id, state))

        except Exception as e:
            logger.warning(f"Failed to load {state_file}: {e}")
            continue

    logger.info(
        f"Loaded {len(all_states)} states from {len(set(s[0] for s in all_states))} trajectories for scene {scene_num}"
    )
    return all_states


def build_object_tree_with_uniqueness(objects_metadata):
    """
    Build object relationship tree and classify into unique and non-unique non-movable top-level objects.

    Args:
        objects_metadata: List of object metadata from AI2-THOR

    Returns:
        dict: Dictionary containing object tree and classifications
    """
    # Initialize object information mapping
    object_info = {}
    for obj in objects_metadata:
        obj_id = obj['objectId']
        object_info[obj_id] = {
            'object_type': obj['objectType'],
            'object_id': obj_id,
            'pickupable': obj['pickupable'],
            'parent_receptacles': obj.get('parentReceptacles', []),
            'position': obj['position'],
            'is_container': False
        }

    # Identify containers
    for obj in objects_metadata:
        if obj.get('parentReceptacles', []):
            for parent_id in obj['parentReceptacles']:
                if parent_id in object_info:
                    object_info[parent_id]['is_container'] = True

    # Find top-level objects (objects not in containers)
    root_objects = []
    for obj_id, info in object_info.items():
        if not info['parent_receptacles']:
            root_objects.append(info)

    # Count of each objectType
    type_counts = {}
    for obj in objects_metadata:
        obj_type = obj['objectType']
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

    # Classify top-level objects
    unique_non_movable_top_level = []
    non_unique_non_movable_top_level = []

    for obj in root_objects:
        if not obj['pickupable']:  # non-movable
            obj_type = obj['object_type']
            if type_counts[obj_type] == 1:  # unique
                unique_non_movable_top_level.append(obj)
            else:  # non-unique
                non_unique_non_movable_top_level.append(obj)

    return {
        'all_objects':
        object_info,
        'root_objects':
        root_objects,
        'unique_non_movable_top_level':
        unique_non_movable_top_level,
        'non_unique_non_movable_top_level':
        non_unique_non_movable_top_level,
        'total_objects':
        len(object_info),
        'root_objects_count':
        len(root_objects),
        'container_count':
        sum(1 for obj in object_info.values() if obj['is_container'])
    }


def calculate_target_position(obj_pos, agent_pos):
    """
    Calculate target position 1 unit outward from object along XZ direction.

    Args:
        obj_pos: Object position {'x': x, 'y': y, 'z': z}
        agent_pos: Agent position {'x': x, 'y': y, 'z': z}

    Returns:
        dict: Target position {'x': x, 'y': y, 'z': z}
    """
    dx = agent_pos['x'] - obj_pos['x']
    dz = agent_pos['z'] - obj_pos['z']
    distance = math.sqrt(dx * dx + dz * dz)

    if distance < 0.01:  # Avoid division by zero
        # If agent and object are at almost same position, choose random direction
        target_x = obj_pos['x'] + DISTANCE_FROM_REFERENCE
        target_z = obj_pos['z']
    else:
        # Move 1 unit along direction from object to agent
        target_x = obj_pos['x'] + (dx / distance) * DISTANCE_FROM_REFERENCE
        target_z = obj_pos['z'] + (dz / distance) * DISTANCE_FROM_REFERENCE

    return {'x': target_x, 'y': agent_pos['y'], 'z': target_z}


def get_spatial_relation(obj_pos, ref_pos):
    """
    Determine the spatial relationship between two positions using cardinal directions.

    Coordinate system definition:
    - Y-axis: Vertical (up/down)
    - XZ-plane: Horizontal directions
      - +Z direction = North
      - +X direction = East
      - -Z direction = South
      - -X direction = West

    Args:
        obj_pos: Object position
        ref_pos: Reference object position

    Returns:
        str: Spatial relation description (N, S, E, W, NE, NW, SE, SW)
    """
    dx = obj_pos['x'] - ref_pos['x']  # East-West (+X=East)
    dz = obj_pos['z'] - ref_pos['z']  # North-South (+Z=North)

    # arctan2(dz, dx) returns: 0°=+X (East), 90°=+Z (North)
    angle_rad = np.arctan2(dz, dx)
    angle_deg = np.degrees(angle_rad)

    # Convert to North=0° reference system (clockwise)
    # 90° (North) -> 0°, 0° (East) -> 90°
    angle = (90 - angle_deg) % 360

    # Determine direction based on angle (0°=North, clockwise increase)
    # North: 337.5-360 or 0-22.5
    if 337.5 <= angle or angle < 22.5:
        return 'N'
    # Northeast: 22.5-67.5
    elif 22.5 <= angle < 67.5:
        return 'NE'
    # East: 67.5-112.5
    elif 67.5 <= angle < 112.5:
        return 'E'
    # Southeast: 112.5-157.5
    elif 112.5 <= angle < 157.5:
        return 'SE'
    # South: 157.5-202.5
    elif 157.5 <= angle < 202.5:
        return 'S'
    # Southwest: 202.5-247.5
    elif 202.5 <= angle < 247.5:
        return 'SW'
    # West: 247.5-292.5
    elif 247.5 <= angle < 292.5:
        return 'W'
    # Northwest: 292.5-337.5
    else:
        return 'NW'


def rotation_to_cardinal(rotation):
    """
    Convert rotation angle to cardinal direction (8 directions).

    Coordinate system definition:
    - Agent rotation 0° faces North (+Z direction)
    - Angle increases clockwise
    - N: 0°, NE: 45°, E: 90°, SE: 135°, S: 180°, SW: 225°, W: 270°, NW: 315°

    Args:
        rotation: Rotation angle in degrees

    Returns:
        str: Cardinal direction abbreviation (N, S, E, W, NE, NW, SE, SW)
    """
    # Normalize to 0-360
    rotation = rotation % 360
    # North: 337.5-360 or 0-22.5
    if rotation >= 337.5 or rotation < 22.5:
        return 'N'
    # Northeast: 22.5-67.5
    elif 22.5 <= rotation < 67.5:
        return 'NE'
    # East: 67.5-112.5
    elif 67.5 <= rotation < 112.5:
        return 'E'
    # Southeast: 112.5-157.5
    elif 112.5 <= rotation < 157.5:
        return 'SE'
    # South: 157.5-202.5
    elif 157.5 <= rotation < 202.5:
        return 'S'
    # Southwest: 202.5-247.5
    elif 202.5 <= rotation < 247.5:
        return 'SW'
    # West: 247.5-292.5
    elif 247.5 <= rotation < 292.5:
        return 'W'
    # Northwest: 292.5-337.5
    else:  # 292.5 <= rotation < 337.5
        return 'NW'


def horizon_to_suffix(horizon):
    """
    Convert horizon (pitch) angle to suffix.

    Args:
        horizon: Horizon angle in degrees

    Returns:
        str: 'D' for looking down, 'U' for looking up, '' for level
    """
    if horizon > 5:
        return 'D'
    elif horizon < -5:
        return 'U'
    else:
        return ''


def find_nearest_non_movable_object(agent_pos, objects_metadata):
    """
    Find the nearest visible non-movable object to the agent.

    Args:
        agent_pos: Agent position dictionary {'x', 'y', 'z'}
        objects_metadata: List of all object metadata in the scene

    Returns:
        dict: Nearest non-movable object info {'object_type', 'object_id', 'position'} or None
    """
    nearest_obj = None
    min_distance = float('inf')

    for obj in objects_metadata:
        obj_id = obj['objectId']
        # Only consider non-movable objects
        if not obj.get('pickupable', True):
            obj_pos = obj['position']
            # Calculate 3D distance
            distance = math.sqrt((agent_pos['x'] - obj_pos['x'])**2 +
                                 (agent_pos['y'] - obj_pos['y'])**2 +
                                 (agent_pos['z'] - obj_pos['z'])**2)

            if distance < min_distance:
                min_distance = distance
                nearest_obj = {
                    'object_type': obj['objectType'],
                    'object_id': obj_id,
                    'position': obj_pos
                }

    return nearest_obj


def generate_compact_location_description_stage135(env_connector, agent_pos,
                                                   agent_rotation,
                                                   visible_objects,
                                                   reference_obj):
    """
    Generate compact location description (for stages 1, 3, 5).
    Format: {object_type}_{cardinal_direction}
    Example: garbagecan_SE

    Args:
        env_connector: ThorConnector instance
        agent_pos: Agent position
        agent_rotation: Agent rotation angle
        visible_objects: List of visible object IDs
        reference_obj: Reference object dict (containing object_type and position)

    Returns:
        str: Compact location description
    """
    ref_type = reference_obj['object_type'].lower()
    ref_pos = reference_obj['position']

    # Get spatial relation and convert to cardinal direction
    cardinal = get_spatial_relation(agent_pos, ref_pos)

    # Generate compact name
    compact_desc = f"{cardinal}_{ref_type}"

    return compact_desc


def generate_compact_location_description_stage24(env_connector, agent_pos,
                                                  agent_rotation,
                                                  agent_horizon,
                                                  visible_objects,
                                                  reference_obj):
    ref_type = reference_obj['object_type'].lower()
    ref_pos = reference_obj['position']

    # Get spatial relation and convert to cardinal direction
    spatial_cardinal = get_spatial_relation(agent_pos, ref_pos)

    # Get facing direction
    facing_cardinal = rotation_to_cardinal(agent_rotation)

    # Get horizon (pitch) suffix
    horizon_suffix = horizon_to_suffix(agent_horizon)

    # Build the facing part
    if horizon_suffix:
        facing_part = f"{facing_cardinal}_{horizon_suffix}"
    else:
        facing_part = f"{facing_cardinal}"

    # Generate compact name
    compact_desc = f"{spatial_cardinal}_{ref_type}_{facing_part}"

    return compact_desc


def add_path_point_with_deduplication(new_point, existing_points):
    """
    Add path point with deduplication check.

    Deduplication rules (in priority order):
    1. Distance and angle similarity check: Skip if XZ distance < 1m AND rotation diff < 45° AND horizon diff < 30°
       - If visible object count differs, keep the one with more objects
       - If visible object count is same, keep the one with higher total visible pixels
    2. Visible objects containment check:
       - New point fully contained by existing -> Skip
       - New point fully contains existing -> Remove existing point

    Args:
        new_point: New path point (containing visible_objects set and optional total_visible_pixels)
        existing_points: List of existing path points

    Returns:
        tuple: (whether_added, removed_indices_list, updated_path_points_list)
    """
    new_pos = new_point['position']
    new_rot = new_point['rotation']
    new_horizon = new_point['horizon']
    new_visible_set = set(new_point['visible_objects'])
    new_total_pixels = new_point.get('total_visible_pixels', 0)

    # Rule 1: distance and angle similarity check (primary rule)
    for idx, existing_point in enumerate(existing_points):
        existing_pos = existing_point['position']
        existing_rot = existing_point['rotation']
        existing_horizon = existing_point['horizon']

        # Calculate XZ-plane distance
        xz_distance = math.sqrt((new_pos['x'] - existing_pos['x'])**2 +
                                (new_pos['z'] - existing_pos['z'])**2)

        # Calculate rotation difference
        rot_diff = abs(new_rot - existing_rot)
        if rot_diff > 180:
            rot_diff = 360 - rot_diff

        # Calculate horizon difference
        horizon_diff = abs(new_horizon - existing_horizon)

        # Check whether too similar
        if xz_distance < 1.5 and rot_diff < 45 and horizon_diff < 30:
            logger.info(
                f"Skipping new path point: too similar to path point {idx} "
                f"(distance: {xz_distance:.2f}m, rotation diff: {rot_diff:.1f}°, horizon diff: {horizon_diff:.1f}°)"
            )
            # Replace if the new path point sees more objects
            existing_visible_set = set(existing_point['visible_objects'])
            existing_total_pixels = existing_point.get('total_visible_pixels',
                                                       0)

            if len(new_visible_set) > len(existing_visible_set):
                logger.info(
                    f"  But new path point sees more objects ({len(new_visible_set)} vs {len(existing_visible_set)}), replacing path point {idx}"
                )
                updated_points = [
                    p if i != idx else new_point
                    for i, p in enumerate(existing_points)
                ]
                return (True, [idx], updated_points)
            elif len(new_visible_set) == len(existing_visible_set):
                # When visible object counts are equal, compare total pixel count
                if new_total_pixels > existing_total_pixels:
                    logger.info(
                        f"  Same visible object count ({len(new_visible_set)}), "
                        f"but new path point has higher total pixel count ({new_total_pixels} vs {existing_total_pixels}), replacing path point {idx}"
                    )
                    updated_points = [
                        p if i != idx else new_point
                        for i, p in enumerate(existing_points)
                    ]
                    return (True, [idx], updated_points)
                else:
                    logger.info(
                        f"  Same visible object count ({len(new_visible_set)}), "
                        f"but existing path point has higher or equal total pixel count ({existing_total_pixels} vs {new_total_pixels}), skipping"
                    )
                    return (False, [], existing_points)
            else:
                return (False, [], existing_points)

    # Rule 2: visible object containment check
    for idx, existing_point in enumerate(existing_points):
        existing_visible_set = set(existing_point['visible_objects'])
        existing_total_pixels = existing_point.get('total_visible_pixels', 0)

        # Rule 2a: new path point fully contained by an existing path point
        if new_visible_set.issubset(existing_visible_set):
            # Skip directly if the new path point sees fewer objects
            if len(new_visible_set) < len(existing_visible_set):
                logger.info(
                    f"Skipping new path point: fully contained by path point {idx} (visible objects: {len(new_visible_set)} vs {len(existing_visible_set)})"
                )
                return (False, [], existing_points)
            # If object counts are equal, compare total pixel count
            elif len(new_visible_set) == len(existing_visible_set):
                if new_total_pixels > existing_total_pixels:
                    logger.info(
                        f"New path point has same object count but higher pixel count, replacing path point {idx} "
                        f"(visible objects: {len(new_visible_set)}, pixels: {new_total_pixels} vs {existing_total_pixels})"
                    )
                    updated_points = [
                        p if i != idx else new_point
                        for i, p in enumerate(existing_points)
                    ]
                    return (True, [idx], updated_points)
                else:
                    logger.info(
                        f"Skipping new path point: fully contained by path point {idx} with lower or equal pixel count "
                        f"(visible objects: {len(new_visible_set)}, pixels: {new_total_pixels} vs {existing_total_pixels})"
                    )
                    return (False, [], existing_points)

        # Rule 2b: new path point fully contains an existing path point
        if new_visible_set.issuperset(existing_visible_set):
            # Remove the existing path point if the new one sees more objects, or same count with higher pixel count
            if len(new_visible_set) > len(existing_visible_set):
                logger.info(
                    f"New path point contains path point {idx}, will remove path point {idx} "
                    f"(visible objects: {len(new_visible_set)} vs {len(existing_visible_set)})"
                )
            elif len(new_visible_set) == len(
                    existing_visible_set
            ) and new_total_pixels >= existing_total_pixels:
                logger.info(
                    f"New path point has same object count as path point {idx} but higher or equal pixel count, will remove path point {idx} "
                    f"(visible objects: {len(new_visible_set)}, pixels: {new_total_pixels} vs {existing_total_pixels})"
                )
            else:
                # Only remove when the new path point contains the existing one and sees more objects
                # If only the pixel count is higher with equal object count, do not remove (keep stability)
                continue

            # Create a new list (without the removed path point)
            updated_points = [
                p for i, p in enumerate(existing_points) if i != idx
            ]
            # Recurse (removal may create new containment relations)
            return add_path_point_with_deduplication(new_point, updated_points)

    # Rule 3: independent path point -> add
    existing_points.append(new_point)
    return (True, [], existing_points)


def global_deduplication(all_path_points):
    """
    Perform global deduplication on all path points to eliminate cross-stage redundant points.

    Args:
        all_path_points: List of all path points (containing phase1, phase2, phase3)

    Returns:
        list: Deduplicated path points list
    """
    logger.info(f"=== Global dedup: checking {len(all_path_points)} path points ===")

    # Sort by visible object count descending, prefer keeping path points that see more objects
    sorted_points = sorted(all_path_points,
                           key=lambda p: len(p['visible_objects']),
                           reverse=True)

    deduplicated_points = []
    removed_count = 0

    for point in sorted_points:
        added, removed_indices, deduplicated_points = add_path_point_with_deduplication(
            point, deduplicated_points)

        if added:
            # Keep the original phase info for debugging
            pass
        else:
            removed_count += 1
            phase = point.get('phase', '?')
            logger.info(f"  Removed phase {phase} path point: {point['loc_desc'][:50]}...")

    logger.info(
        f"Global dedup done: reduced from {len(all_path_points)} to {len(deduplicated_points)} (removed {removed_count})"
    )

    return deduplicated_points


def deduplicate_location_descriptions(all_path_points):
    """
    Ensure all path points have unique loc_desc by adding _#1, _#2, etc. suffixes to distinguish duplicates.

    Args:
        all_path_points: List of all path points

    Returns:
        list: Path points list with deduplicated loc_desc
    """
    logger.info(f"=== loc_desc dedup: checking {len(all_path_points)} path points ===")

    # Group by loc_desc
    desc_groups = {}
    for i, point in enumerate(all_path_points):
        desc = point['loc_desc']
        if desc not in desc_groups:
            desc_groups[desc] = []
        desc_groups[desc].append((i, point))

    modified_count = 0

    for desc, points in desc_groups.items():
        if len(points) > 1:
            # Duplicate descriptions found, add suffixes to distinguish them
            logger.info(f"  Found duplicate description: '{desc}' (appears {len(points)} times)")

            # Sort by position, rotation and horizon so numbering stays consistent
            sorted_points = sorted(points,
                                   key=lambda x:
                                   (x[1]['position']['x'], x[1]['position']
                                    ['z'], x[1]['rotation'], x[1]['horizon']))

            # Add _#1, _#2, ... suffixes to each duplicate description
            for idx, (i, point) in enumerate(sorted_points, start=1):
                new_desc = f"{desc}_#{idx}"
                all_path_points[i]['loc_desc'] = new_desc
                modified_count += 1
                logger.info(f"    Path point {i+1}: '{desc}' -> '{new_desc}'")

    logger.info(f"loc_desc dedup done: modified descriptions of {modified_count} path points")

    return all_path_points


def generate_init_action_path_points(env_connector, traj_data_list):
    """
    Generate initial path points from all trajectories' init_action.

    Args:
        env_connector: ThorConnector instance
        traj_data_list: List of trajectory data dictionaries

    Returns:
        list: List of initial path points
    """
    init_path_points = []

    logger.info(f"=== Generating init-position path points from trajectories ===")
    logger.info(f"Number of trajectories: {len(traj_data_list)}")

    for traj_data in traj_data_list:
        init_action = traj_data.get('scene', {}).get('init_action', {})
        trajectory_id = traj_data.get('task_id', 'unknown')

        if not init_action or init_action.get('action') != 'TeleportFull':
            continue

        # Teleport to initial position
        env_connector.step({
            'action':
            'TeleportFull',
            'x':
            init_action['x'],
            'y':
            init_action.get('y', env_connector.agent_height),
            'z':
            init_action['z'],
            'rotation':
            init_action.get('rotation', 0),
            'horizon':
            init_action.get('horizon', 0),
            'rotateOnTeleport':
            True
        })

        if not env_connector.last_event.metadata['lastActionSuccess']:
            logger.warning(
                f"Failed to teleport to init position for {trajectory_id}")
            continue

        # Get visible objects with pixel counts
        visible_objs, total_pixels = get_visible_objects_with_pixel_counts(
            env_connector)

        # Get reference object
        objects_metadata = env_connector.last_event.metadata['objects']
        nearest_non_movable = find_nearest_non_movable_object(
            init_action, objects_metadata)

        if not nearest_non_movable:
            logger.warning(f"No reference object found for init position")
            continue

        ref_obj_dict = {
            'object_type': nearest_non_movable['object_type'],
            'object_id': nearest_non_movable['object_id'],
            'position': nearest_non_movable['position']
        }

        # Generate location description
        final_pos = env_connector.last_event.metadata['agent']['position']
        final_rot = env_connector.last_event.metadata['agent']['rotation']['y']
        final_horizon = env_connector.last_event.metadata['agent'][
            'cameraHorizon']

        loc_desc = generate_compact_location_description_stage135(
            env_connector, final_pos, final_rot, visible_objs, ref_obj_dict)

        path_point = {
            'position': final_pos,
            'rotation': final_rot,
            'horizon': final_horizon,
            'loc_desc': loc_desc,
            'visible_objects': visible_objs,
            'total_visible_pixels': total_pixels,
            'phase': -1,  # Mark as initial path point
            'reference_object': ref_obj_dict['object_id'],
            'reference_object_type': ref_obj_dict['object_type'],
            'source_trajectory': trajectory_id,
            'source_step': 'init'
        }

        init_path_points.append(path_point)

    logger.info(f"Generated {len(init_path_points)} init action path points")
    return init_path_points


def generate_phase0_path_points(env_connector, all_states, visibility_tracker):
    """
    Phase 0: Generate path points from trajectory state records.

    Extracts all agent positions from state records, sorts by visible object count,
    deduplicates, and generates location descriptions.

    Args:
        env_connector: ThorConnector instance
        all_states: List of (trajectory_id, state) tuples from load_state_records_for_scene
        visibility_tracker: Visibility tracking dictionary

    Returns:
        list: List of phase 0 path points
    """
    path_points = []
    dedup_count = 0

    logger.info(f"=== Phase 0: extracting path points from trajectory state records ===")
    logger.info(f"Total states: {len(all_states)}")

    # Step 1: Extract all candidate path points from state records
    candidates = []
    for trajectory_id, state in all_states:
        agent_state = state['agent_state']

        candidate = {
            'position': agent_state['position'],
            'rotation': agent_state['rotation']['y'],
            'horizon': agent_state['horizon'],
            'visible_objects': state['visible_objects'],
            'source': trajectory_id,
            'step': state['step']
        }
        candidates.append(candidate)

    logger.info(f"Extracted {len(candidates)} candidate path points")

    # Step 2: Sort by visible object count (descending)
    candidates.sort(key=lambda x: len(x['visible_objects']), reverse=True)
    logger.info(
        f"Sorted by visible object count, range: {len(candidates[0]['visible_objects'])} - {len(candidates[-1]['visible_objects'])}"
    )

    # Step 3: Deduplicate and add path points
    added_count = 0
    skipped_count = 0

    for idx, candidate in enumerate(candidates):
        # Check if this point has already been covered by existing points
        visible_set = set(candidate['visible_objects'])
        new_objects = visible_set - visibility_tracker['visible_objects']

        # Skip if no new objects (for efficiency)
        if not new_objects and idx > 0:
            skipped_count += 1
            continue

        # Find nearest non-movable object for reference
        nearest_non_movable = find_nearest_non_movable_object(
            candidate['position'],
            env_connector.last_event.metadata['objects'])

        if not nearest_non_movable:
            logger.warning(f"No reference object found, skipping candidate {idx}")
            continue

        ref_obj_dict = {
            'object_type': nearest_non_movable['object_type'],
            'object_id': nearest_non_movable['object_id'],
            'position': nearest_non_movable['position']
        }

        # Generate location description
        loc_desc = generate_compact_location_description_stage135(
            env_connector, candidate['position'], candidate['rotation'],
            candidate['visible_objects'], ref_obj_dict)

        path_point = {
            'position': candidate['position'],
            'rotation': candidate['rotation'],
            'horizon': candidate['horizon'],
            'loc_desc': loc_desc,
            'visible_objects': candidate['visible_objects'],
            'phase': 0,
            'reference_object': ref_obj_dict['object_id'],
            'reference_object_type': ref_obj_dict['object_type'],
            'source_trajectory': candidate['source'],
            'source_step': candidate['step']
        }

        # Deduplicate
        added, removed_indices, path_points = add_path_point_with_deduplication(
            path_point, path_points)

        if added:
            added_count += 1
            # Update visibility tracker
            visibility_tracker['visible_objects'].update(visible_set)

            # Recalculate coverage
            coverage = check_visibility_coverage(
                visibility_tracker['all_objects'],
                visibility_tracker['visible_objects'],
                visibility_tracker['objects_in_closed'])
            visibility_tracker['coverage_percentage'] = coverage[
                'coverage_percentage']

            if added_count % 50 == 0:
                logger.info(f"  Progress: {added_count}/{len(candidates)}, "
                            f"coverage: {coverage['coverage_percentage']:.1f}%")
        else:
            dedup_count += 1

    logger.info(f"Phase 0 done: added {added_count} path points, "
                f"deduped {dedup_count}, skipped {skipped_count}")
    logger.info(f"Coverage after phase 0: {visibility_tracker['coverage_percentage']:.1f}%")

    return path_points


def generate_phase0_path_points_from_trajectories(env_connector,
                                                  traj_data_list, scene_num,
                                                  visibility_tracker):
    """
    Phase 0: Replay trajectories in real-time to generate path points from navigation actions.

    Processing flow:
    1. Load all trajectory data for the specified scene
    2. For each trajectory:
       a. Apply init_action to initialize scene
       b. Execute object_poses to restore object positions
       c. Execute low_actions one by one
       d. Only execute navigation actions, skip interaction actions
       e. Record position and visible objects at each step
    3. Sort by visible object count and deduplicate to generate path points

    Args:
        env_connector: ThorConnector instance
        traj_data_list: List of trajectory data (from load_trajectory_for_scene)
        scene_num: Scene number
        visibility_tracker: Visibility tracking dictionary

    Returns:
        list: List of phase 0 path points
    """
    # Define navigation actions
    NAVIGATION_ACTIONS = {
        'LookUp', 'LookDown', 'RotateLeft', 'RotateRight', 'MoveAhead',
        'MoveBack', 'MoveLeft', 'MoveRight'
    }

    # Define interaction actions (to skip)
    INTERACTION_ACTIONS = {
        'PickupObject', 'PutObject', 'OpenObject', 'CloseObject',
        'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject'
    }

    path_points = []
    all_candidates = []

    logger.info(f"=== Phase 0: replaying navigation actions from trajectories in real time ===")
    logger.info(f"Number of trajectories: {len(traj_data_list)}")

    # Filter trajectories for current scene
    scene_trajectories = [
        td for td in traj_data_list if td['scene']['scene_num'] == scene_num
    ]
    logger.info(f"Trajectories for scene {scene_num}: {len(scene_trajectories)}")

    if not scene_trajectories:
        logger.warning(f"No trajectory data found for scene {scene_num}")
        return path_points

    # Replay each trajectory
    for traj_idx, traj_data in enumerate(scene_trajectories):
        trajectory_id = traj_data.get('task_id', f'traj_{traj_idx}')
        logger.info(
            f"\n[Trajectory {traj_idx+1}/{len(scene_trajectories)}] {trajectory_id}")

        try:
            # Reset scene
            scene_name = f'FloorPlan{scene_num}'
            # env_connector.reset(scene_name)

            # Restore object positions
            # env_connector.restore_scene(traj_data['scene']['object_poses'],
            #                             traj_data['scene']['object_toggles'],
            #                             traj_data['scene']['dirty_and_empty'])

            # Apply initial action
            env_connector.step(dict(traj_data['scene']['init_action']))

            # Record initial state
            agent_pos = env_connector.last_event.metadata['agent']['position']
            agent_rot = env_connector.last_event.metadata['agent']['rotation'][
                'y']
            agent_horizon = env_connector.last_event.metadata['agent'][
                'cameraHorizon']
            visible_objs, total_pixels = get_visible_objects_with_pixel_counts(
                env_connector)

            candidate = {
                'position': agent_pos,
                'rotation': agent_rot,
                'horizon': agent_horizon,
                'visible_objects': visible_objs,
                'total_visible_pixels': total_pixels,
                'source': trajectory_id,
                'step': 0,
                'action': 'INIT'
            }
            all_candidates.append(candidate)

            # Execute low_actions
            low_actions = traj_data.get('plan', {}).get('low_actions', [])
            logger.info(f"  Total actions: {len(low_actions)}")

            step_count = 0
            nav_count = 0
            skip_count = 0

            for action_idx, ll_action in enumerate(low_actions):
                api_action = ll_action.get('api_action', {})
                action_name = api_action.get('action', '')
                discrete_action = ll_action.get('discrete_action',
                                                {}).get('action', action_name)

                # Determine action type
                if action_name in NAVIGATION_ACTIONS:
                    # Execute navigation action
                    params = {}
                    if 'moveMagnitude' in api_action:
                        params['moveMagnitude'] = api_action['moveMagnitude']
                    if 'degrees' in api_action:
                        params['degrees'] = api_action['degrees']
                    if 'forceAction' in api_action:
                        params['forceAction'] = api_action['forceAction']

                    event = env_connector.step(
                        dict(action=action_name, **params))

                    if event.metadata['lastActionSuccess']:
                        # Record state
                        agent_pos = env_connector.last_event.metadata['agent'][
                            'position']
                        agent_rot = env_connector.last_event.metadata['agent'][
                            'rotation']['y']
                        agent_horizon = env_connector.last_event.metadata[
                            'agent']['cameraHorizon']
                        visible_objs, total_pixels = get_visible_objects_with_pixel_counts(
                            env_connector)

                        candidate = {
                            'position': agent_pos,
                            'rotation': agent_rot,
                            'horizon': agent_horizon,
                            'visible_objects': visible_objs,
                            'total_visible_pixels': total_pixels,
                            'source': trajectory_id,
                            'step': action_idx + 1,
                            'action': discrete_action
                        }
                        all_candidates.append(candidate)
                        nav_count += 1

                        if nav_count % 10 == 0:
                            logger.info(f"    Executed {nav_count} navigation actions")
                    else:
                        logger.warning(
                            f"    Action failed: {action_name} (step {action_idx})")

                elif action_name in INTERACTION_ACTIONS:
                    # Skip interaction actions
                    skip_count += 1
                else:
                    logger.warning(f"    Unknown action: {action_name}")

                step_count += 1

            logger.info(
                f"  Done: total steps {step_count}, navigation actions {nav_count}, skipped interactions {skip_count}")

        except Exception as e:
            logger.error(f"  Error replaying trajectory {trajectory_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"Total candidate path points: {len(all_candidates)}")

    if not all_candidates:
        logger.warning("No candidate path points generated")
        return path_points

    # Sort by visible object count (descending)
    all_candidates.sort(key=lambda x: len(x['visible_objects']), reverse=True)
    logger.info(
        f"Sorted by visible object count, range: {len(all_candidates[0]['visible_objects'])} - {len(all_candidates[-1]['visible_objects'])}"
    )

    # Deduplicate and add path points
    added_count = 0
    skipped_count = 0

    for idx, candidate in enumerate(all_candidates):
        visible_set = set(candidate['visible_objects'])
        new_objects = visible_set - visibility_tracker['visible_objects']

        # Skip if no new objects (for efficiency)
        if not new_objects and idx > 0:
            skipped_count += 1
            continue

        # Find nearest non-movable object for reference
        nearest_non_movable = find_nearest_non_movable_object(
            candidate['position'],
            env_connector.last_event.metadata['objects'])

        if not nearest_non_movable:
            logger.warning(f"No reference object found, skipping candidate {idx}")
            continue

        ref_obj_dict = {
            'object_type': nearest_non_movable['object_type'],
            'object_id': nearest_non_movable['object_id'],
            'position': nearest_non_movable['position']
        }

        # Generate location description
        loc_desc = generate_compact_location_description_stage135(
            env_connector, candidate['position'], candidate['rotation'],
            candidate['visible_objects'], ref_obj_dict)

        path_point = {
            'position': candidate['position'],
            'rotation': candidate['rotation'],
            'horizon': candidate['horizon'],
            'loc_desc': loc_desc,
            'visible_objects': candidate['visible_objects'],
            'total_visible_pixels': candidate.get('total_visible_pixels', 0),
            'phase': 0,
            'reference_object': ref_obj_dict['object_id'],
            'reference_object_type': ref_obj_dict['object_type'],
            'source_trajectory': candidate['source'],
            'source_step': candidate['step'],
            'action': candidate.get('action', 'Unknown')
        }

        # Deduplicate
        added, removed_indices, path_points = add_path_point_with_deduplication(
            path_point, path_points)

        if added:
            added_count += 1
            visibility_tracker['visible_objects'].update(visible_set)

            # Recalculate coverage
            coverage = check_visibility_coverage(
                visibility_tracker['all_objects'],
                visibility_tracker['visible_objects'],
                visibility_tracker['objects_in_closed'])
            visibility_tracker['coverage_percentage'] = coverage[
                'coverage_percentage']

            if added_count % 50 == 0:
                logger.info(f"  Progress: {added_count}/{len(all_candidates)}, "
                            f"coverage: {coverage['coverage_percentage']:.1f}%")
        else:
            skipped_count += 1

    logger.info(f"Phase 0 done: added {added_count} path points, deduped {skipped_count}")
    logger.info(f"Coverage after phase 0: {visibility_tracker['coverage_percentage']:.1f}%")

    return path_points


def generate_phase1_path_points(env_connector, reference_objects):
    """
    Phase 1: Generate base path points using reference objects.

    Args:
        env_connector: ThorConnector instance
        reference_objects: List of reference objects

    Returns:
        list: List of path point dictionaries
    """
    path_points = []
    success_count = 0

    logger.info(f"=== Phase 1: generating base path points ===")
    logger.info(f"Number of reference objects: {len(reference_objects)}")

    for i, ref_obj in enumerate(reference_objects):
        obj_id = ref_obj['object_id']
        obj_type = ref_obj['object_type']
        is_container = ref_obj['is_container']

        logger.info(
            f"[{i+1}/{len(reference_objects)}] Processing {obj_type} ({obj_id})")

        # Navigate near the object
        ret_msg = env_connector.nav_obj_by_id(obj_id)
        if ret_msg != "":
            logger.warning(f"  Navigation failed: {ret_msg}")
            continue

        # Get current position
        agent_pos = env_connector.last_event.metadata['agent']['position']

        # Calculate target position (1 unit outward)
        obj_pos = ref_obj['position']
        target_pos = calculate_target_position(obj_pos, agent_pos)

        # Find the closest reachable position
        closest_reachable = env_connector.find_close_reachable_position(
            [target_pos['x'], target_pos['y'], target_pos['z']])

        # Calculate rotation facing the object (same computation as nav_obj_by_id)
        loc = obj_pos
        closest_loc = closest_reachable

        # Calculate desired rotation angle (see nav_obj_by_id)
        rot_angle = math.atan2(-(loc['x'] - closest_loc[0]),
                               loc['z'] - closest_loc[2])
        if rot_angle > 0:
            rot_angle -= 2 * math.pi
        rotation = -(180 / math.pi) * rot_angle  # in degrees

        # Calculate desired horizon angle
        camera_height = env_connector.agent_height + constants.CAMERA_HEIGHT_OFFSET
        xz_dist = math.hypot(loc['x'] - closest_loc[0],
                             loc['z'] - closest_loc[2])
        hor_angle = math.atan2((loc['y'] - camera_height), xz_dist)
        hor_angle = (180 / math.pi) * hor_angle  # in degrees
        horizon = -hor_angle * 0.9  # adjust angle for better view

        # Teleport to the target position
        env_connector.step({
            'action': 'TeleportFull',
            'x': closest_reachable[0],
            'y': env_connector.agent_height,
            'z': closest_reachable[2],
            'rotation': rotation,
            'horizon': horizon,
            'rotateOnTeleport': True
        })

        if not env_connector.last_event.metadata['lastActionSuccess']:
            logger.warning(f"  Teleport failed")
            continue

        # Get visible objects and pixel counts
        visible_objs, total_pixels = get_visible_objects_with_pixel_counts(
            env_connector)

        # Generate location description (using reference object)
        final_pos = env_connector.last_event.metadata['agent']['position']
        final_rot = env_connector.last_event.metadata['agent']['rotation']['y']
        final_horizon = env_connector.last_event.metadata['agent'][
            'cameraHorizon']
        loc_desc = generate_compact_location_description_stage135(
            env_connector, final_pos, final_rot, visible_objs, ref_obj)

        path_point = {
            'position': final_pos,
            'rotation': final_rot,
            'horizon': final_horizon,
            'loc_desc': loc_desc,
            'visible_objects': visible_objs,
            'total_visible_pixels': total_pixels,
            'phase': 1,
            'reference_object': obj_id,
            'reference_object_type': obj_type
        }

        # Add path point with the deduplication function
        added, removed_indices, path_points = add_path_point_with_deduplication(
            path_point, path_points)

        if added:
            success_count += 1
            logger.info(f"  ✓ Success, visible object count: {len(visible_objs)}")
            if removed_indices:
                logger.info(f"    Removed redundant path points: {removed_indices}")
        else:
            logger.info(f"  ⊗ Skipped (fully contained by existing path points)")

    logger.info(f"Phase 1 done: generated {success_count}/{len(reference_objects)} path points")
    return path_points


def check_visibility_coverage(all_objects, visible_objects_set,
                              objects_in_closed):
    """
    Check visibility coverage.

    Args:
        all_objects: Set of all object IDs
        visible_objects_set: Set of observed object IDs
        objects_in_closed: Set of object IDs in closed containers

    Returns:
        dict: Coverage statistics
    """
    eligible_objects = all_objects - objects_in_closed
    visible = eligible_objects & visible_objects_set
    unseen = eligible_objects - visible_objects_set

    coverage = len(visible) / len(
        eligible_objects) * 100 if eligible_objects else 100

    return {
        'total_objects': len(eligible_objects),
        'visible_objects': len(visible),
        'unseen_objects': list(unseen),
        'coverage_percentage': coverage
    }


def generate_phase2_path_points(env_connector, reference_objects,
                                visibility_tracker):
    """
    Phase 2: Navigate to unique non-movable objects, record initial rotation, then perform rotation exploration.

    Args:
        env_connector: ThorConnector instance
        reference_objects: List of unique non-movable objects
        visibility_tracker: Visibility tracking dictionary

    Returns:
        list: List of newly added path points
    """
    new_points = []
    dedup_count = 0  # count of deduplicated points

    logger.info(f"=== Phase 2: rotation exploration ===")
    logger.info(f"Number of reference objects: {len(reference_objects)}")
    logger.info(f"Current coverage: {visibility_tracker['coverage_percentage']:.1f}%")

    for ref_obj in reference_objects:
        obj_id = ref_obj['object_id']
        obj_type = ref_obj['object_type']

        logger.info(f"Processing {obj_type} ({obj_id})")

        # Navigate near the object
        ret_msg = env_connector.nav_obj_by_id(obj_id)
        if ret_msg != "":
            logger.warning(f"  Navigation failed: {ret_msg}")
            continue

        # Get current position and initial rotation
        agent_pos = env_connector.last_event.metadata['agent']['position']
        initial_rotation = env_connector.last_event.metadata['agent'][
            'rotation']['y']

        # Calculate target position (1 unit outward)
        obj_pos = ref_obj['position']
        target_pos = calculate_target_position(obj_pos, agent_pos)

        # Find the closest reachable position
        closest_reachable = env_connector.find_close_reachable_position(
            [target_pos['x'], target_pos['y'], target_pos['z']])

        pos = {
            'x': closest_reachable[0],
            'y': env_connector.agent_height,
            'z': closest_reachable[2]
        }

        # Find the angle closest to the initial rotation
        angle_diffs = []
        for h_angle in HORIZONTAL_ANGLES:
            diff = abs(h_angle - initial_rotation)
            if diff > 180:
                diff = 360 - diff
            angle_diffs.append((h_angle, diff))

        # Sort by angle difference, exclude the closest angle
        angle_diffs.sort(key=lambda x: x[1])
        excluded_angle = angle_diffs[0][0]  # angle closest to the initial rotation
        logger.info(
            f"  Initial rotation: {initial_rotation:.1f}°, excluded angle: {excluded_angle}°")

        # Try the remaining angles
        for h_angle, _ in angle_diffs[1:]:  # skip the first (closest)
            for v_angle in VERTICAL_ANGLES:
                # Teleport to the new angle
                env_connector.step({
                    'action': 'TeleportFull',
                    'x': pos['x'],
                    'y': pos['y'],
                    'z': pos['z'],
                    'rotation': h_angle,
                    'horizon': v_angle,
                    'rotateOnTeleport': True
                })

                if not env_connector.last_event.metadata['lastActionSuccess']:
                    continue

                # Get visible objects and pixel counts
                visible_objs_list, total_pixels = get_visible_objects_with_pixel_counts(
                    env_connector)
                visible_objs = set(visible_objs_list)

                # Check for new objects
                new_objects = visible_objs - visibility_tracker[
                    'visible_objects']

                if new_objects:
                    # Build the reference object dict
                    ref_obj_dict = {
                        'object_type': obj_type,
                        'object_id': obj_id,
                        'position': obj_pos
                    }

                    # Generate location description
                    loc_desc = generate_compact_location_description_stage24(
                        env_connector, pos, h_angle, v_angle,
                        list(visible_objs), ref_obj_dict)

                    new_point = {
                        'position': pos,
                        'rotation': h_angle,
                        'horizon': v_angle,
                        'loc_desc': loc_desc,
                        'visible_objects': list(visible_objs),
                        'total_visible_pixels': total_pixels,
                        'phase': 2,
                        'reference_object': obj_id,
                        'reference_object_type': obj_type,
                        'initial_rotation': initial_rotation
                    }

                    # Dedup within the new phase 2 path points
                    added, removed_indices, new_points = add_path_point_with_deduplication(
                        new_point, new_points)

                    if added:
                        # Update visibility tracking
                        visibility_tracker['visible_objects'].update(
                            visible_objs)

                        # Recalculate coverage
                        coverage = check_visibility_coverage(
                            visibility_tracker['all_objects'],
                            visibility_tracker['visible_objects'],
                            visibility_tracker['objects_in_closed'])
                        visibility_tracker['coverage_percentage'] = coverage[
                            'coverage_percentage']

                        logger.info(
                            f"  Added new path point, found {len(new_objects)} new objects, coverage: {coverage['coverage_percentage']:.1f}%"
                        )

                    else:
                        dedup_count += 1
                        logger.info(f"  Skipped redundant path point (dedup count: {dedup_count})")

    logger.info(f"Phase 2 done: added {len(new_points)} path points, deduped {dedup_count}")
    return new_points


def sort_objects_by_round_robin(reference_objects):
    """
    Sort non-unique reference objects using round-robin to avoid consecutive objects of same type and nearby locations.

    Args:
        reference_objects: List of reference objects

    Returns:
        list: Sorted list of reference objects
    """
    if not reference_objects:
        return []

    # Group by objectType
    type_groups = {}
    for obj in reference_objects:
        obj_type = obj['object_type']
        if obj_type not in type_groups:
            type_groups[obj_type] = []
        type_groups[obj_type].append(obj)

    # Round-robin selection: take one object from each type in turn
    sorted_objects = []
    max_group_size = max(len(group) for group in type_groups.values())

    for i in range(max_group_size):
        for obj_type in sorted(type_groups.keys()):
            if i < len(type_groups[obj_type]):
                sorted_objects.append(type_groups[obj_type][i])

    return sorted_objects


def generate_phase3_path_points(env_connector, reference_objects,
                                visibility_tracker):
    """
    Phase 3: Generate path points using non-unique reference objects (in round-robin order).

    Args:
        env_connector: ThorConnector instance
        reference_objects: List of reference objects
        visibility_tracker: Visibility tracking dictionary

    Returns:
        tuple: (new_path_points_list, phase3_path_points_list_with_full_info)
    """
    new_points = []
    phase3_points = []  # store full phase 3 path point info for phase 4
    dedup_count = 0

    logger.info(f"=== Phase 3: using non-unique reference objects ===")
    logger.info(f"Number of reference objects: {len(reference_objects)}")

    # Round-robin sort the non-unique reference objects
    sorted_reference_objects = sort_objects_by_round_robin(reference_objects)
    logger.info(f"Object count after round-robin sort: {len(sorted_reference_objects)}")
    logger.info(f"Current coverage: {visibility_tracker['coverage_percentage']:.1f}%")

    for i, ref_obj in enumerate(sorted_reference_objects):
        obj_id = ref_obj['object_id']
        obj_type = ref_obj['object_type']

        logger.info(
            f"[{i+1}/{len(sorted_reference_objects)}] Processing {obj_type} ({obj_id})"
        )

        # Navigate near the object
        ret_msg = env_connector.nav_obj_by_id(obj_id)
        if ret_msg != "":
            logger.warning(f"  Navigation failed: {ret_msg}")
            continue

        # Get current position and initial rotation
        agent_pos = env_connector.last_event.metadata['agent']['position']
        initial_rotation = env_connector.last_event.metadata['agent'][
            'rotation']['y']

        # Calculate target position (1 unit outward)
        obj_pos = ref_obj['position']
        target_pos = calculate_target_position(obj_pos, agent_pos)

        # Find the closest reachable position
        closest_reachable = env_connector.find_close_reachable_position(
            [target_pos['x'], target_pos['y'], target_pos['z']])

        # Calculate rotation facing the object (same computation as nav_obj_by_id)
        loc = obj_pos
        closest_loc = closest_reachable

        # Calculate desired rotation angle (see nav_obj_by_id)
        rot_angle = math.atan2(-(loc['x'] - closest_loc[0]),
                               loc['z'] - closest_loc[2])
        if rot_angle > 0:
            rot_angle -= 2 * math.pi
        rotation = -(180 / math.pi) * rot_angle  # in degrees

        # Calculate desired horizon angle
        camera_height = env_connector.agent_height + constants.CAMERA_HEIGHT_OFFSET
        xz_dist = math.hypot(loc['x'] - closest_loc[0],
                             loc['z'] - closest_loc[2])
        hor_angle = math.atan2((loc['y'] - camera_height), xz_dist)
        hor_angle = (180 / math.pi) * hor_angle  # in degrees
        horizon = -hor_angle * 0.9  # adjust angle for better view

        # Teleport to the target position
        env_connector.step({
            'action': 'TeleportFull',
            'x': closest_reachable[0],
            'y': env_connector.agent_height,
            'z': closest_reachable[2],
            'rotation': rotation,
            'horizon': horizon,
            'rotateOnTeleport': True
        })

        if not env_connector.last_event.metadata['lastActionSuccess']:
            logger.warning(f"  Teleport failed")
            continue

        # Get visible objects and pixel counts
        visible_objs_list, total_pixels = get_visible_objects_with_pixel_counts(
            env_connector)
        visible_objs = set(visible_objs_list)

        # Check for new objects
        new_objects = visible_objs - visibility_tracker['visible_objects']

        if new_objects:
            # Generate location description (using reference object)
            final_pos = env_connector.last_event.metadata['agent']['position']
            final_rot = env_connector.last_event.metadata['agent']['rotation'][
                'y']
            final_horizon = env_connector.last_event.metadata['agent'][
                'cameraHorizon']
            loc_desc = generate_compact_location_description_stage135(
                env_connector, final_pos, final_rot, list(visible_objs),
                ref_obj)

            new_point = {
                'position': final_pos,
                'rotation': final_rot,
                'horizon': final_horizon,
                'loc_desc': loc_desc,
                'visible_objects': list(visible_objs),
                'total_visible_pixels': total_pixels,
                'phase': 3,
                'reference_object': obj_id,
                'reference_object_type': obj_type,
                'initial_rotation': initial_rotation
            }

            # Dedup within the new phase 3 path points
            added, removed_indices, new_points = add_path_point_with_deduplication(
                new_point, new_points)

            if added:
                # Save the full phase 3 path point info
                phase3_points.append(new_point)

                # Update visibility tracking
                visibility_tracker['visible_objects'].update(visible_objs)

                # Recalculate coverage
                coverage = check_visibility_coverage(
                    visibility_tracker['all_objects'],
                    visibility_tracker['visible_objects'],
                    visibility_tracker['objects_in_closed'])
                visibility_tracker['coverage_percentage'] = coverage[
                    'coverage_percentage']

                logger.info(
                    f"  ✓ Success, found {len(new_objects)} new objects, coverage: {coverage['coverage_percentage']:.1f}%"
                )
            else:
                dedup_count += 1
                logger.info(f"  ⊗ Skipped redundant path point")
        else:
            logger.info(f"  No new objects found, skipping")

    logger.info(f"Phase 3 done: added {len(new_points)} path points, deduped {dedup_count}")
    return new_points, phase3_points


def generate_phase4_path_points(env_connector, phase3_points,
                                visibility_tracker):
    """
    Phase 4: Explore different angles at phase 3 locations (following phase 2 logic).

    Args:
        env_connector: ThorConnector instance
        phase3_points: List of phase 3 path points (with full info)
        visibility_tracker: Visibility tracking dictionary

    Returns:
        list: List of newly added path points
    """
    new_points = []
    dedup_count = 0  # count of deduplicated points

    logger.info(f"=== Phase 4: rotation exploration on non-unique reference objects ===")
    logger.info(f"Checking {len(phase3_points)} phase 3 path points")
    logger.info(f"Current coverage: {visibility_tracker['coverage_percentage']:.1f}%")

    for point in phase3_points:

        pos = point['position']
        obj_id = point['reference_object']
        obj_type = point['reference_object_type']
        initial_rotation = point['initial_rotation']

        logger.info(f"Processing {obj_type} ({obj_id})")

        # Find the angle closest to the initial rotation
        angle_diffs = []
        for h_angle in HORIZONTAL_ANGLES:
            diff = abs(h_angle - initial_rotation)
            if diff > 180:
                diff = 360 - diff
            angle_diffs.append((h_angle, diff))

        # Sort by angle difference, exclude the closest angle
        angle_diffs.sort(key=lambda x: x[1])
        excluded_angle = angle_diffs[0][0]  # angle closest to the initial rotation
        logger.info(
            f"  Initial rotation: {initial_rotation:.1f}°, excluded angle: {excluded_angle}°")

        # Try the remaining angles
        for h_angle, _ in angle_diffs[1:]:  # skip the first (closest)
            for v_angle in VERTICAL_ANGLES:
                # Teleport to the new angle
                env_connector.step({
                    'action': 'TeleportFull',
                    'x': pos['x'],
                    'y': pos['y'],
                    'z': pos['z'],
                    'rotation': h_angle,
                    'horizon': v_angle,
                    'rotateOnTeleport': True
                })

                if not env_connector.last_event.metadata['lastActionSuccess']:
                    continue

                # Get visible objects and pixel counts
                visible_objs_list, total_pixels = get_visible_objects_with_pixel_counts(
                    env_connector)
                visible_objs = set(visible_objs_list)

                # Check for new objects
                new_objects = visible_objs - visibility_tracker[
                    'visible_objects']

                if new_objects:
                    # Get reference object info
                    all_objects = env_connector.last_event.metadata['objects']
                    ref_obj_info = next(
                        (o for o in all_objects
                         if o['objectId'] == point['reference_object']), None)

                    # Build the reference object dict
                    ref_obj_dict = {
                        'object_type':
                        obj_type,
                        'object_id':
                        obj_id,
                        'position':
                        ref_obj_info['position'] if ref_obj_info else pos
                    }

                    # Generate location description
                    loc_desc = generate_compact_location_description_stage24(
                        env_connector, pos, h_angle, v_angle,
                        list(visible_objs), ref_obj_dict)

                    new_point = {
                        'position': pos,
                        'rotation': h_angle,
                        'horizon': v_angle,
                        'loc_desc': loc_desc,
                        'visible_objects': list(visible_objs),
                        'total_visible_pixels': total_pixels,
                        'phase': 4,
                        'reference_object': obj_id,
                        'reference_object_type': obj_type,
                        'initial_rotation': initial_rotation
                    }

                    # Dedup within the new phase 4 path points
                    added, removed_indices, new_points = add_path_point_with_deduplication(
                        new_point, new_points)

                    if added:
                        # Update visibility tracking
                        visibility_tracker['visible_objects'].update(
                            visible_objs)

                        # Recalculate coverage
                        coverage = check_visibility_coverage(
                            visibility_tracker['all_objects'],
                            visibility_tracker['visible_objects'],
                            visibility_tracker['objects_in_closed'])
                        visibility_tracker['coverage_percentage'] = coverage[
                            'coverage_percentage']

                        logger.info(
                            f"  Added new path point, found {len(new_objects)} new objects, coverage: {coverage['coverage_percentage']:.1f}%"
                        )

                    else:
                        dedup_count += 1
                        logger.info(f"  Skipped redundant path point (dedup count: {dedup_count})")

    logger.info(f"Phase 4 done: added {len(new_points)} path points, deduped {dedup_count}")
    return new_points


def generate_phase5_path_points(env_connector, objects_metadata,
                                visibility_tracker):
    """
    Phase 5: Fill-in stage, use remaining unseen objects as reference objects to generate path points.

    Note: This function assumes global deduplication and visibility_tracker reset/recheck are completed before execution.

    Processing logic:
    1. Find all unseen objects (excluding objects in closed containers)
    2. Sort by priority: unique non-movable > non-unique non-movable > movable
    3. For each unseen object, navigate nearby and generate path point
    4. If new objects discovered:
       - Find nearest non-movable object as reference
       - Use reference object to generate location description
       - Keep path point
    5. Skip if no new objects discovered
    6. Early termination when 100% coverage reached

    Args:
        env_connector: ThorConnector instance
        objects_metadata: List of all object metadata in the scene
        visibility_tracker: Visibility tracking dictionary (reset and rechecked)

    Returns:
        list: List of newly added path points
    """
    new_points = []
    dedup_count = 0

    logger.info(f"=== Phase 5: using unseen objects as reference objects ===")
    logger.info(f"Current coverage: {visibility_tracker['coverage_percentage']:.1f}%")

    # if visibility_tracker['coverage_percentage'] >= 100:
    #     logger.info("Coverage already 100%, skipping phase 5")
    #     return new_points

    # Find all unseen objects (excluding objects in closed containers)
    eligible_objects = visibility_tracker['all_objects'] - visibility_tracker[
        'visible_objects'] - visibility_tracker['objects_in_closed']

    if not eligible_objects:
        logger.info("No unseen objects, skipping phase 5")
        return new_points

    logger.info(f"Number of unseen objects: {len(eligible_objects)}")

    # Group by object type, process unique objects first
    object_tree = build_object_tree_with_uniqueness(objects_metadata)
    unseen_objects_info = []

    for obj_id in eligible_objects:
        obj = next((o for o in objects_metadata if o['objectId'] == obj_id),
                   None)
        if obj:
            unseen_objects_info.append(obj)

    # Prioritize unique non-movable objects, then non-unique non-movable, then movable
    unique_non_movable = []
    non_unique_non_movable = []
    movable = []

    for obj in unseen_objects_info:
        obj_id = obj['objectId']
        if obj_id in object_tree['all_objects']:
            obj_info = object_tree['all_objects'][obj_id]
            if not obj['pickupable']:
                # Check uniqueness
                type_counts = {}
                for o in objects_metadata:
                    obj_type = o['objectType']
                    type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

                if type_counts[obj['objectType']] == 1:
                    unique_non_movable.append(obj)
                else:
                    non_unique_non_movable.append(obj)
            else:
                movable.append(obj)

    # Process by priority
    priority_objects = unique_non_movable + non_unique_non_movable + movable

    logger.info(f"  Unique non-movable objects: {len(unique_non_movable)}")
    logger.info(f"  Non-unique non-movable objects: {len(non_unique_non_movable)}")
    logger.info(f"  Movable objects: {len(movable)}")

    for i, ref_obj in enumerate(priority_objects):

        if visibility_tracker['coverage_percentage'] >= 100:
            logger.info("Coverage reached 100%, stopping phase 5")
            break

        obj_id = ref_obj['objectId']
        obj_type = ref_obj['objectType']

        logger.info(
            f"[{i+1}/{len(priority_objects)}] Processing {obj_type} ({obj_id})")

        # Navigate near the object
        ret_msg = env_connector.nav_obj_by_id(obj_id)
        if ret_msg != "":
            logger.warning(f"  Navigation failed: {ret_msg}")
            continue
        # Get current position and initial rotation
        agent_pos = env_connector.last_event.metadata['agent']['position']
        initial_rotation = env_connector.last_event.metadata['agent'][
            'rotation']['y']
        obj_pos = ref_obj['position']

        # Get visible objects and pixel counts
        visible_objs_list, total_pixels = get_visible_objects_with_pixel_counts(
            env_connector)
        visible_objs = set(visible_objs_list)

        # Check for new objects
        new_objects = visible_objs - visibility_tracker['visible_objects']

        if not new_objects:
            # Find the angle closest to the initial rotation
            angle_diffs = []
            for h_angle in HORIZONTAL_ANGLES:
                diff = abs(h_angle - initial_rotation)
                if diff > 180:
                    diff = 360 - diff
                angle_diffs.append((h_angle, diff))

            # Sort by angle difference, exclude the closest angle
            angle_diffs.sort(key=lambda x: x[1])

            # Try the remaining angles
            for h_angle, _ in angle_diffs[1:]:  # skip the first (closest)
                for v_angle in VERTICAL_ANGLES:
                    # Teleport to the new angle
                    env_connector.step({
                        'action': 'TeleportFull',
                        'x': agent_pos['x'],
                        'y': agent_pos['y'],
                        'z': agent_pos['z'],
                        'rotation': h_angle,
                        'horizon': v_angle,
                        'rotateOnTeleport': True
                    })

                    if not env_connector.last_event.metadata[
                            'lastActionSuccess']:
                        continue

                    # Get visible objects and pixel counts
                    visible_objs_list, total_pixels = get_visible_objects_with_pixel_counts(
                        env_connector)
                    visible_objs = set(visible_objs_list)

                    # Check for new objects
                    new_objects = visible_objs - visibility_tracker[
                        'visible_objects']

                    if new_objects:
                        break

        if not new_objects:

            # Calculate target position (1 unit outward)
            target_pos = calculate_target_position(obj_pos, agent_pos)

            # Find the closest reachable position
            closest_reachable = env_connector.find_close_reachable_position(
                [target_pos['x'], target_pos['y'], target_pos['z']])

            # Calculate rotation facing the object (same computation as nav_obj_by_id)
            loc = obj_pos
            closest_loc = closest_reachable

            # Calculate desired rotation angle (see nav_obj_by_id)
            rot_angle = math.atan2(-(loc['x'] - closest_loc[0]),
                                   loc['z'] - closest_loc[2])
            if rot_angle > 0:
                rot_angle -= 2 * math.pi
            rotation = -(180 / math.pi) * rot_angle  # in degrees

            # Calculate desired horizon angle
            camera_height = env_connector.agent_height + constants.CAMERA_HEIGHT_OFFSET
            xz_dist = math.hypot(loc['x'] - closest_loc[0],
                                 loc['z'] - closest_loc[2])
            hor_angle = math.atan2((loc['y'] - camera_height), xz_dist)
            hor_angle = (180 / math.pi) * hor_angle  # in degrees
            horizon = -hor_angle * 0.9  # adjust angle for better view

            # Teleport to the target position
            env_connector.step({
                'action': 'TeleportFull',
                'x': closest_reachable[0],
                'y': env_connector.agent_height,
                'z': closest_reachable[2],
                'rotation': rotation,
                'horizon': horizon,
                'rotateOnTeleport': True
            })

            if not env_connector.last_event.metadata['lastActionSuccess']:
                logger.warning(f"  Teleport failed")
                continue

            # Get visible objects and pixel counts
            visible_objs_list, total_pixels = get_visible_objects_with_pixel_counts(
                env_connector)
            visible_objs = set(visible_objs_list)

            # Check for new objects
            new_objects = visible_objs - visibility_tracker['visible_objects']

        if not new_objects:
            # Find the angle closest to the initial rotation
            angle_diffs = []
            for h_angle in HORIZONTAL_ANGLES:
                diff = abs(h_angle - initial_rotation)
                if diff > 180:
                    diff = 360 - diff
                angle_diffs.append((h_angle, diff))

            # Sort by angle difference, exclude the closest angle
            angle_diffs.sort(key=lambda x: x[1])

            # Try the remaining angles
            for h_angle, _ in angle_diffs[1:]:  # skip the first (closest)
                for v_angle in VERTICAL_ANGLES:
                    # Teleport to the new angle
                    env_connector.step({
                        'action': 'TeleportFull',
                        'x': closest_reachable[0],
                        'y': env_connector.agent_height,
                        'z': closest_reachable[2],
                        'rotation': h_angle,
                        'horizon': v_angle,
                        'rotateOnTeleport': True
                    })

                    if not env_connector.last_event.metadata[
                            'lastActionSuccess']:
                        continue

                    # Get visible objects and pixel counts
                    visible_objs_list, total_pixels = get_visible_objects_with_pixel_counts(
                        env_connector)
                    visible_objs = set(visible_objs_list)

                    # Check for new objects
                    new_objects = visible_objs - visibility_tracker[
                        'visible_objects']

                    if new_objects:
                        break

        if new_objects:
            # Find the nearest non-movable object as reference object
            final_pos = env_connector.last_event.metadata['agent']['position']
            nearest_non_movable = find_nearest_non_movable_object(
                final_pos, objects_metadata)

            ref_obj_dict = {
                'object_type': nearest_non_movable['object_type'],
                'object_id': nearest_non_movable['object_id'],
                'position': nearest_non_movable['position']
            }

            logger.info(
                f"  Reference object: {ref_obj_dict['object_type']} ({ref_obj_dict['object_id']})"
            )

            # Generate location description
            final_rot = env_connector.last_event.metadata['agent']['rotation'][
                'y']
            final_horizon = env_connector.last_event.metadata['agent'][
                'cameraHorizon']
            loc_desc = generate_compact_location_description_stage135(
                env_connector, final_pos, final_rot, list(visible_objs),
                ref_obj_dict)

            new_point = {
                'position': final_pos,
                'rotation': final_rot,
                'horizon': final_horizon,
                'loc_desc': loc_desc,
                'visible_objects': list(visible_objs),
                'total_visible_pixels': total_pixels,
                'phase': 5,
                'reference_object': ref_obj_dict['object_id'],
                'reference_object_type': ref_obj_dict['object_type'],
                'initial_rotation': initial_rotation
            }

            # Use the deduplication function
            added, removed_indices, new_points = add_path_point_with_deduplication(
                new_point, new_points)

            if added:
                # Update visibility tracking
                visibility_tracker['visible_objects'].update(visible_objs)

                # Recalculate coverage
                coverage = check_visibility_coverage(
                    visibility_tracker['all_objects'],
                    visibility_tracker['visible_objects'],
                    visibility_tracker['objects_in_closed'])
                visibility_tracker['coverage_percentage'] = coverage[
                    'coverage_percentage']

                logger.info(
                    f"  ✓ Success, found {len(new_objects)} new objects, coverage: {coverage['coverage_percentage']:.1f}%"
                )
            else:
                dedup_count += 1
                logger.info(f"  ⊗ Skipped redundant path point")
        else:
            logger.info(f"  No new objects found, skipping")

    logger.info(f"Phase 5 done: added {len(new_points)} path points, deduped {dedup_count}")
    return new_points


def get_objects_in_closed_containers(objects_metadata):
    """
    Get all objects in closed containers.

    Args:
        objects_metadata: List of object metadata from AI2-THOR

    Returns:
        set: Set of object IDs in closed containers
    """
    object_info = {}
    for obj in objects_metadata:
        obj_id = obj['objectId']
        object_info[obj_id] = {
            'object_type': obj['objectType'],
            'isOpen': obj.get('isOpen', False),
            'openable': obj.get('openable', False),
            'parent_receptacles': obj.get('parentReceptacles', []),
            'children': []
        }

    objects_in_closed = set()
    for obj_id, info in object_info.items():
        if info['parent_receptacles'] is not None:
            for parent_id in info['parent_receptacles']:
                if object_info[parent_id][
                        'openable'] and not object_info[parent_id]['isOpen']:
                    objects_in_closed.add(obj_id)
                    break

    return objects_in_closed


def save_path_points_with_descriptions(scene_name, path_points, coverage_stats,
                                       output_dir):
    """
    Save path points to JSON file.

    Args:
        scene_name: Scene name
        path_points: List of path points
        coverage_stats: Coverage statistics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{scene_name}-path-points.json")

    json_data = {
        'path_points': [{
            'x': point['position']['x'],
            'y': point['position']['y'],
            'z': point['position']['z'],
            'rotation': point['rotation'],
            'horizon': point['horizon'],
            'loc_desc': point['loc_desc']
        } for point in path_points],
        'metadata': {
            'scene': scene_name,
            'total_path_points': len(path_points)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"Path points saved to: {output_file}")


def generate_markdown_report(scene_name,
                             path_points,
                             object_tree,
                             coverage_stats,
                             output_dir,
                             verification_results=None,
                             verified_coverage=None):
    """
    Generate detailed report in Markdown format.

    Args:
        scene_name: Scene name
        path_points: List of path points
        object_tree: Object tree structure
        coverage_stats: Coverage statistics
        output_dir: Output directory
        verification_results: Verification results (optional)
        verified_coverage: Verified coverage (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir,
                               f"{scene_name}-path-points-report.md")

    with open(report_file, 'w', encoding='utf-8') as f:
        # Title
        f.write(f"# Path Point Generation Report: {scene_name}\n\n")

        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total objects: {coverage_stats['total_objects']}\n")
        f.write(f"- Visible objects: {coverage_stats['visible_objects']}\n")
        f.write(f"- Coverage: {coverage_stats['coverage_percentage']:.1f}%\n")
        f.write(f"- Total path points: {len(path_points)}\n\n")

        # Add verification statistics if verification results exist
        if verification_results and verified_coverage:
            f.write("## Verification Statistics\n\n")
            f.write(
                f"- Verified coverage: {verified_coverage['coverage_percentage']:.1f}%\n")
            f.write(f"- Visible objects after verification: {verified_coverage['visible_objects']}\n")

            successful_verifications = sum(
                1 for r in verification_results.values() if r['success'])
            f.write(
                f"- Successfully verified path points: {successful_verifications}/{len(path_points)}\n\n"
            )

        # Per-phase statistics
        f.write("## Path Points by Phase\n\n")
        for phase in [-1, 0, 1, 2, 3, 4, 5]:
            phase_points = [p for p in path_points if p['phase'] == phase]
            phase_name = "init position" if phase == -1 else f"phase {phase}"
            f.write(f"- {phase_name}: {len(phase_points)} path points\n")
        f.write("\n")

        # Path point details
        f.write("## Path Point Details\n\n")
        for phase in [-1, 0, 1, 2, 3, 4, 5]:
            phase_points = [p for p in path_points if p['phase'] == phase]
            if not phase_points:
                continue

            phase_name = "init position" if phase == -1 else f"phase {phase}"
            f.write(f"### {phase_name} ({len(phase_points)} path points)\n\n")

            for i, point in enumerate(phase_points, 1):
                f.write(f"#### Path Point {i}\n")
                f.write(f"- Reference object: {point['reference_object']}\n")
                f.write(
                    f"- Position: ({point['position']['x']:.2f}, {point['position']['y']:.2f}, {point['position']['z']:.2f})\n"
                )
                f.write(f"- Rotation: {point['rotation']:.1f}°\n")
                f.write(f"- Horizon: {point['horizon']:.1f}°\n")
                f.write(f"- Location description: {point['loc_desc']}\n")
                f.write(f"- Visible object count at generation: {len(point['visible_objects'])}\n")
                f.write(f"- Visible objects at generation: {point['visible_objects']}\n")

                # For phase 0, add source information
                if phase == 0:
                    f.write(
                        f"- Source trajectory: {point.get('source_trajectory', 'N/A')}\n")
                    f.write(f"- Source step: {point.get('source_step', 'N/A')}\n")

                # Add verification info
                if verification_results:
                    loc_desc = point['loc_desc']
                    if loc_desc in verification_results:
                        result = verification_results[loc_desc]
                        if result['success']:
                            # Add image reference
                            rel_image_path = os.path.relpath(
                                result['image_path'], output_dir)
                            f.write(
                                f"- Viewpoint image: [{rel_image_path}]({rel_image_path})\n"
                            )
                            f.write(
                                f"- Visible object count at verification: {len(result['visible_objects'])}\n"
                            )
                            f.write(
                                f"- Visible objects at verification: {result['visible_objects']}\n")
                        else:
                            f.write(f"- Verification: teleport failed\n")

                f.write("\n")

        # Full object tree
        f.write("## Full Object Tree\n\n")
        f.write(f"Total objects: {object_tree['total_objects']}\n")
        f.write(f"Top-level objects: {object_tree['root_objects_count']}\n")
        f.write(f"Containers: {object_tree['container_count']}\n\n")

        # Unique non-movable top-level objects
        f.write("### Unique Non-Movable Top-Level Objects\n\n")
        unique_objects = object_tree['unique_non_movable_top_level']
        for obj in unique_objects:
            obj_id = obj['object_id']
            obj_type = obj['object_type']
            f.write(f"#### {obj_type} ({obj_id})\n")

            # Find path points that see this object
            visible_at_points = []
            for idx, point in enumerate(path_points):
                if obj_id in point['visible_objects']:
                    visible_at_points.append(idx)

            f.write(
                f"- Position: ({obj['position']['x']:.2f}, {obj['position']['y']:.2f}, {obj['position']['z']:.2f})\n"
            )
            f.write(f"- Visible path point count: {len(visible_at_points)}\n")
            if visible_at_points:
                f.write(f"- Visible path point indices: {visible_at_points}\n")
            f.write("\n")

        # Non-unique non-movable top-level objects
        f.write("### Non-Unique Non-Movable Top-Level Objects\n\n")
        non_unique_objects = object_tree['non_unique_non_movable_top_level']
        # Group by type
        type_groups = {}
        for obj in non_unique_objects:
            obj_type = obj['object_type']
            if obj_type not in type_groups:
                type_groups[obj_type] = []
            type_groups[obj_type].append(obj)

        for obj_type, objects in sorted(type_groups.items()):
            f.write(f"#### {obj_type} ({len(objects)} total)\n\n")
            for obj in objects:
                obj_id = obj['object_id']
                f.write(f"- **{obj_id}**\n")
                f.write(
                    f"  - Position: ({obj['position']['x']:.2f}, {obj['position']['y']:.2f}, {obj['position']['z']:.2f})\n"
                )

                # Find path points that see this object
                visible_at_points = []
                for idx, point in enumerate(path_points):
                    if obj_id in point['visible_objects']:
                        visible_at_points.append(idx)

                f.write(f"  - Visible path point count: {len(visible_at_points)}\n")
                if visible_at_points:
                    f.write(f"  - Visible path point indices: {visible_at_points}\n")
                f.write("\n")

        # Objects in containers
        f.write("### Objects in Containers\n\n")
        all_objects = object_tree['all_objects']
        objects_in_containers = {}

        for obj_id, obj_info in all_objects.items():
            if obj_info['parent_receptacles']:
                # This object is in a container
                for parent_id in obj_info['parent_receptacles']:
                    if parent_id not in objects_in_containers:
                        objects_in_containers[parent_id] = []
                    objects_in_containers[parent_id].append(obj_info)

        if objects_in_containers:
            for parent_id, contained_objs in objects_in_containers.items():
                parent_obj = all_objects.get(parent_id)
                if parent_obj:
                    f.write(
                        f"#### Container: {parent_obj['object_type']} ({parent_id})\n\n"
                    )
                    f.write(
                        f"- Position: ({parent_obj['position']['x']:.2f}, {parent_obj['position']['y']:.2f}, {parent_obj['position']['z']:.2f})\n"
                    )
                    f.write(f"- Contained objects: {len(contained_objs)}\n\n")

                    for obj in contained_objs:
                        f.write(
                            f"- **{obj['object_type']}** ({obj['object_id']})\n"
                        )

                        # Find path points that see this object
                        visible_at_points = []
                        for idx, point in enumerate(path_points):
                            if obj['object_id'] in point['visible_objects']:
                                visible_at_points.append(idx)

                        pickupable = "pickupable" if obj['pickupable'] else "non-pickupable"
                        f.write(f"  - {pickupable}\n")
                        f.write(f"  - Visible path point count: {len(visible_at_points)}\n")
                        if visible_at_points:
                            f.write(f"  - Visible path point indices: {visible_at_points}\n")
                        f.write("\n")
        else:
            f.write("(None)\n\n")

        # Top-level movable objects
        f.write("### Top-Level Movable Objects\n\n")
        movable_root_objects = []
        for obj_id, obj_info in all_objects.items():
            if not obj_info['parent_receptacles'] and obj_info['pickupable']:
                movable_root_objects.append(obj_info)

        if movable_root_objects:
            # Group by type
            type_groups = {}
            for obj in movable_root_objects:
                obj_type = obj['object_type']
                if obj_type not in type_groups:
                    type_groups[obj_type] = []
                type_groups[obj_type].append(obj)

            for obj_type, objects in sorted(type_groups.items()):
                f.write(f"#### {obj_type} ({len(objects)} total)\n\n")
                for obj in objects:
                    obj_id = obj['object_id']
                    f.write(f"- **{obj_id}**\n")
                    f.write(
                        f"  - Position: ({obj['position']['x']:.2f}, {obj['position']['y']:.2f}, {obj['position']['z']:.2f})\n"
                    )

                    # Find path points that see this object
                    visible_at_points = []
                    for idx, point in enumerate(path_points):
                        if obj_id in point['visible_objects']:
                            visible_at_points.append(idx)

                    f.write(f"  - Visible path point count: {len(visible_at_points)}\n")
                    if visible_at_points:
                        f.write(f"  - Visible path point indices: {visible_at_points}\n")
                    f.write("\n")
        else:
            f.write("(None)\n\n")

        # Unobserved objects
        if coverage_stats['unseen_objects']:
            f.write("## Unobserved Objects\n\n")
            for obj_id in coverage_stats['unseen_objects']:
                obj = object_tree['all_objects'].get(obj_id)
                if obj:
                    f.write(f"- {obj['object_type']} ({obj_id})\n")
                    f.write(
                        f"  - Position: ({obj['position']['x']:.2f}, {obj['position']['y']:.2f}, {obj['position']['z']:.2f})\n"
                    )
            f.write("\n")

    logger.info(f"Markdown report saved to: {report_file}")


def verify_and_save_path_point_images(env_connector, scene_name, path_points):
    """
    Teleport to each path point, save images, and re-verify visible objects.

    Args:
        env_connector: ThorConnector instance
        scene_name: Scene name
        path_points: List of path points

    Returns:
        dict: Verification results containing actual visible objects for each path point
    """
    logger.info(f"=== Starting path point verification and image saving ===")

    # Create the image output directory
    image_dir = os.path.join(REPORT_DIR, f'{scene_name}_path_point_images')
    os.makedirs(image_dir, exist_ok=True)

    verification_results = {}

    for idx, point in enumerate(path_points):
        loc_desc = point['loc_desc']
        logger.info(f"[{idx+1}/{len(path_points)}] Verifying path point: {loc_desc[:60]}...")

        # Teleport to the path point
        x, y, z = point['position']['x'], point['position']['y'], point[
            'position']['z']
        rotation = point['rotation']
        horizon = point['horizon']

        # Teleport to the specified position and angle
        event = env_connector.step(
            dict(action='TeleportFull',
                 x=x,
                 y=y,
                 z=z,
                 rotation=rotation,
                 horizon=horizon,
                 rotateOnTeleport=True))

        if not env_connector.last_event.metadata['lastActionSuccess']:
            logger.warning(f"  ⚠ Teleport failed, skipping this path point")
            verification_results[loc_desc] = {
                'success': False,
                'visible_objects': [],
                'image_path': None
            }
            continue

        # Get actually visible objects and pixel counts
        visible_obj_ids, total_pixels = get_visible_objects_with_pixel_counts(
            env_connector)
        logger.info(f"  Actual visible object count: {len(visible_obj_ids)}, total pixels: {total_pixels}")

        # Save image using a simplified loc_desc as filename
        # Replace characters unsuitable for filenames
        safe_desc = loc_desc.replace(' ', '_').replace(',',
                                                       '').replace('/',
                                                                   '_')[:50]
        image_filename = f'{safe_desc}.png'
        image_path = os.path.join(image_dir, image_filename)

        from PIL import Image
        frame = env_connector.last_event.frame
        # Convert numpy array to PIL Image
        if isinstance(frame, np.ndarray):
            pil_image = Image.fromarray(frame)
        else:
            pil_image = frame
        pil_image.save(image_path)
        logger.info(f"  Image saved: {image_path}")

        # Record verification result keyed by loc_desc
        verification_results[loc_desc] = {
            'success': True,
            'visible_objects': visible_obj_ids,
            'image_path': image_path,
            'image_filename': image_filename
        }

    # Check loc_desc uniqueness
    if len(verification_results) != len(path_points):
        logger.error(
            f"⚠️ Warning: duplicate loc_desc found! Verification result count ({len(verification_results)}) != path point count ({len(path_points)})"
        )
        duplicates = [
            desc for desc in [p['loc_desc'] for p in path_points]
            if [p['loc_desc'] for p in path_points].count(desc) > 1
        ]
        for dup in set(duplicates):
            count = [p['loc_desc'] for p in path_points].count(dup)
            logger.error(f"  Duplicate loc_desc: {dup[:60]}... (appears {count} times)")
    else:
        logger.info(f"✓ All loc_desc values are unique ({len(verification_results)} total)")

    logger.info(f"=== Verification done, images saved in: {image_dir} ===")

    return verification_results


def filter_overlapping_objects_by_distance(object_poses,
                                           distance_threshold=0.3,
                                           priority_types=None):
    """
    Filter out objects that are too close to each other in 3D space.

    Priority objects (from priority_types) are always kept and can exclude
    other nearby objects from being added.

    Args:
        object_poses: List of object pose dictionaries
        distance_threshold: Distance threshold in meters (default: 0.3)
        priority_types: Set of object types to prioritize (default: None)

    Returns:
        list: Filtered list of object poses
    """
    if priority_types is None:
        priority_types = set()

    filtered_poses = []

    # Separate priority and other objects
    priority_poses = []
    other_poses = []

    for obj in object_poses:
        obj_type = obj.get('objectType',
                           obj.get('objectName', '').split('_')[0])
        if obj_type in priority_types:
            priority_poses.append(obj)
        else:
            other_poses.append(obj)

    # Add priority objects first (without distance checking)
    for obj in priority_poses:
        filtered_poses.append(obj)

    # Add other objects (checking distance against already-added objects)
    for new_obj in other_poses:
        new_pos = new_obj['position']
        is_overlapping = False

        for existing_obj in filtered_poses:
            existing_pos = existing_obj['position']

            # Calculate 3D Euclidean distance
            distance = math.sqrt((new_pos['x'] - existing_pos['x'])**2 +
                                 (new_pos['y'] - existing_pos['y'])**2 +
                                 (new_pos['z'] - existing_pos['z'])**2)

            if distance < distance_threshold:
                is_overlapping = True
                break

        if not is_overlapping:
            filtered_poses.append(new_obj)

    return filtered_poses


def extract_interacting_objects_from_trajectories(traj_data_list):
    """
    Extract object types involved in interaction actions from trajectory data.

    Parses plan.low_actions to identify interaction actions (PickupObject, PutObject,
    OpenObject, etc.) and extracts objectId converted to objectType.

    Args:
        traj_data_list: List of trajectory data dictionaries

    Returns:
        set: Set of object types involved in interactions
    """
    interacting_object_types = set()

    for traj_data in traj_data_list:
        low_actions = traj_data.get('plan', {}).get('low_actions', [])

        for action_data in low_actions:
            api_action = action_data.get('api_action', {})
            action_name = api_action.get('action', '')

            # Identify interaction actions
            if action_name in [
                    'PickupObject', 'PutObject', 'OpenObject', 'CloseObject',
                    'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject'
            ]:
                object_id = api_action.get('objectId', '')
                if object_id and '|' in object_id:
                    # Extract objectType from "ObjectType|+x|+y|+z" format
                    object_type = object_id.split('|')[0]
                    interacting_object_types.add(object_type)
                receptacle_id = api_action.get('receptacleObjectId', '')
                if receptacle_id and '|' in receptacle_id:
                    receptacle_type = receptacle_id.split('|')[0]
                    interacting_object_types.add(receptacle_type)

    return interacting_object_types


def load_trajectory_for_scene(scene_num):
    """
    Load a representative trajectory for the given scene to get scene initialization data.

    Args:
        scene_num: Scene number (e.g., 1, 2, 301)

    Returns:
        list: [Trajectory data]
    """
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data/json_2.1.0/')

    # Search for any trajectory for this scene
    traj_data_list = []
    for task_dir in sorted(glob(os.path.join(data_path, '*'))):
        if not os.path.isdir(task_dir):
            continue

        for trial_dir in sorted(glob(os.path.join(task_dir, 'trial_*'))):
            json_file = os.path.join(trial_dir, 'pp', 'ann_0.json')
            if not os.path.exists(json_file):
                continue

            try:
                with open(json_file, 'r') as f:
                    traj_data = json.load(f)

                # Check if this trajectory is for the target scene
                if traj_data['scene']['scene_num'] == scene_num:
                    traj_data_list.append(traj_data)
            except Exception as e:
                logger.warning(f"Failed to read {json_file}: {e}")
                continue

    return traj_data_list


def process_scene(scene_name='FloorPlan1'):
    """
    Main function: Execute complete five-phase path point generation process.

    Args:
        scene_name: Scene name
    """
    logger.info(f"{'='*60}")
    logger.info(f"  Processing scene: {scene_name}")
    logger.info(f"{'='*60}\n")

    # Initialize environment
    env_connector = ThorConnector(x_display='1')

    # Load trajectory data for scene initialization info
    scene_num = int(scene_name.replace('FloorPlan', ''))
    traj_data = load_trajectory_for_scene(scene_num)

    # Reset scene
    env_connector.reset(scene_name)

    # If trajectory data found, fully initialize using its scene config
    if traj_data:
        logger.info("Fully initializing scene with trajectory data")

        # Collect all object positions
        object_poses = []
        for td in traj_data:
            object_poses.extend(td['scene']['object_poses'])

        logger.info(f"Collected {len(object_poses)} objects from trajectories")

        # Extract object types involved in interaction actions
        interacting_object_types = extract_interacting_objects_from_trajectories(
            traj_data)
        logger.info(f"Number of object types involved in interaction actions: {len(interacting_object_types)}")
        if interacting_object_types:
            logger.info(f"  Examples: {list(interacting_object_types)[:5]}")

        # Filter with priority for interaction objects
        object_poses_filtered = filter_overlapping_objects_by_distance(
            object_poses,
            distance_threshold=OVERLAP_DISTANCE_THRESHOLD,
            priority_types=interacting_object_types)

        logger.info(
            f"Remaining {len(object_poses_filtered)} objects after spatial distance threshold filtering "
            f"(removed {len(object_poses) - len(object_poses_filtered)} objects)")

        # Initialize the scene with the filtered object list
        env_connector.restore_scene(object_poses_filtered, [],
                                    traj_data[0]['scene']['dirty_and_empty'])
        env_connector.step(dict(traj_data[0]['scene']['init_action']))
        logger.info("Scene initialization complete")
    else:
        logger.warning("No trajectory data found, using default scene config")

    env_connector.reachable_positions, env_connector.reachable_position_kdtree = env_connector.get_reachable_positions(
    )

    # Build object tree
    objects_metadata = env_connector.last_event.metadata['objects']
    object_tree = build_object_tree_with_uniqueness(objects_metadata)

    logger.info(f"Scene stats:")
    logger.info(f"  Total objects: {object_tree['total_objects']}")
    logger.info(f"  Top-level objects: {object_tree['root_objects_count']}")
    logger.info(
        f"  Unique non-movable top-level objects: {len(object_tree['unique_non_movable_top_level'])}")
    logger.info(
        f"  Non-unique non-movable top-level objects: {len(object_tree['non_unique_non_movable_top_level'])}"
    )

    # Get objects in closed containers
    objects_in_closed = get_objects_in_closed_containers(objects_metadata)
    # objects_in_closed = set()

    # Initialize visibility tracking
    all_objects = set(obj['objectId'] for obj in objects_metadata)
    visibility_tracker = {
        'all_objects': all_objects,
        'visible_objects': set(),
        'objects_in_closed': objects_in_closed,
        'coverage_percentage': 0
    }

    all_path_points = []

    # === Before phase 0: add init positions of all trajectories ===
    if traj_data:
        logger.info("=== Adding trajectory init-position path points ===")
        init_points = generate_init_action_path_points(env_connector,
                                                       traj_data)
        all_path_points.extend(init_points)

        # Update visibility tracking
        for point in init_points:
            visibility_tracker['visible_objects'].update(
                point['visible_objects'])

        coverage = check_visibility_coverage(
            all_objects, visibility_tracker['visible_objects'],
            objects_in_closed)
        visibility_tracker['coverage_percentage'] = coverage[
            'coverage_percentage']
        logger.info(f"Coverage after init positions: {coverage['coverage_percentage']:.1f}%")

    # === Phase 0: replay navigation actions from trajectories in real time ===
    scene_num = int(scene_name.replace('FloorPlan', ''))
    if traj_data:
        phase0_points = generate_phase0_path_points_from_trajectories(
            env_connector, traj_data, scene_num, visibility_tracker)
        all_path_points.extend(phase0_points)
    else:
        logger.warning(f"No trajectory data for scene {scene_name}, skipping phase 0")

    # Phase 1: use unique non-movable objects
    phase1_points = generate_phase1_path_points(
        env_connector, object_tree['unique_non_movable_top_level'])
    all_path_points.extend(phase1_points)

    # Update visibility tracking
    for point in phase1_points:
        visibility_tracker['visible_objects'].update(point['visible_objects'])

    coverage = check_visibility_coverage(all_objects,
                                         visibility_tracker['visible_objects'],
                                         objects_in_closed)
    visibility_tracker['coverage_percentage'] = coverage['coverage_percentage']

    logger.info(f"Coverage after phase 1: {coverage['coverage_percentage']:.1f}%")

    # Phase 2: rotation exploration (navigate to unique non-movable objects and record initial rotation)
    phase2_points = generate_phase2_path_points(
        env_connector, object_tree['unique_non_movable_top_level'],
        visibility_tracker)
    all_path_points.extend(phase2_points)

    # Phase 3: use non-unique objects if needed
    phase3_points = []
    if visibility_tracker['coverage_percentage'] < 100:
        phase3_new_points, phase3_points = generate_phase3_path_points(
            env_connector, object_tree['non_unique_non_movable_top_level'],
            visibility_tracker)
        all_path_points.extend(phase3_new_points)

    # Phase 4: rotation exploration on non-unique reference objects
    if phase3_points and visibility_tracker['coverage_percentage'] < 100:
        phase4_points = generate_phase4_path_points(env_connector,
                                                    phase3_points,
                                                    visibility_tracker)
        all_path_points.extend(phase4_points)

    # === Global dedup and visibility recheck before phase 5 ===
    logger.info(f"=== Preparing phase 5: global dedup and visibility recheck ===")

    # Global dedup: remove cross-phase redundant path points
    logger.info(f"Total path points before dedup: {len(all_path_points)}")
    all_path_points = global_deduplication(all_path_points)
    logger.info(f"Path point count after dedup: {len(all_path_points)}")

    # Reset visibility_tracker
    visibility_tracker['visible_objects'] = set()
    visibility_tracker['coverage_percentage'] = 0

    # Recheck visible objects of all path points
    logger.info("Rechecking object visibility for all path points...")
    for i, point in enumerate(all_path_points):
        # Teleport to this path point
        env_connector.step({
            'action': 'TeleportFull',
            'x': point['position']['x'],
            'y': env_connector.agent_height,
            'z': point['position']['z'],
            'rotation': point['rotation'],
            'horizon': point['horizon'],
            'rotateOnTeleport': True
        })

        if not env_connector.last_event.metadata['lastActionSuccess']:
            logger.warning(f"  Path point {i+1} teleport failed, skipping")
            continue

        # Get visible objects and pixel counts
        visible_objs_list, total_pixels = get_visible_objects_with_pixel_counts(
            env_connector)
        visible_objs = set(visible_objs_list)
        visibility_tracker['visible_objects'].update(visible_objs)

        # Update the path point's visible_objects (with latest data)
        point['visible_objects'] = list(visible_objs)
        point['total_visible_pixels'] = total_pixels

    # Recalculate coverage
    coverage = check_visibility_coverage(all_objects,
                                         visibility_tracker['visible_objects'],
                                         objects_in_closed)
    visibility_tracker['coverage_percentage'] = coverage['coverage_percentage']

    logger.info(f"Coverage after recheck: {coverage['coverage_percentage']:.1f}%")

    # === Phase 5: gap-filling stage ===
    # For remaining unseen objects, use them as reference objects to generate path points
    if visibility_tracker['coverage_percentage'] < 100:
        phase5_points = generate_phase5_path_points(env_connector,
                                                    objects_metadata,
                                                    visibility_tracker)
        all_path_points.extend(phase5_points)

    # Global dedup: remove cross-phase redundant path points
    logger.info(f"Total path points before dedup: {len(all_path_points)}")
    all_path_points = global_deduplication(all_path_points)
    logger.info(f"Path point count after dedup: {len(all_path_points)}")

    # loc_desc dedup: ensure all path point descriptions are unique
    all_path_points = deduplicate_location_descriptions(all_path_points)

    # Recompute visibility (ensure coverage is still 100% after dedup)
    visibility_after_dedup = set()
    for point in all_path_points:
        visibility_after_dedup.update(point['visible_objects'])

    # Final coverage
    all_path_points = all_path_points
    final_coverage = check_visibility_coverage(all_objects,
                                               visibility_after_dedup,
                                               objects_in_closed)

    logger.info(f"Final coverage: {final_coverage['coverage_percentage']:.1f}%")
    logger.info(f"Final total path points: {len(all_path_points)}")

    # Verify path points and save images
    verification_results = verify_and_save_path_point_images(
        env_connector, scene_name, all_path_points)

    # Compute actual coverage after verification
    verified_visible_objects = set()
    for point_id, result in verification_results.items():
        if result['success']:
            verified_visible_objects.update(result['visible_objects'])

    verified_coverage = check_visibility_coverage(all_objects,
                                                  verified_visible_objects,
                                                  objects_in_closed)

    logger.info(f"Coverage after verification: {verified_coverage['coverage_percentage']:.1f}%")

    # Save path points
    save_path_points_with_descriptions(scene_name, all_path_points,
                                       final_coverage, OUTPUT_DIR)

    # Generate Markdown report (with verification results)
    generate_markdown_report(scene_name, all_path_points, object_tree,
                             final_coverage, REPORT_DIR, verification_results,
                             verified_coverage)

    # Cleanup
    env_connector.stop()

    logger.info(f"Scene {scene_name} processing complete")


if __name__ == "__main__":
    # Get scene list
    scene_names = [
        os.path.basename(x).split('-')[0]
        for x in glob('pgibench/envs/pgi_alfred/gen/layouts/*.npy')
    ]
    # Get list of processed scenes
    done_scene_names = [
        os.path.basename(x).split('-')[0] for x in glob('running/*.md')
    ]

    # Filter scenes to process
    todo_scene_names = [s for s in scene_names if s not in done_scene_names]
    todo_scene_names = todo_scene_names[::-1]

    for scene_name in todo_scene_names:
        try:
            process_scene(scene_name)
        except Exception as e:
            logger.error(f"Error processing scene {scene_name}: {e}", exc_info=True)
