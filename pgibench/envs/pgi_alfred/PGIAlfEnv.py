"""
AI2THOR Environment for Household Robot Task Simulation

This module provides a custom OpenAI Gym environment for simulating household robot tasks
using the AI2THOR framework. It supports various object interactions and task scenarios.

Dependencies:
- ai2thor
- gym
- numpy
- PIL
"""

import os
import re
import time
import gym
import json
import yaml
import random
import numpy as np
from PIL import Image

# Import custom modules
from pgibench.envs.pgi_alfred.eqa.thor_tracker import ThorTracker
import pgibench.envs.pgi_alfred.utils as utils
from pgibench.envs.pgi_alfred.utils import alfred_objs, alfred_open_obj, alfred_pick_obj, alfred_slice_obj, alfred_open_obj, alfred_toggle_obj, alfred_recep
from pgibench.envs.pgi_alfred.gen.constants import VAL_RECEPTACLE_OBJECTS, OBJ_PARENTS
from pgibench.envs.pgi_alfred.gen import constants
from pgibench.envs.pgi_alfred.thor_connector import ThorConnector
from pgibench.envs.pgi_alfred.action_command_parser import (parse_command,
                                                            resolve_command,
                                                            ResolverContext)
from pgibench.envs.pgi_alfred.eqa.eqa_generator import EQAGenerator
from pgibench.main import logger

# Action type constants
ACTION_GLOBAL = 'global'
ACTION_PICKUP = 'pickup'
ACTION_DROP = 'drop'
ACTION_PUTDOWN = 'putdown'
ACTION_OPEN = 'open'
ACTION_CLOSE = 'close'
ACTION_TOGGLE_ON = 'toggle_on'
ACTION_TOGGLE_OFF = 'toggle_off'
ACTION_SLICE = 'slice'
ACTION_GOTO = 'goto'
ACTION_NAV_OBJ = 'nav_obj'
ACTION_GOTO_COORD = 'goto_coord'

# global information
X_DISPLAY = '1'  # default X display for AI2-THOR; overridable via PGIAlfEnv(x_display=...)
ALFRED_REWARD_PATH = os.path.join(os.path.dirname(__file__),
                                  'config/rewards.json')

# Display-layer global action name -> execution-layer (PascalCase) command accepted
# by ThorConnector.llm_skill_interact(). The display layer exposes every action in
# a uniform lower-case + space form; the boundary layer (_build_interaction_string,
# parse_command) translates it back to the execution layer's canonical name.
_DISPLAY2EXEC_GLOBAL = {
    'look up': 'LookUp',
    'look down': 'LookDown',
    'rotate left': 'RotateLeft',
    'rotate right': 'RotateRight',
    'move ahead': 'MoveAhead',
    'move back': 'MoveBack',
    'move left': 'MoveLeft',
    'move right': 'MoveRight',
    'save waypoint': 'SaveWaypoint',
}


def get_global_action_space():
    """
    Generate a comprehensive action space for the environment.

    Returns:
        list: A list of supported action strings for navigation and view control.
              All names use the lower-case + space convention shared with the
              object-interaction actions (e.g. 'pick up', 'turn on').
    """
    action_space = []

    # Navigation and view control actions (lower-case + space)
    action_space.extend([
        "look up", "look down", "rotate left", "rotate right", "move ahead",
        "move back", "move left", "move right", "save waypoint"
    ])

    return action_space


def _rotation_to_direction(rotation_y):
    """Map a heading angle (0=North, clockwise) to an 8-point cardinal string.

    Same convention as ThorConnector.get_agent_position() / ThorEnv
    .get_agent_facing_direction(): rotation 0 faces +Z (North) and increases
    clockwise. Used here to expose a waypoint's facing direction alongside
    its numeric rotation.
    """
    angle = rotation_y % 360
    if 337.5 <= angle or angle < 22.5:
        return 'north'
    elif 22.5 <= angle < 67.5:
        return 'northeast'
    elif 67.5 <= angle < 112.5:
        return 'east'
    elif 112.5 <= angle < 157.5:
        return 'southeast'
    elif 157.5 <= angle < 202.5:
        return 'south'
    elif 202.5 <= angle < 247.5:
        return 'southwest'
    elif 247.5 <= angle < 292.5:
        return 'west'
    else:  # 292.5 <= angle < 337.5
        return 'northwest'


class PGIAlfEnv(gym.Env):
    """
    Custom OpenAI Gym environment for simulating household robot tasks.
    
    Attributes:
        env (ThorConnector): Interface for AI2THOR interactions
        action_space (gym.spaces.Discrete): Discrete action space 
        language_skill_set (list): Readable action descriptions
    """

    def __init__(self,
                 eval_set='base',
                 exp_name='',
                 down_sample_ratio=1.0,
                 selected_indexes=[],
                 detection_box=False,
                 resolution=500,
                 action_display_mode='visible',
                 visual_prompt=False,
                 enable_eqa=False,
                 nav_mode=None,
                 object_reference_mode='id',
                 bbox_iou_threshold=0.3,
                 point_dist_ratio=0.3,
                 instruction_mode='micro',
                 x_display=X_DISPLAY):
        """
        Initialize the AI2THOR environment.

        Args:
            eval_set: Which evaluation set to use
            exp_name: Experiment name for logging
            down_sample_ratio: Ratio to downsample the dataset
            selected_indexes: Specific episode indexes to use
            detection_box: Whether to draw detection boxes on images
            resolution: Screen resolution
            action_display_mode: 'full' (show all actions) or 'visible' (show only relevant actions)
            enable_eqa: If True, enable EQA question generation
            nav_mode: list of navigation modes. Valid items:
                'default_waypoint' - goto pre-loaded path points (alias: 'path_point')
                'created_waypoint' - enable SaveWaypoint action and custom waypoint goto
                'object' - navigate to visible objects
                'coordinate' - GoTo (x,y,z) direct coordinate navigation
                Empty list = no navigation. String auto-converted to single-element list.
            x_display: X display number for AI2-THOR (e.g. '1' -> DISPLAY=:1).
                Useful when the default display is occupied by another user.
        """
        super().__init__()
        assert instruction_mode in ('macro', 'micro'), \
            f"instruction_mode must be 'macro' or 'micro', got {instruction_mode}"
        self.instruction_mode = instruction_mode
        self.reward_config_path = ALFRED_REWARD_PATH
        self.resolution = resolution
        self.visual_prompt = visual_prompt
        # ai2thor's Controller.start() builds DISPLAY as ':' + x_display, so the
        # value here must be the bare display number. Normalize a leading ':'
        # away so callers may pass either '2' or ':2'.
        x_display = str(x_display).lstrip(':')
        self.env = ThorConnector(x_display=x_display,
                                 player_screen_height=resolution,
                                 player_screen_width=resolution,
                                 visual_prompt=visual_prompt)

        # load dataset
        self.down_sample_ratio = down_sample_ratio
        self.dataset = self._load_dataset(eval_set)
        if len(selected_indexes):
            self.dataset = [self.dataset[i] for i in selected_indexes]

        # Episode tracking
        self.number_of_episodes = len(self.dataset)
        self._reset = False
        self._current_episode_num = 0
        self.selected_indexes = selected_indexes
        self._initial_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 250
        self._cur_invalid_actions = 0
        self._max_invalid_actions = 15
        self._episode_start_time = 0
        self.episode_log = []

        # Task-related attributes
        self.episode_language_instruction = ''
        self.episode_data = None

        # env feedback and image save
        # feedback verbosity, 0: concise, 1: verbose
        self.feedback_verbosity = 0
        self.log_path = 'running/pgi_alfred/{}'.format(exp_name)

        self.detection = detection_box or visual_prompt  # add detection in image

        # Action display mode: 'full' (enumerate all), 'visible' (enumerate visible/nearby),
        # or 'command' (free-text command output; disambiguated via object_reference_mode)
        assert action_display_mode in ['full', 'visible', 'command'], \
            f"action_display_mode must be 'full', 'visible' or 'command', got {action_display_mode}"
        self.action_display_mode = action_display_mode
        if action_display_mode == 'command':
            assert object_reference_mode in [
                'id', 'bbox', 'point', 'ordinal', 'type'], \
                f"object_reference_mode must be one of id/bbox/point/ordinal/type, got {object_reference_mode}"
            self.object_reference_mode = object_reference_mode
        else:
            self.object_reference_mode = None
        # bbox/point matching thresholds for command-mode reference resolution.
        self.bbox_iou_threshold = float(bbox_iou_threshold)
        self.point_dist_ratio = float(point_dist_ratio)
        # Normalize nav_mode to list with backward-compatible aliases
        _VALID_MODES = ('default_waypoint', 'created_waypoint', 'object',
                        'coordinate')
        _ALIAS = {'path_point': 'default_waypoint'}
        if nav_mode is None:
            nav_mode = ['default_waypoint']
        elif isinstance(nav_mode, str):
            nav_mode = [nav_mode]
        elif not isinstance(nav_mode, (list, tuple)):
            raise ValueError(
                f"nav_mode must be str/list, got {type(nav_mode)}")
        self.nav_mode = [_ALIAS.get(m, m) for m in nav_mode]
        for m in self.nav_mode:
            assert m in _VALID_MODES, \
                f"Invalid nav_mode '{m}', valid: {_VALID_MODES}"
        # Predefined path points are only relevant for default_waypoint. When
        # only created_waypoint is active, the unified list starts empty and is
        # grown exclusively via SaveWaypoint.
        self.env.load_predefined_path_points = \
            'default_waypoint' in self.nav_mode

        # Static action dictionary (built incrementally)
        self.static_action_space_built = False
        self.static_language_skill_dict = {
        }  # Key: static action with objectId
        self.static_action_to_idx = {}  # Map static_key to canonical index
        self._known_objects = set()  # Track objects we've already processed

        # Scene tracking for reset detection
        self._current_scene_name = None  # Track current FloorPlan for reset detection

        # Reverse mapping for action execution (display_skill → static_key)
        self.display_to_static_key = {}

        # Dynamic display state
        self.object_id_to_current_locdesc = {}  # obj_id → current locDesc
        self.language_skill_set = []  # Dynamic display skills with loc_desc

        # Initialize with global actions
        self.language_skill_set = get_global_action_space()
        self.action_space = gym.spaces.Discrete(len(
            self.language_skill_set)) if len(
                self.language_skill_set) > 0 else None

        # thor tracker for QA generation
        self.thor_tracker = ThorTracker()
        self.enable_thor_tracking = enable_eqa  # Enable tracking if EQA is enabled

        # EQA generator
        self.eqa_generator = None
        self.enable_eqa = enable_eqa
        if self.enable_eqa:
            self.eqa_generator = EQAGenerator(self.thor_tracker, constants)

    def _build_static_action_dict(self):
        """
        Build static action dictionary with objectId-based keys.
        This is called once on the first reset() to create the complete action space.

        The dictionary keys use immutable objectIds, while values contain metadata
        including display templates with {loc_desc} placeholders.
        """

        static_dict = {}
        idx = 0

        # 1. Add global navigation actions (indices 0-8)
        global_actions = get_global_action_space()
        for action in global_actions:
            static_key = action
            static_dict[static_key] = {
                'canonical_idx': idx,
                'action_type': ACTION_GLOBAL,
                'display_template': action
            }
            self.static_action_to_idx[static_key] = idx
            idx += 1

        # 2. Get all objects in the scene
        all_objects = self.env.last_event.metadata['objects']

        # Categorize objects by interaction type
        pickupable_objects = []  # (obj_id, obj_type)
        openable_objects = []  # (obj_id, obj_type)
        toggleable_objects = []  # (obj_id, obj_type)
        sliceable_objects = []  # (obj_id, obj_type)
        all_receptacles = []  # (obj_id, obj_type)

        for obj in all_objects:
            obj_id = obj['objectId']
            obj_type = obj['objectType']

            # Check if object is pickupable (including sliced versions)
            if obj['pickupable']:
                is_pickupable = (obj_type in alfred_pick_obj
                                 or obj_type.endswith('Sliced'))
                if is_pickupable:
                    pickupable_objects.append((obj_id, obj_type))

            if obj.get('openable', False) and obj_type in alfred_open_obj:
                openable_objects.append((obj_id, obj_type))

            if obj.get('toggleable', False) and obj_type in alfred_toggle_obj:
                toggleable_objects.append((obj_id, obj_type))

            if obj.get('sliceable', False) and obj_type in alfred_slice_obj:
                sliceable_objects.append((obj_id, obj_type))

            # Track all receptacles for putdown actions
            if obj_type in VAL_RECEPTACLE_OBJECTS:
                all_receptacles.append((obj_id, obj_type))

        # 3. Generate pickup actions
        for obj_id, obj_type in pickupable_objects:
            static_key = f"pick up {obj_id}"
            static_dict[static_key] = {
                'canonical_idx': idx,
                'action_type': ACTION_PICKUP,
                'obj_id': obj_id,
                'obj_type': obj_type,
                'display_template': 'pick up {loc_desc}'
            }
            self.static_action_to_idx[static_key] = idx
            idx += 1

        # 4. Add generic drop action
        static_key = "drop the object in hand"
        static_dict[static_key] = {
            'canonical_idx': idx,
            'action_type': ACTION_DROP,
            'display_template': 'drop the object in hand'
        }
        self.static_action_to_idx[static_key] = idx
        idx += 1

        # 5. Generate putdown actions for all valid combinations
        # Format: "put down {held_obj_id} to {recep_id}"
        # Preposition is always 'to' (display form); the execution layer keys only
        # on the receptacle objectId.
        for held_obj_id, held_obj_type in pickupable_objects:
            # Get parent type (e.g., AppleSliced -> Apple)
            parent_obj_type = OBJ_PARENTS.get(held_obj_type, held_obj_type)

            # Find valid receptacles for this object type
            for recep_id, recep_type in all_receptacles:
                # Check if this receptacle can hold this object
                if recep_type in VAL_RECEPTACLE_OBJECTS:
                    valid_objects = VAL_RECEPTACLE_OBJECTS[recep_type]
                    if parent_obj_type in valid_objects:
                        static_key = f"put down {held_obj_id} to {recep_id}"
                        static_dict[static_key] = {
                            'canonical_idx':
                            idx,
                            'action_type':
                            ACTION_PUTDOWN,
                            'held_obj_id':
                            held_obj_id,
                            'held_obj_type':
                            held_obj_type,
                            'recep_id':
                            recep_id,
                            'recep_type':
                            recep_type,
                            'display_template':
                            'put down {held_obj_type} to {recep_loc_desc}'
                        }
                        self.static_action_to_idx[static_key] = idx
                        idx += 1

        # 6. Generate open/close actions
        for obj_id, obj_type in openable_objects:
            # Open action
            static_key = f"open {obj_id}"
            static_dict[static_key] = {
                'canonical_idx': idx,
                'action_type': ACTION_OPEN,
                'obj_id': obj_id,
                'obj_type': obj_type,
                'display_template': 'open {loc_desc}'
            }
            self.static_action_to_idx[static_key] = idx
            idx += 1

            # Close action
            static_key = f"close {obj_id}"
            static_dict[static_key] = {
                'canonical_idx': idx,
                'action_type': ACTION_CLOSE,
                'obj_id': obj_id,
                'obj_type': obj_type,
                'display_template': 'close {loc_desc}'
            }
            self.static_action_to_idx[static_key] = idx
            idx += 1

        # 7. Generate toggle on/off actions
        for obj_id, obj_type in toggleable_objects:
            # Toggle on
            static_key = f"turn on {obj_id}"
            static_dict[static_key] = {
                'canonical_idx': idx,
                'action_type': ACTION_TOGGLE_ON,
                'obj_id': obj_id,
                'obj_type': obj_type,
                'display_template': 'turn on {loc_desc}'
            }
            self.static_action_to_idx[static_key] = idx
            idx += 1

            # Toggle off
            static_key = f"turn off {obj_id}"
            static_dict[static_key] = {
                'canonical_idx': idx,
                'action_type': ACTION_TOGGLE_OFF,
                'obj_id': obj_id,
                'obj_type': obj_type,
                'display_template': 'turn off {loc_desc}'
            }
            self.static_action_to_idx[static_key] = idx
            idx += 1

        # 8. Generate slice actions
        for obj_id, obj_type in sliceable_objects:
            static_key = f"slice {obj_id}"
            static_dict[static_key] = {
                'canonical_idx': idx,
                'action_type': ACTION_SLICE,
                'obj_id': obj_id,
                'obj_type': obj_type,
                'display_template': 'slice {loc_desc}'
            }
            self.static_action_to_idx[static_key] = idx
            idx += 1

        # 9. Generate navigation actions based on nav_mode (multi-select)
        if 'object' in self.nav_mode:
            # Navigate to objects: generate for all interactive objects
            nav_obj_candidates = set()
            for obj_id, obj_type in (pickupable_objects + openable_objects +
                                     toggleable_objects + sliceable_objects):
                nav_obj_candidates.add((obj_id, obj_type))
            # Also include receptacles and other alfred_objs (e.g., Fridge, CounterTop)
            for obj_id, obj_type in all_receptacles:
                nav_obj_candidates.add((obj_id, obj_type))
            for obj_id, obj_type in sorted(nav_obj_candidates):
                static_key = f"navigate to {obj_id}"
                if static_key not in static_dict:
                    static_dict[static_key] = {
                        'canonical_idx': idx,
                        'action_type': ACTION_NAV_OBJ,
                        'obj_id': obj_id,
                        'obj_type': obj_type,
                        'display_template': 'navigate to {loc_desc}'
                    }
                    self.static_action_to_idx[static_key] = idx
                    idx += 1
        # NOTE: 'goto waypoint_#NN' actions are NOT registered here. Path points
        # (predefined + SaveWaypoint-created) are a single evolving list, so goto
        # actions are generated dynamically each step in _update_language_skill_set().
        if 'coordinate' in self.nav_mode:
            # GoTo (x,y,z) coordinate navigation: template action, VLM replaces with actual coords
            static_key = "GoTo (x,y,z)"
            static_dict[static_key] = {
                'canonical_idx': idx,
                'action_type': ACTION_GOTO_COORD,
                'display_template': 'GoTo (x,y,z)'
            }
            self.static_action_to_idx[static_key] = idx
            idx += 1

        # Store the static dictionary
        self.static_language_skill_dict = static_dict

        # Update gym action space with the total size
        self.action_space = gym.spaces.Discrete(idx)

        # Mark as built
        self.static_action_space_built = True

        logger.info(f"Built static action dictionary: {idx} total actions")

        # Initialize known objects set
        for obj in all_objects:
            self._known_objects.add(obj['objectId'])

    def _clear_static_action_structures(self):
        """
        Clear all static action space structures in preparation for rebuild.

        This is called when switching to a new scene with different path_points
        and object configurations.
        """
        # Clear static dictionaries
        self.static_language_skill_dict.clear()
        self.static_action_to_idx.clear()

        # Clear tracking sets
        self._known_objects.clear()

        # Clear dynamic mappings
        self.display_to_static_key.clear()
        self.object_id_to_current_locdesc.clear()

        # Reset flag
        self.static_action_space_built = False

        logger.info("Cleared static action space structures for scene switch")

    def _add_new_objects_to_static_dict(self, new_object_ids):
        """
        Incrementally add actions for new objects that appeared in the scene.

        This preserves existing static dictionary entries and only adds new ones.

        Args:
            new_object_ids: Set of new object IDs to add
        """
        from pgibench.envs.pgi_alfred.gen.constants import VAL_RECEPTACLE_OBJECTS, OBJ_PARENTS

        if not new_object_ids:
            return

        logger.info(
            f"Adding {len(new_object_ids)} new objects to static action dictionary"
        )

        # Get current max index
        current_max_idx = max(self.static_action_to_idx.values()
                              ) if self.static_action_to_idx else -1
        idx = current_max_idx + 1

        # Get all objects from scene
        all_objects = self.env.last_event.metadata['objects']
        new_objects = [
            obj for obj in all_objects if obj['objectId'] in new_object_ids
        ]

        # Get all receptacles (for putdown actions)
        all_receptacles = []
        for obj in all_objects:
            obj_id = obj['objectId']
            obj_type = obj['objectType']
            if obj_type in VAL_RECEPTACLE_OBJECTS:
                all_receptacles.append((obj_id, obj_type))

        # Categorize new objects by interaction type
        pickupable_objects = []
        openable_objects = []
        toggleable_objects = []
        sliceable_objects = []

        for obj in new_objects:
            obj_id = obj['objectId']
            obj_type = obj['objectType']

            # Check if object is pickupable (including sliced versions)
            if obj['pickupable']:
                # Check if type is directly in alfred_pick_obj OR is a sliced version
                is_pickupable = (obj_type in alfred_pick_obj
                                 or obj_type.endswith('Sliced'))
                if is_pickupable:
                    pickupable_objects.append((obj_id, obj_type))

            if obj.get('openable', False) and obj_type in alfred_open_obj:
                openable_objects.append((obj_id, obj_type))

            if obj.get('toggleable', False) and obj_type in alfred_toggle_obj:
                toggleable_objects.append((obj_id, obj_type))

            if obj.get('sliceable', False) and obj_type in alfred_slice_obj:
                sliceable_objects.append((obj_id, obj_type))

        # Generate pickup actions for new objects
        for obj_id, obj_type in pickupable_objects:
            static_key = f"pick up {obj_id}"
            if static_key not in self.static_language_skill_dict:
                self.static_language_skill_dict[static_key] = {
                    'canonical_idx': idx,
                    'action_type': ACTION_PICKUP,
                    'obj_id': obj_id,
                    'obj_type': obj_type,
                    'display_template': 'pick up {loc_desc}'
                }
                self.static_action_to_idx[static_key] = idx
                idx += 1

        # Generate putdown actions: new objects -> all existing receptacles
        for held_obj_id, held_obj_type in pickupable_objects:
            parent_obj_type = OBJ_PARENTS.get(held_obj_type, held_obj_type)

            for recep_id, recep_type in all_receptacles:
                if recep_type in VAL_RECEPTACLE_OBJECTS:
                    valid_objects = VAL_RECEPTACLE_OBJECTS[recep_type]
                    if parent_obj_type in valid_objects:
                        static_key = f"put down {held_obj_id} to {recep_id}"

                        if static_key not in self.static_language_skill_dict:
                            self.static_language_skill_dict[static_key] = {
                                'canonical_idx':
                                idx,
                                'action_type':
                                ACTION_PUTDOWN,
                                'held_obj_id':
                                held_obj_id,
                                'held_obj_type':
                                held_obj_type,
                                'recep_id':
                                recep_id,
                                'recep_type':
                                recep_type,
                                'display_template':
                                'put down {held_obj_type} to {recep_loc_desc}'
                            }
                            self.static_action_to_idx[static_key] = idx
                            idx += 1

        # Generate putdown actions: all existing objects -> new receptacles (if any)
        existing_pickupable = []
        for obj in all_objects:
            if obj['objectId'] not in new_object_ids:
                obj_id = obj['objectId']
                obj_type = obj['objectType']
                if obj['pickupable']:
                    is_pickupable = (obj_type in alfred_pick_obj
                                     or obj_type.endswith('Sliced'))
                    if is_pickupable:
                        existing_pickupable.append((obj_id, obj_type))

        new_receptacles = [(obj_id, obj_type)
                           for obj_id, obj_type in all_receptacles
                           if obj_id in new_object_ids]

        for held_obj_id, held_obj_type in existing_pickupable:
            parent_obj_type = OBJ_PARENTS.get(held_obj_type, held_obj_type)

            for recep_id, recep_type in new_receptacles:
                if recep_type in VAL_RECEPTACLE_OBJECTS:
                    valid_objects = VAL_RECEPTACLE_OBJECTS[recep_type]
                    if parent_obj_type in valid_objects:
                        static_key = f"put down {held_obj_id} to {recep_id}"

                        if static_key not in self.static_language_skill_dict:
                            self.static_language_skill_dict[static_key] = {
                                'canonical_idx':
                                idx,
                                'action_type':
                                ACTION_PUTDOWN,
                                'held_obj_id':
                                held_obj_id,
                                'held_obj_type':
                                held_obj_type,
                                'recep_id':
                                recep_id,
                                'recep_type':
                                recep_type,
                                'display_template':
                                'put down {held_obj_type} to {recep_loc_desc}'
                            }
                            self.static_action_to_idx[static_key] = idx
                            idx += 1

        # Generate open/close actions for new objects
        for obj_id, obj_type in openable_objects:
            # Open action
            static_key = f"open {obj_id}"
            if static_key not in self.static_language_skill_dict:
                self.static_language_skill_dict[static_key] = {
                    'canonical_idx': idx,
                    'action_type': ACTION_OPEN,
                    'obj_id': obj_id,
                    'obj_type': obj_type,
                    'display_template': 'open {loc_desc}'
                }
                self.static_action_to_idx[static_key] = idx
                idx += 1

            # Close action
            static_key = f"close {obj_id}"
            if static_key not in self.static_language_skill_dict:
                self.static_language_skill_dict[static_key] = {
                    'canonical_idx': idx,
                    'action_type': ACTION_CLOSE,
                    'obj_id': obj_id,
                    'obj_type': obj_type,
                    'display_template': 'close {loc_desc}'
                }
                self.static_action_to_idx[static_key] = idx
                idx += 1

        # Generate toggle on/off actions for new objects
        for obj_id, obj_type in toggleable_objects:
            # Toggle on
            static_key = f"turn on {obj_id}"
            if static_key not in self.static_language_skill_dict:
                self.static_language_skill_dict[static_key] = {
                    'canonical_idx': idx,
                    'action_type': ACTION_TOGGLE_ON,
                    'obj_id': obj_id,
                    'obj_type': obj_type,
                    'display_template': 'turn on {loc_desc}'
                }
                self.static_action_to_idx[static_key] = idx
                idx += 1

            # Toggle off
            static_key = f"turn off {obj_id}"
            if static_key not in self.static_language_skill_dict:
                self.static_language_skill_dict[static_key] = {
                    'canonical_idx': idx,
                    'action_type': ACTION_TOGGLE_OFF,
                    'obj_id': obj_id,
                    'obj_type': obj_type,
                    'display_template': 'turn off {loc_desc}'
                }
                self.static_action_to_idx[static_key] = idx
                idx += 1

        # Generate slice actions for new objects
        for obj_id, obj_type in sliceable_objects:
            static_key = f"slice {obj_id}"
            if static_key not in self.static_language_skill_dict:
                self.static_language_skill_dict[static_key] = {
                    'canonical_idx': idx,
                    'action_type': ACTION_SLICE,
                    'obj_id': obj_id,
                    'obj_type': obj_type,
                    'display_template': 'slice {loc_desc}'
                }
                self.static_action_to_idx[static_key] = idx
                idx += 1

        # Generate nav_obj actions for new objects (if nav_mode=='object')
        if 'object' in self.nav_mode:
            nav_candidates = (pickupable_objects + openable_objects +
                              toggleable_objects + sliceable_objects)
            new_recep_nav = [(obj_id, obj_type)
                             for obj_id, obj_type in new_receptacles]
            nav_candidates += new_recep_nav
            for obj_id, obj_type in nav_candidates:
                static_key = f"navigate to {obj_id}"
                if static_key not in self.static_language_skill_dict:
                    self.static_language_skill_dict[static_key] = {
                        'canonical_idx': idx,
                        'action_type': ACTION_NAV_OBJ,
                        'obj_id': obj_id,
                        'obj_type': obj_type,
                        'display_template': 'navigate to {loc_desc}'
                    }
                    self.static_action_to_idx[static_key] = idx
                    idx += 1

        # Update gym action space size
        self.action_space = gym.spaces.Discrete(idx)

        # Mark new objects as known
        self._known_objects.update(new_object_ids)

        logger.info(
            f"Added {len(new_object_ids)} new objects, total actions: {idx}")

    def _update_language_skill_set(self):
        """
        Update the dynamic display list (language_skill_set) with current locDesc values.

        This method:
        1. Checks for new objects and adds them to static dictionary
        2. Updates object_id_to_current_locdesc mapping
        3. Generates display actions based on action_display_mode
        4. Creates reverse mapping (display_to_static_key)
        """
        # 1. Update locDesc for all objects and check for new objects
        self.env.update_loc_descs()
        self.object_id_to_current_locdesc.clear()

        current_objects = set()
        for obj in self.env.last_event.metadata['objects']:
            obj_id = obj['objectId']
            obj_type = obj['objectType']
            loc_desc = obj.get('locDesc', obj['objectType'])
            self.object_id_to_current_locdesc[obj_id] = loc_desc
            current_objects.add(obj_id)

        # 2. Check for new objects and incrementally update static dictionary
        new_objects = current_objects - self._known_objects
        if new_objects:
            self._add_new_objects_to_static_dict(new_objects)

        # command mode: no enumerated action list; only refresh locDesc / known objects
        if self.action_display_mode == 'command':
            self.language_skill_set = get_global_action_space()
            self.display_to_static_key.clear()
            return

        # 3. Allowed objects (visible/nearby for 'visible'; all for 'full')
        allowed_objs = self._get_allowed_objs()

        # 4. Get current held object
        held_obj_id, held_obj_type = self.env.get_held_object()

        # 5. Generate display actions
        display_actions = []
        self.display_to_static_key.clear()

        for static_key, static_value in self.static_language_skill_dict.items(
        ):
            action_type = static_value['action_type']

            should_include = self._should_include_action_visible(
                action_type, static_value, allowed_objs, held_obj_id)

            if should_include:
                # Format the display action
                display_action = self._format_display_action(
                    static_key, static_value)
                display_actions.append(display_action)
                self.display_to_static_key[display_action] = static_key

        # 5b. Dynamically generate 'goto waypoint_#NN' actions from the unified
        # path point list (predefined points + SaveWaypoint-created points).
        # Path points are an evolving list, so they cannot live in the static
        # action dict. Resolved at step() via a dedicated goto string branch
        # (no display_to_static_key lookup needed).
        if ('default_waypoint' in self.nav_mode
                or 'created_waypoint' in self.nav_mode) \
                and self.env.path_points_keys is not None:
            for loc_desc in self.env.path_points_keys:
                display_action = f"goto {loc_desc}"
                display_actions.append(display_action)
                # Sentinel mapping: keeps step()'s reverse lookup consistent but
                # the goto string branch intercepts before any static lookup.
                self.display_to_static_key[display_action] = \
                    f"__goto__{loc_desc}"

        # 6. Update language_skill_set
        self.language_skill_set = display_actions

    def _get_allowed_objs(self):
        """Set of objectIds valid under the current display mode's range.

        Perception-grounding policy: an object is actionable only when it is
        actually visible in the agent's view. 'visible' and 'command' modes
        therefore restrict to visible objects only — nearby-but-unseen objects
        are not actionable (the agent must turn/look/move to bring them into
        view first); 'full' mode covers all scene objects.

        Note: get_nearby_objects() is retained for observation/tracker display,
        just no longer contributes to executability.
        """
        if self.action_display_mode in ('visible', 'command'):
            return set(self.env.get_visible_objects())
        return set(x['objectId']
                   for x in self.env.last_event.metadata['objects'])

    def _build_resolver_context(self):
        """Build a ResolverContext snapshot from live simulator state."""
        env = self.env
        meta = env.last_event.metadata
        dets = getattr(env.last_event, 'instance_detections2D', None) or {}
        import math as _math
        _diag = _math.hypot(self.resolution, self.resolution)
        return ResolverContext(
            allowed_objs=self._get_allowed_objs(),
            instance_detections2D={
                k: list(v)
                for k, v in dets.items()
            },
            id_2_name=dict(env.id_2_name),
            name_2_id={
                n: oid
                for oid, n in env.id_2_name.items()
            },
            image_size=(self.resolution, self.resolution),
            iou_threshold=self.bbox_iou_threshold,
            point_dist_threshold=self.point_dist_ratio * _diag,
            locdesc_map=dict(self.object_id_to_current_locdesc),
            object_types={
                o['objectId']: o['objectType']
                for o in meta['objects']
            },
            object_distances={
                o['objectId']: o.get('distance', 1e8)
                for o in meta['objects']
            })

    def get_object_catalog(self, mode=None):
        """Return a text object catalog for command mode (prompt / UI use).

        For ``object_reference_mode='id'`` (mode='id') returns one line per
        visible object with its ``<type>_#<n>`` ID and the bbox normalised to
        [0,1] relative to the image (origin top-left, bottom-right (1,1)):

            - mug_#0 [0.121, 0.083, 0.398, 0.361]
            - apple_#0 [0.500, 0.420, 0.620, 0.550]

        The visible set, bbox source and ID labels are exactly those used by
        ``save_image()`` (``get_visible_objects()`` + ``instance_detections2D``
        + ``id_2_name``), so the text stays consistent with the image overlay.
        Returns "" for every other reference mode.
        """
        if mode is None:
            mode = self.object_reference_mode
        if mode != 'id':
            return ""

        env = self.env
        id_2_name = getattr(env, 'id_2_name', {}) or {}
        dets = getattr(env.last_event, 'instance_detections2D', None) or {}
        visible_objs = set(env.get_visible_objects())
        res = float(self.resolution) if self.resolution else 1.0
        res = res if res > 0 else 1.0

        lines = []
        for obj_id, bbox in dets.items():
            # Keep the text consistent with save_image(): visible + labelled +
            # has a detection box. Skip anything missing any of the three so the
            # listed IDs/bboxes never drift from the image overlay.
            if obj_id not in visible_objs or obj_id not in id_2_name:
                continue
            # AI2-THOR stores instance_detections2D values as numpy arrays, so
            # coerce to a plain list before any truthiness/length check to avoid
            # "truth value of an array is ambiguous" errors (see _build_resolver
            # _context which does the same `list(v)` cast for the resolvers).
            try:
                bbox = list(bbox)
            except TypeError:
                continue
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (bbox[0], bbox[1], bbox[2], bbox[3])
            norm = [x1 / res, y1 / res, x2 / res, y2 / res]
            coord = ", ".join(f"{v:.3f}" for v in norm)
            lines.append(f"- {id_2_name[obj_id]} [{coord}]")
        return "\n".join(lines)

    def _invalid_action_return(self,
                               action,
                               reasoning,
                               message,
                               resolved_target=None):
        """Early-return tuple for an unparseable/invalid command (command mode) or an
        out-of-space action (enumerated mode). Mirrors step()'s post-processing so
        downstream planner / tracker / UI see a consistent info dict."""
        self._cur_invalid_actions += 1
        subtask_details = self.env.get_subtask_details()
        goal_conditions = self.env.get_goal_conditions_met()
        info = {
            'instruction':
            self.episode_language_instruction,
            'env_step':
            self._current_step,
            'env_feedback':
            f"Last action is invalid. {self._sanitize_feedback_message(message)}\n"
            f"Please reflect on the reason for this failure and try a"
            f" different action to correct it.",
            'episode_elapsed_seconds':
            time.time() - self._episode_start_time,
            'last_action_success':
            0.0,
            'action_id':
            action,
            'action_description':
            action if isinstance(action, str) else str(action),
            'reasoning':
            reasoning,
            'resolved_target':
            resolved_target,
            'object_reference_mode':
            self.object_reference_mode,
            'agent_position':
            self.env.get_agent_position(),
            'task_success':
            float(self.env.get_goal_satisfied()),
            'task_progress':
            (goal_conditions[0] / goal_conditions[1]
             if goal_conditions[1] else 0.0),
            'num_subtasks':
            len(subtask_details),
            'total_goal_conditions':
            goal_conditions[1],
            'subtask_completion':
            (sum(1 for d in subtask_details if d['completed']) /
             len(subtask_details)) if subtask_details else 0.0,
            'subtask_details':
            subtask_details,
            'object_states': {
                "cooled_objects": self.env.cooled_objects,
                "heated_objects": self.env.heated_objects,
                "cleaned_objects": self.env.cleaned_objects,
                "visible_objs": self.env.get_visible_objects(),
                "nearby_objs": self.env.get_nearby_objects(),
            },
        }
        self.episode_log.append(info)
        self._update_language_skill_set()
        if self.enable_thor_tracking:
            tracker_skill_set = self.language_skill_set
            if self.action_display_mode == 'command':
                tracker_skill_set = sorted(self._get_allowed_objs())
            self.thor_tracker.update_step(env=self.env,
                                          action_info=info,
                                          reasoning=reasoning,
                                          language_skill_set=tracker_skill_set)
        done = (self._current_step >= self._max_episode_steps
                or info['task_success']
                or self._cur_invalid_actions >= self._max_invalid_actions)
        obs = {'head_rgb': self.env.last_event.frame}
        return obs, 0.0, done, info

    def _should_include_action_visible(self, action_type, static_value,
                                       allowed_objs, held_obj_id):
        """
        Determine if an action should be included in visible mode.

        Args:
            action_type: Type of action (ACTION_PICKUP, etc.)
            static_value: Action metadata from static dictionary
            allowed_objs: Set of visible/nearby object IDs
            held_obj_id: Currently held object ID (or None)

        Returns:
            True if action should be included, False otherwise
        """
        # Always include global and drop actions
        if action_type == ACTION_GLOBAL:
            # save waypoint only visible in created_waypoint mode
            if static_value.get('display_template') == 'save waypoint':
                return 'created_waypoint' in self.nav_mode
            return True

        if action_type == ACTION_DROP:
            return held_obj_id is not None

        # Pickup actions: only if object is visible/nearby and not already held
        if action_type == ACTION_PICKUP:
            obj_id = static_value['obj_id']
            return obj_id in allowed_objs and held_obj_id is None

        # Putdown actions: only if holding that specific object and receptacle is visible/nearby
        if action_type == ACTION_PUTDOWN:
            held_obj_from_key = static_value['held_obj_id']
            recep_id = static_value['recep_id']
            # Only include if holding the specific object AND receptacle is nearby
            return held_obj_id == held_obj_from_key and recep_id in allowed_objs

        if action_type == ACTION_SLICE:
            return held_obj_id is not None and (
                'knives' in held_obj_id.lower()
                or 'knife' in held_obj_id.lower())

        # Open/Close/Toggle: only if object is visible/nearby
        if action_type in [
                ACTION_OPEN, ACTION_CLOSE, ACTION_TOGGLE_ON, ACTION_TOGGLE_OFF
        ]:
            obj_id = static_value['obj_id']
            return obj_id in allowed_objs

        # Goto actions: visible if default or created waypoint mode is active
        if action_type == ACTION_GOTO:
            return ('default_waypoint' in self.nav_mode
                    or 'created_waypoint' in self.nav_mode)

        # GoTo coordinate: visible only in coordinate mode
        if action_type == ACTION_GOTO_COORD:
            return 'coordinate' in self.nav_mode

        # Nav_obj actions: object must be visible or nearby
        if action_type == ACTION_NAV_OBJ:
            obj_id = static_value['obj_id']
            return obj_id in allowed_objs

        return False

    def _format_display_action(self, static_key, static_value):
        """
        Format a display action string by replacing {loc_desc} placeholders with current values.

        Args:
            static_key: The static key (e.g., "pick up Apple|-03.52|+00.00|+02.72")
            static_value: The action metadata dictionary

        Returns:
            Formatted display action string (e.g., "pick up Apple on Table")
        """
        action_type = static_value['action_type']
        template = static_value['display_template']

        # Global and drop actions: no placeholders
        if action_type in [ACTION_GLOBAL, ACTION_DROP]:
            return template

        # Pickup: replace {loc_desc} with current locDesc
        if action_type == ACTION_PICKUP:
            obj_id = static_value['obj_id']
            loc_desc = self.object_id_to_current_locdesc.get(obj_id, obj_id)
            return template.replace('{loc_desc}', loc_desc)

        # Putdown: replace both held_obj_type and recep_loc_desc
        if action_type == ACTION_PUTDOWN:
            held_obj_type = static_value['held_obj_type']
            recep_id = static_value['recep_id']
            recep_loc_desc = self.object_id_to_current_locdesc.get(
                recep_id, recep_id)
            return f"put down {held_obj_type.lower()} to {recep_loc_desc}"

        # Open/Close/Toggle/Slice: replace {loc_desc}
        if action_type in [
                ACTION_OPEN, ACTION_CLOSE, ACTION_TOGGLE_ON, ACTION_TOGGLE_OFF,
                ACTION_SLICE
        ]:
            obj_id = static_value['obj_id']
            loc_desc = self.object_id_to_current_locdesc.get(obj_id, obj_id)
            return template.replace('{loc_desc}', loc_desc)

        # Goto: already formatted
        if action_type == ACTION_GOTO:
            return template

        # GoTo coordinate: template with no placeholders (VLM replaces with actual coords)
        if action_type == ACTION_GOTO_COORD:
            return template

        # Nav_obj: replace {loc_desc} like pickup
        if action_type == ACTION_NAV_OBJ:
            obj_id = static_value['obj_id']
            loc_desc = self.object_id_to_current_locdesc.get(obj_id, obj_id)
            return template.replace('{loc_desc}', loc_desc)

        # Fallback
        return template

    def _build_interaction_string(self, static_key, static_value):
        """
        Build the interaction string for ThorConnector.llm_skill_interact().

        This converts a static action key and metadata into the actual command
        that will be sent to the AI2-THOR simulator.

        Args:
            static_key: The static action key (e.g., "pick up Apple|-03.52|+00.00|+02.72")
            static_value: The action metadata dictionary

        Returns:
            String command to pass to llm_skill_interact()
        """
        action_type = static_value['action_type']

        # Global actions: translate the lower-case display name back to the
        # execution layer's PascalCase command.
        if action_type == ACTION_GLOBAL:
            return _DISPLAY2EXEC_GLOBAL[static_key]

        # Pickup: "pick up {obj_id}"
        if action_type == ACTION_PICKUP:
            return f"pick up {static_value['obj_id']}"

        # Drop: return as-is
        if action_type == ACTION_DROP:
            return static_key

        # Putdown: "put down {recep_id}"
        if action_type == ACTION_PUTDOWN:
            return f"put down {static_value['recep_id']}"

        # Open/Close: "{action} {obj_id}"
        if action_type == ACTION_OPEN:
            return f"open {static_value['obj_id']}"
        if action_type == ACTION_CLOSE:
            return f"close {static_value['obj_id']}"

        # Toggle: "toggle {obj_id}"
        if action_type in [ACTION_TOGGLE_ON]:
            return f"turn on {static_value['obj_id']}"

        # Toggle: "toggle {obj_id}"
        if action_type in [ACTION_TOGGLE_OFF]:
            return f"turn off {static_value['obj_id']}"

        # Slice: "slice {obj_id}"
        if action_type == ACTION_SLICE:
            return f"slice {static_value['obj_id']}"

        # Goto: "goto {loc_desc}"
        if action_type == ACTION_GOTO:
            return f"goto {static_value['loc_desc']}"

        # Nav_obj: "navigate to {obj_id}"
        if action_type == ACTION_NAV_OBJ:
            return f"navigate to {static_value['obj_id']}"

        # GoTo coordinate: "GoTo (x,y,z)" — ThorConnector parses coordinates (thor_connector.py:237-245)
        if action_type == ACTION_GOTO_COORD:
            return static_key

        # Fallback: return static_key
        return static_key

    def _save_current_waypoint(self):
        """Save agent's current pose as a custom waypoint_#XX path point.
        Continues numbering from max existing index in path_points_keys.
        Appends to path_points/path_points_keys; the matching 'goto' action is
        generated automatically on the next _update_language_skill_set() call.
        Only valid for the current episode (cleared by reset -> restore_scene)."""
        agent = self.env.last_event.metadata['agent']
        pos = agent['position']
        rotation_y = agent['rotation']['y'] % 360
        horizon = agent['cameraHorizon']

        # Determine next index from existing path_points_keys
        if self.env.path_points is None:
            self.env.path_points = {}
            self.env.path_points_keys = []
            max_idx = -1
        else:
            max_idx = -1
            for key in self.env.path_points_keys:
                if key.startswith('waypoint_#'):
                    try:
                        num = int(key.split('#')[1])
                        if num > max_idx:
                            max_idx = num
                    except ValueError:
                        continue

        new_idx = max_idx + 1
        new_loc_desc = f'waypoint_#{new_idx:02d}'

        # Save to path_points (structure matches load_path_points)
        self.env.path_points[new_loc_desc] = {
            'x': pos['x'],
            'y': pos['y'],
            'z': pos['z'],
            'rotation': rotation_y,
            'horizon': horizon
        }
        self.env.path_points_keys.append(new_loc_desc)
        # Rebuild KD-tree so the new point is reflected in nearest-neighbour
        # queries (e.g. get_agent_position closest_path_point feedback).
        self.env.path_points_kdtree = self.env._build_path_points_kdtree()

        logger.info(f"Saved waypoint {new_loc_desc} at "
                    f"({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}) "
                    f"rotation={rotation_y:.1f} horizon={horizon:.1f}")
        return new_loc_desc

    @staticmethod
    def _tasks_root():
        # All eval_sets live under data/tasks/<eval_set>/<task_dir>/task.yaml.
        # New eval_sets are discovered at runtime by scanning this directory,
        # so there is no hard-coded ValidEvalSets list.
        return os.path.join(os.path.dirname(__file__), 'data', 'tasks')

    @classmethod
    def available_eval_sets(cls):
        """Return sorted names of all eval_sets currently present on disk."""
        tasks_root = cls._tasks_root()
        if not os.path.isdir(tasks_root):
            return []
        return sorted(d for d in os.listdir(tasks_root)
                      if os.path.isdir(os.path.join(tasks_root, d)))

    def _load_dataset(self, eval_set):
        # Scan data/tasks/<eval_set>/<task_dir>/task.yaml directly. Each task
        # instance lives in its own directory; repeat_idx is encoded in the dir
        # name suffix (_r<N>), so every directory is a single task instance.
        tasks_root = self._tasks_root()
        eval_dir = os.path.join(tasks_root, eval_set)
        if not os.path.isdir(eval_dir):
            raise FileNotFoundError(
                f"Task directory not found for eval_set='{eval_set}': {eval_dir}. "
                f"Available: {self.available_eval_sets()}")
        dataset = []
        for name in sorted(os.listdir(eval_dir)):
            yaml_path = os.path.join(eval_dir, name, 'task.yaml')
            if not os.path.isfile(yaml_path):
                continue
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            ins = cfg.get('instructions')
            if isinstance(ins, dict):
                instruction = ins.get(self.instruction_mode, '') or ins.get(
                    'micro', '')
            else:
                instruction = cfg.get('instruction', '')
            dataset.append({
                # path relative to data/, consumed by utils.load_task
                'task': f'tasks/{eval_set}/{name}',
                'repeat_idx': 0,
                'instruction': instruction,
            })
        if 0 <= self.down_sample_ratio < 1:
            select_every = round(1 / self.down_sample_ratio)
            dataset = dataset[0:len(dataset):select_every]
        return dataset

    def current_episode(self):
        """Return current episode"""
        res = None
        try:
            task = self.dataset[self._current_episode_num]
            res = utils.load_task(task)
        except:
            print("episode failed to load trying next episode")
            self.current_episode_num += 1
            self.current_episode()
        return res

    def _reset_controller(self, task):
        """Restore scene from a task name and replace instruction"""
        traj_data = utils.load_task(task)
        traj_data['turk_annotations']['anns'][
            task['repeat_idx']]['task_desc'] = task["instruction"]
        self.episode_data = traj_data
        args_dict = {
            'pframe': 300,
            'fast_epoch': False,
            'use_templated_goals': False,
            'dout': 'exp/model',
            'pp_folder': 'pp',
            'reward_config': self.reward_config_path,
            'max_steps': 1000
        }
        model_args = utils.dotdict(args_dict)

        self.episode_language_instruction = task["instruction"]
        scene = traj_data['scene']
        scene_name = 'FloorPlan%d' % scene['scene_num']
        logger.info(f"Restoring scene {scene_name}...")
        if scene.get('random_spawn'):
            # newly composed task: rebuild the spawned layout deterministically
            from pgibench.envs.pgi_alfred.env.scene_snapshot import random_initialize_scene
            rs = scene['random_spawn']
            random_initialize_scene(self.env, scene['scene_num'], rs['seed'],
                                    rs['object_counts'], rs.get('distractors'))
            # two-stage restore: re-create (above) then force each instance back
            # to its frozen pose. Instance-name match verified by probe.
            if scene.get('object_poses'):
                self.env.step({
                    'action': 'SetObjectPoses',
                    'objectPoses': scene['object_poses']
                })
        else:
            # legacy task: restore exact object_poses
            self.env.reset(scene_name)
            self.env.restore_scene(scene['object_poses'],
                                   scene['object_toggles'],
                                   scene['dirty_and_empty'])
            if scene['init_action']['action'] == 'TeleportFull':
                del scene['init_action']["rotateOnTeleport"]
                scene['init_action']["standing"] = True
            self.env.step(dict(scene['init_action']))
        self.env.set_task(traj_data,
                          model_args,
                          reward_type='dense',
                          max_episode_length=self._max_episode_steps)
        #############################
        # Note: Action space generation is now handled in reset() to support static dictionary

    def reset(self):
        """
        Reset the environment for a new episode.

        Returns:
            observation
        """
        assert self._current_episode_num < self.number_of_episodes
        self._reset_controller(self.dataset[self._current_episode_num])

        # Get current scene name after reset_controller
        current_scene = self.env.last_event.metadata['sceneName']

        # Always rebuild static action space on reset.
        # Clears custom waypoints from previous episode; path_points already
        # reloaded by restore_scene() in _reset_controller.
        if self._current_scene_name != current_scene:
            logger.info(
                f"Scene changed from {self._current_scene_name} to {current_scene}"
            )
        self._clear_static_action_structures()
        self._current_scene_name = current_scene
        self._build_static_action_dict()

        # Update dynamic display list with current locDesc values
        self._update_language_skill_set()

        self._current_step = 0
        self._cur_invalid_actions = 0
        self._current_episode_num += 1
        obs = {
            'head_rgb': self.env.last_event.frame,
        }
        self._reset = True
        self.episode_log = []
        self._episode_start_time = time.time()

        # Initialize thor tracker if enabled
        if self.enable_thor_tracking:
            episode_id = f"{self.dataset[self._current_episode_num - 1]['task']}_{self._current_episode_num - 1}"
            scene_name = self.env.last_event.metadata['sceneName']
            self.thor_tracker.reset_episode(
                episode_id=episode_id,
                scene_name=scene_name,
                task_instruction=self.episode_language_instruction)
            self.thor_tracker.update_step(
                self.env, {
                    'action_id': None,
                    'action_description': '(initial)',
                    'env_step': 0,
                    'last_action_success': True,
                    'env_feedback': 'environment reset'
                },
                language_skill_set=self.language_skill_set)

        return obs

    def step(self,
             action,
             reasoning='',
             generate_question=False,
             target_bbox=None,
             target_point=None):
        """
        Execute a single environment step.

        Args:
            action: int index into language_skill_set (full/visible mode), or a command
                    string (command mode; also 'goto ...' / 'GoTo (...)' in any mode).
            reasoning: Optional reasoning trace.
            generate_question: If True and EQA enabled, generate a random question.
            target_bbox: [x1,y1,x2,y2] normalized [0,1] box (command mode, 'bbox' reference).
            target_point: [cx,cy] normalized [0,1] point (command mode, 'point' reference).
        Returns:
            tuple: (observation, reward, done, info)
        """
        assert self._reset, 'Reset env before stepping'
        info = {}
        self._current_step += 1

        # 1. Resolve display_skill / raw command
        if type(action) == int:
            display_skill = self.language_skill_set[action]
        elif type(action) == str:
            display_skill = action
        else:
            raise NotImplementedError

        resolved_target = None
        interaction_str = None
        done_forced = False

        # 2. Execute
        if self.action_display_mode == 'command' and isinstance(action, str):
            # command mode: parse free-text -> resolve reference -> interaction string
            parsed = parse_command(display_skill)
            if parsed is None:
                return self._invalid_action_return(
                    action, reasoning,
                    f'unrecognised command: {display_skill!r}')
            if target_bbox is not None:
                parsed.target_bbox = target_bbox
            if target_point is not None:
                parsed.target_point = target_point
            if parsed.action_type == 'done':
                event = {'success': True, 'message': 'Done'}
                done_forced = True
            elif (parsed.action_type == ACTION_GLOBAL
                  and parsed.global_name == 'SaveWaypoint'):
                # SaveWaypoint records the current pose as a new path point; it must NOT
                # be forwarded to llm_skill_interact (which does not recognise it). Only
                # meaningful when created_waypoint nav_mode is enabled.
                if 'created_waypoint' not in self.nav_mode:
                    event = {
                        'success':
                        False,
                        'message':
                        'SaveWaypoint requires created_waypoint nav_mode'
                    }
                else:
                    new_loc_desc = self._save_current_waypoint()
                    event = {
                        'success': True,
                        'message': f'Saved waypoint {new_loc_desc}'
                    }
            else:
                ctx = self._build_resolver_context()
                held_obj_id, _ = self.env.get_held_object()
                interaction_str, resolved_target, reason = resolve_command(
                    parsed, ctx, held_obj_id, self.object_reference_mode)
                if interaction_str is None:
                    return self._invalid_action_return(action, reasoning,
                                                       reason, resolved_target)
                event = self.env.llm_skill_interact(interaction_str)
            if not event['success']:
                self._cur_invalid_actions += 1
        elif isinstance(display_skill, str) and display_skill.startswith("GoTo ") \
                and "(" in display_skill:
            # free-format GoTo (x,y,z) coordinate action
            event = self.env.llm_skill_interact(display_skill)
            if not event['success']:
                self._cur_invalid_actions += 1
        elif isinstance(display_skill,
                        str) and display_skill.startswith("goto "):
            # 'goto waypoint_#NN' path-point action (dynamic, not in static dict)
            loc_desc = display_skill[len("goto "):].strip()
            event = self.env.llm_skill_interact(f"goto {loc_desc}")
            if not event['success']:
                self._cur_invalid_actions += 1
        else:
            # enumerated mode: display_skill -> static_key -> interaction string
            static_key = self.display_to_static_key.get(display_skill)
            if static_key is None:
                error_msg = f"Invalid action: {display_skill} not found in current action space"
                logger.warning(error_msg)
                return self._invalid_action_return(action, reasoning,
                                                   error_msg)
            static_value = self.static_language_skill_dict[static_key]
            if static_value['action_type'] == ACTION_GLOBAL \
                    and static_key == 'save waypoint':
                new_loc_desc = self._save_current_waypoint()
                event = {
                    'success': True,
                    'message': f'Saved waypoint {new_loc_desc}'
                }
            else:
                interaction_str = self._build_interaction_string(
                    static_key, static_value)
                event = self.env.llm_skill_interact(interaction_str)
            if not event['success']:
                self._cur_invalid_actions += 1

        # 3. Reward and goal conditions
        reward, done = self.env.get_transition_reward()
        subgoal_met = self.env.get_goal_conditions_met()
        info['task_success'] = float(self.env.get_goal_satisfied())
        info['task_progress'] = subgoal_met[0] / subgoal_met[1]
        subtask_details = self.env.get_subtask_details()
        info['num_subtasks'] = len(subtask_details)
        info['total_goal_conditions'] = subgoal_met[1]
        info['subtask_completion'] = (
            sum(1 for d in subtask_details if d['completed']) /
            len(subtask_details)) if subtask_details else 0.0
        info['subtask_details'] = subtask_details
        if done_forced:
            done = True

        obs = {'head_rgb': self.env.last_event.frame}

        # 4. Check if episode should end
        if (self._current_step >= self._max_episode_steps
                or info['task_success']
                or self._cur_invalid_actions >= self._max_invalid_actions):
            done = True

        # 5. Environment feedback and info
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['env_feedback'] = self.get_env_feedback(event)
        info['episode_elapsed_seconds'] = time.time(
        ) - self._episode_start_time
        info['last_action_success'] = float(event['success'])
        info['agent_position'] = self.env.get_agent_position()
        info['object_states'] = {
            "cooled_objects": self.env.cooled_objects,
            "heated_objects": self.env.heated_objects,
            "cleaned_objects": self.env.cleaned_objects,
            "visible_objs": self.env.get_visible_objects(),
            "nearby_objs": self.env.get_nearby_objects()
        }
        info['action_id'] = action
        info['action_description'] = interaction_str or display_skill
        info['reasoning'] = reasoning
        info['resolved_target'] = resolved_target
        info['object_reference_mode'] = self.object_reference_mode
        self.episode_log.append(info)

        # 6. Update dynamic display list after action (before tracker)
        self._update_language_skill_set()

        # 7. Update thor tracker if enabled (command mode records the object catalog
        #    as available_actions instead of the enumerated skill list)
        if self.enable_thor_tracking:
            tracker_skill_set = self.language_skill_set
            if self.action_display_mode == 'command':
                tracker_skill_set = sorted(self._get_allowed_objs())
            self.thor_tracker.update_step(env=self.env,
                                          action_info=info,
                                          reasoning=reasoning,
                                          language_skill_set=tracker_skill_set)

        # 8. Generate EQA question if requested
        if self.enable_eqa and generate_question and self.eqa_generator:
            question = self.eqa_generator.generate_random_question(
                timestep=self._current_step)
            if question:
                info['eqa_question'] = question

        return obs, reward, done, info

    def _sanitize_feedback_message(self, message):
        """Replace simulator-internal objectIds in a feedback message with the
        natural reference form used in the current mode.

        ObjectIds (e.g. ``Mug|-03.52|+00.00|+02.72``) and other low-level
        object references are rewritten to:
            - command mode 'id'            -> ``mug_#0`` (env.id_2_name)
            - command mode 'bbox/point/
               ordinal/type'               -> object type (e.g. ``Mug``)
            - enumerated mode (None)       -> locDesc (e.g. ``mug on table``)

        Any objectId token without a known mapping is left unchanged. The
        rewrite is applied only inside the message body; it never changes which
        object the error refers to.

        Args:
            message (str): Raw feedback message possibly containing objectIds.

        Returns:
            str: Message with objectIds replaced by the mode-appropriate
                 reference form.
        """
        if not message:
            return message

        # All simulator objectIds contain the '|' separator (they are encoded as
        # '<Type>|<x>|<y>|<z>'). Match every such token so a message never leaks
        # raw coordinates to the agent.
        tokens = set(re.findall(r'[A-Za-z_][\w]*\|[-+\d.eE|]+', message))

        replacements = {}
        for obj_id in tokens:
            # Enumerated mode (object_reference_mode is None): prefer the human
            # locDesc (e.g. 'mug on table') shown in the action list.
            if self.object_reference_mode is None:
                desc = self.object_id_to_current_locdesc.get(obj_id)
                if not desc:
                    # Fallback: object type if locDesc is unavailable.
                    desc = self.env.id_2_name.get(obj_id)
                    if desc:
                        desc = obj_id.split('|')[0]
                    else:
                        desc = obj_id
                replacements[obj_id] = desc
            elif self.object_reference_mode == 'id':
                # Command id mode: agent addresses objects as '<type>_#<n>'.
                replacements[obj_id] = self.env.id_2_name.get(obj_id, obj_id)
            else:
                # bbox/point/ordinal/type: the agent refers to objects by type
                # plus a visual cue, so expose only the object type.
                replacements[obj_id] = obj_id.split('|')[0]

        if not replacements:
            return message
        # Sort by length (longest first) so a prefix objectId is not rewritten
        # before a longer one that shares the prefix.
        for obj_id in sorted(replacements, key=len, reverse=True):
            message = message.replace(obj_id, replacements[obj_id])
        
        # Replace "Agent 0" with "you" for more natural language
        message = message.replace("Agent 0", "you")
        
        return message

    def get_env_feedback(self, info):
        """
        Generate feedback message for the current step.
        Args:
            info (dict): Action execution information (event metadata from llm_skill_interact)
        Returns:
            str: Descriptive message about step outcome
        """
        msg = ''
        if info["success"]:
            msg = f"Last action executed successfully."
        else:
            body = self._sanitize_feedback_message(info['message'])
            msg = (f"Last action is invalid. {body} Please reflect on the "
                   f"reason for this failure and try a different action to "
                   f"correct it.")
        return msg

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def get_available_waypoints(self):
        """Return waypoint entries available for `goto` in the current scene.

        Returns a list of dicts (one per path point) when default/created
        waypoint navigation is active and the scene has path points; empty
        otherwise. Each entry exposes the waypoint's name, its (x, z) ground-
        plane position (matching the agent-state convention which omits y) and
        its facing orientation (cardinal direction, numeric rotation and camera
        horizon), so the planner can render coordinates/orientation into the
        user prompt without reading ThorConnector internals::

            {'name': 'waypoint_#00', 'x': 1.25, 'z': 0.50,
             'rotation': 0.0, 'direction': 'north', 'horizon': 30.0}
        """
        if not ({'default_waypoint', 'created_waypoint'} & set(self.nav_mode)):
            return []
        keys = getattr(self.env, 'path_points_keys', None) or []
        path_points = getattr(self.env, 'path_points', None) or {}
        waypoints = []
        for name in keys:
            pt = path_points.get(name) or {}
            rotation = pt.get('rotation', 0.0)
            waypoints.append({
                'name': name,
                'x': pt.get('x', 0.0),
                'z': pt.get('z', 0.0),
                'rotation': rotation,
                'direction': _rotation_to_direction(rotation),
                'horizon': pt.get('horizon', 0.0),
            })
        return waypoints

    def save_image(self, *args, **kwargs):
        """Save current agent view as a PNG image."""
        episode_idx = self._current_episode_num if not len(
            self.selected_indexes
        ) else self.selected_indexes[self._current_episode_num - 1] + 1

        folder = self.log_path + '/images/episode_{}'.format(episode_idx)
        if not os.path.exists(folder):
            os.makedirs(folder)
        img = Image.fromarray(self.env.last_event.frame)
        # Decide whether to annotate bboxes and which label set to use.
        # - detection_box / visual_prompt: draw with id_2_name labels (existing behaviour)
        # - command mode with a visual reference scheme (id/bbox/point/ordinal):
        #   draw bboxes; label with id_2_name IDs only for 'id', else type names.
        annotate = self.detection
        name_translation = self.env.id_2_name
        if self.action_display_mode == 'command' \
                and self.object_reference_mode in ('id', 'ordinal'):
            annotate = True
            name_translation = self.env.id_2_name if self.object_reference_mode == 'id' else {}
        if annotate:
            visible_objs = set(self.env.get_visible_objects())
            class_and_boxes = {
                k: v
                for k, v in self.env.last_event.instance_detections2D.items()
                if k in visible_objs
            }
            # filter_by_alfred_objs=False: visible_objs already restricts the set
            # to get_visible_objects(), so draw every passed object to keep the
            # labelled set identical to the resolver's visible set.
            img = utils.draw_boxes(img,
                                   class_and_boxes,
                                   name_translation=name_translation,
                                   filter_by_alfred_objs=False)

        # time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        image_path = os.path.join(folder, 'episode_{}_step_{}.png'.format(
            episode_idx, self._current_step))  #, time_stamp))
        img.save(image_path)
        return image_path

    def save_episode_log(self, log_path=None):
        log_path = log_path if log_path is not None else self.log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        # time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        episode_idx = self._current_episode_num if not len(
            self.selected_indexes
        ) else self.selected_indexes[self._current_episode_num - 1] + 1
        filename = 'episode_{}_step_{}.json'.format(
            episode_idx, self._current_step)  #, time_stamp)
        if len(self.episode_log):
            with open(os.path.join(log_path, filename), 'w') as f:
                for item in self.episode_log:
                    if 'object_states' in item:
                        item.pop('object_states')
                    try:
                        json.dump(item, f, ensure_ascii=False)
                    except:
                        import pdb
                        pdb.set_trace()
                    f.write('\n')

    def save_thor_tracking(self, log_path=None):
        """
        Save thor tracking data to JSON file.

        Returns:
            str: Path to saved file, or None if tracking is disabled
        """
        if not self.enable_thor_tracking:
            return None

        if log_path is None:
            log_path = self.log_path

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        episode_idx = self._current_episode_num if not len(
            self.selected_indexes
        ) else self.selected_indexes[self._current_episode_num - 1] + 1

        filename = 'episode_{}_thor.json'.format(episode_idx)
        filepath = os.path.join(log_path, filename)

        self.thor_tracker.save_to_file(filepath)

        logger.info(f"thor tracking data saved to: {filepath}")
        return filepath

    def save_eqa_questions(self, num_questions=10, filepath=None):
        """
        Generate and save EQA questions for the completed episode.

        Args:
            num_questions: Number of questions to generate

        Returns:
            str: Path to saved file, or None if EQA is disabled
        """
        if not self.enable_eqa or not self.eqa_generator:
            return None

        episode_idx = self._current_episode_num if not len(
            self.selected_indexes
        ) else self.selected_indexes[self._current_episode_num - 1] + 1

        # Generate batch of questions
        questions = self.eqa_generator.generate_batch_questions(num_questions)

        # Prepare episode metadata
        episode_metadata = {
            "episode_id": self.thor_tracker.episode_id,
            "scene_name": self.thor_tracker.scene_name,
            "task_instruction": self.thor_tracker.task_instruction,
            "total_timesteps": self._current_step,
            "num_questions_generated": len(questions)
        }

        # Save to file
        if filepath is None:
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            filename = 'episode_{}_eqa.json'.format(episode_idx)
            filepath = os.path.join(self.log_path, filename)

        saved_question = self.eqa_generator.save_to_json(
            filepath, episode_metadata=episode_metadata)

        logger.info(
            f"EQA questions saved to: {filepath} ({len(questions)} questions)")
        return saved_question

    def get_agent_position(self):
        return self.env.get_agent_position()

    def get_state_info(self):
        """Return a structured dict mirroring exactly what ``process_prompt()``
        writes into the user prompt under the current ``action_display_mode`` /
        ``object_reference_mode`` configuration.

        A high-level agent can consume this directly instead of parsing the
        prompt text or assembling the pieces itself. Each key is populated only
        when the corresponding prompt section would actually be shown:

            task                 <- '## Current Task'                 (always)
            agent                <- '## Agent Current State'          (always)
            last_action_result   <- '## Last Action Result' (None on first step)
            available_actions    <- '## Available Actions'   (enumerated only)
            object_catalog       <- '## Visible Objects'      (command + id)
            available_waypoints  <- '## Available Waypoints'  (command only)
                                   list of {name, x, z, rotation, direction, horizon}
        """
        state = {}
        # 1. Current Task — always present.
        state['task'] = {'instruction': self.episode_language_instruction}

        # 2. Agent Current State — always present. The dict from
        #    get_agent_position already carries position/rotation_y/horizon/
        #    direction plus closest_path_point / is_on_path_point /
        #    distance_to_closest_path_point / relative_direction_to_path_point.
        state['agent'] = self.get_agent_position()

        # 3. Last Action Result — only when a previous step recorded a usable
        #    last_action_success (1.0/0.0). On the first step there is no prior
        #    action, so this is None, mirroring process_prompt's gating on
        #    `last_action_result is not None`.
        last = None
        if self.episode_log:
            last_info = self.episode_log[-1]
            success = last_info.get('last_action_success')
            if success is not None:
                last = {
                    'success': float(success) > 0,
                    'feedback': last_info.get('env_feedback', ''),
                    'action_id': last_info.get('action_id'),
                    'action_description': last_info.get('action_description', ''),
                }
        state['last_action_result'] = last

        is_command = (self.action_display_mode == 'command')

        # 4. Available Actions — enumerated list shown only for full/visible
        #    modes; command mode uses the action grammar in the system prompt
        #    instead, so the key is None there.
        state['available_actions'] = (
            list(self.language_skill_set) if not is_command else None)

        # 5. Visible Objects (id, normalised bbox) — only command + id mode.
        #    Reuses get_object_catalog so the text stays consistent with the
        #    '## Visible Objects' prompt section and the image overlay.
        state['object_catalog'] = (
            self.get_object_catalog(mode='id')
            if (is_command and self.object_reference_mode == 'id') else None)

        # 6. Available Waypoints — only command mode; returns waypoint names
        #    (empty list if waypoint nav modes are inactive).
        state['available_waypoints'] = (
            self.get_available_waypoints() if is_command else None)

        return state

    def explore_scene(self, need_reset=False):
        """
        Execute all 'goto path points' actions sequentially and teleport back to initial position.
        This method is used for scene exploration by visiting all predefined navigation waypoints.

        Returns:
            dict: List of (obs, reward, done, info) tuples for each goto action
        """
        assert self._reset, 'Environment must be reset before calling this method'

        result = []

        # 1. Record initial position
        initial_pos = self.env.get_agent_position()

        # 2. Collect all goto actions from current language_skill_set
        goto_actions = []
        for action in self.language_skill_set:
            if action.startswith("goto "):
                goto_actions.append(action)

        if not goto_actions:
            return result

        # 3. Execute each goto action sequentially
        for action_text in goto_actions:
            # Execute the goto action
            obs, reward, done, info = self.step(action_text)
            img_path = self.save_image()
            info['image_path'] = img_path

            # Record step result
            result.append((obs, reward, done, dict(info)))

        # 4. Teleport back to initial position
        self.env.go_to(x=initial_pos['position']['x'],
                       y=initial_pos['position']['y'],
                       z=initial_pos['position']['z'],
                       rotation=initial_pos['rotation_y'],
                       horizon=initial_pos['horizon'])

        # 5. Verify final position
        final_pos = self.env.last_event.metadata['agent']['position']
        final_rotation = self.env.last_event.metadata['agent']['rotation']['y']
        final_horizon = self.env.last_event.metadata['agent']['cameraHorizon']

        # 6. Check if position matches, reset environment if not
        pos_match = (abs(final_pos['x'] - initial_pos['position']['x']) < 0.01
                     and final_pos['y'] > 0.0 and
                     abs(final_pos['z'] - initial_pos['position']['z']) < 0.01)
        rot_match = abs(final_rotation - initial_pos['rotation_y']) < 0.1
        horizon_match = abs(final_horizon - initial_pos['horizon']) < 0.1

        if (not (pos_match and rot_match and horizon_match)) and need_reset:
            # Position mismatch detected, reset environment
            self._current_episode_num -= 1  # Decrement so reset() will use same episode
            self.reset()
        else:
            # Update dynamic action space after successful return
            self._update_language_skill_set()

        return result

    def close(self):
        """Terminate the environment."""
        self.env.stop()


if __name__ == "__main__":
    """
    Interactive usage of PGIAlfEnv. Set ACTION_DISPLAY_MODE to try the three
    parallel display modes:
      'visible' / 'full' : enumerated action list, enter an action index or name
      'command'          : free-text commands, e.g. 'pick up mug_#0'
    """
    import json

    ACTION_DISPLAY_MODE = 'command'  # 'full' | 'visible' | 'command'
    OBJECT_REFERENCE_MODE = 'id'  # id|bbox|point|desc|ordinal|type (command only)

    env = PGIAlfEnv(eval_set='compose',
                    down_sample_ratio=1.0,
                    selected_indexes=[],
                    visual_prompt=False,
                    action_display_mode=ACTION_DISPLAY_MODE,
                    object_reference_mode=OBJECT_REFERENCE_MODE,
                    enable_eqa=True)

    obs = env.reset()
    print(f"action_display_mode={env.action_display_mode} "
          f"object_reference_mode={env.object_reference_mode}")
    if env.eqa_generator:
        print(
            f"Registered EQA generators: {env.eqa_generator.get_registered_generators()}"
        )

    env.explore_scene()

    for step in range(30):
        if env.action_display_mode == 'command':
            print('\n--- Action grammar ---')
            print('pick up <obj> | put down <obj> in/on <recep> | drop | '
                  'open/close <rec> | turn on/off <dev> | slice <obj>')
            print('look up/down rotate left/right move* | goto <wp> | '
                  'navigate to <obj> | GoTo (x,y,z) | Done')
            if env.object_reference_mode == 'type':
                print('--- Object catalog ---')
                print(env.get_object_catalog())
            cmd = input('command: ').strip()
            if not cmd:
                continue
            target_bbox = target_point = None
            if env.object_reference_mode == 'bbox':
                bb = input('target_bbox x1,y1,x2,y2 in [0,1] (blank=skip): ').strip()
                if bb:
                    target_bbox = [float(v) for v in bb.split(',')]
            elif env.object_reference_mode == 'point':
                pp = input('target_point cx,cy in [0,1] (blank=skip): ').strip()
                if pp:
                    target_point = [float(v) for v in pp.split(',')]
            print(f"Executing command: {cmd}")
            obs, reward, done, info = env.step(cmd,
                                               generate_question=True,
                                               target_bbox=target_bbox,
                                               target_point=target_point)
        else:
            print('\n',
                  [(i, name) for i, name in enumerate(env.language_skill_set)],
                  '\n')
            raw = input('action id (or action name, <0 to quit): ').strip()
            if not raw:
                continue
            if raw.lstrip('-').isdigit():
                action = int(raw)
                if action < 0:
                    break
            elif raw in env.language_skill_set:
                action = raw
            else:
                print('unknown action, skip')
                continue
            print(f"Executing: {action}")
            obs, reward, done, info = env.step(action, generate_question=True)

        info.pop('eqa_question', None)
        print({
            k: v
            for k, v in info.items()
            if k in ('env_feedback', 'last_action_success', 'task_success',
                     'resolved_target', 'action_description')
        })
        env.save_image()

        if done:
            break

    # Generate and save batch of EQA questions at episode end
    if env.enable_eqa:
        print("\nGenerating batch EQA questions...")
        saved_question = env.save_eqa_questions(num_questions=50)
        if saved_question:
            print(
                f"Total questions generated: {saved_question['summary']['total_questions']}"
            )
            print(
                f"Question types: {saved_question['summary']['question_types']}"
            )
            print(
                f"Timesteps covered: {saved_question['summary']['timesteps_covered']}"
            )

    env.close()
