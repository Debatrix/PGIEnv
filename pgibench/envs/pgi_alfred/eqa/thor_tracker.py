"""
Spatiotemporal Tracker for EB-ALFRED Environment

This module provides comprehensive tracking of scene changes, agent observations,
and actions to enable QA generation for spatiotemporal memory evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
import json


@dataclass
class ObjectState:
    """Immutable snapshot of an object's state at a specific timestep."""
    object_id: str
    object_type: str
    position: Dict[str, float]  # {'x': float, 'y': float, 'z': float}
    rotation: Dict[str, float]  # {'x': float, 'y': float, 'z': float}

    # Boolean properties
    pickupable: bool
    receptacle: bool
    openable: bool
    toggleable: bool
    sliceable: bool
    dirtyable: bool

    # State flags
    is_open: bool = False
    is_toggled_on: bool = False
    is_sliced: bool = False
    is_dirty: bool = False
    is_visible: bool = False
    distance: float = 0.0

    # Special states (from environment tracking)
    is_heated: bool = False
    is_cooled: bool = False
    is_cleaned: bool = False

    # Containment relationships
    parent_receptacles: List[str] = field(default_factory=list)
    receptacle_object_ids: List[str] = field(default_factory=list)

    # Natural language description
    loc_desc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'objectId': self.object_id,
            'objectType': self.object_type,
            'position': self.position,
            'rotation': self.rotation,
            'pickupable': self.pickupable,
            'receptacle': self.receptacle,
            'openable': self.openable,
            'toggleable': self.toggleable,
            'sliceable': self.sliceable,
            'dirtyable': self.dirtyable,
            'isOpen': self.is_open,
            'isToggledOn': self.is_toggled_on,
            'isSliced': self.is_sliced,
            'isDirty': self.is_dirty,
            'isVisible': self.is_visible,
            'distance': self.distance,
            'isHeated': self.is_heated,
            'isCooled': self.is_cooled,
            'isCleaned': self.is_cleaned,
            'parentReceptacles': self.parent_receptacles,
            'receptacleObjectIds': self.receptacle_object_ids,
            'locDesc': self.loc_desc
        }


@dataclass
class AgentState:
    """Snapshot of agent state at a specific timestep."""
    timestep: int

    # Position and orientation
    position: Dict[str, float]  # {'x': float, 'y': float, 'z': float}
    rotation: Dict[str, float]  # {'x': float, 'y': float, 'z': float}
    camera_horizon: float

    # Directional information
    facing_direction: str  # 'north', 'northeast', etc.
    facing_angle: float

    # Inventory
    inventory_objects: List[Dict[str, str]] = field(default_factory=list)

    # [{'objectId': str, 'objectType': str}, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestep': self.timestep,
            'position': self.position,
            'rotation': self.rotation,
            'cameraHorizon': self.camera_horizon,
            'facingDirection': self.facing_direction,
            'facingAngle': self.facing_angle,
            'inventoryObjects': self.inventory_objects
        }


@dataclass
class ObservationSnapshot:
    """Snapshot of what the agent could observe at a timestep."""
    timestep: int

    # Object visibility
    visible_objects: Set[str]  # objectIds
    nearby_objects: Set[str]  # objectIds within 0.5m
    historically_seen_objects: Set[str]  # cumulative seen objects

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestep': self.timestep,
            'visibleObjects': list(self.visible_objects),
            'nearbyObjects': list(self.nearby_objects),
            'historicallySeenObjects': list(self.historically_seen_objects)
        }


@dataclass
class ActionRecord:
    """Record of an action taken at a timestep."""
    timestep: int

    # Action information
    available_actions: List[str]  # All actions in action space
    executed_action: str  # Action description
    action_success: bool

    # Environment feedback
    env_feedback: str

    # Reasoning (if available)
    reasoning: str = ""

    # command-mode only: objectId the command resolved to (None for enumerated mode)
    resolved_target: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestep': self.timestep,
            'availableActions': self.available_actions,
            'executedAction': self.executed_action,
            'actionSuccess': self.action_success,
            'envFeedback': self.env_feedback,
            'reasoning': self.reasoning,
            'resolvedTarget': self.resolved_target
        }


class ThorTracker:
    """
    Comprehensive tracker for spatiotemporal information in EB-ALFRED environment.

    Records object states, agent states, observations, and actions over time
    to enable QA generation for spatiotemporal memory evaluation.
    """

    def __init__(self):
        """Initialize the tracker."""
        # Temporal data storage
        self.timesteps: List[int] = []

        # Object tracking: {object_id: {timestep: ObjectState}}
        self.object_history: Dict[str, Dict[int,
                                            ObjectState]] = defaultdict(dict)

        # Agent state tracking: {timestep: AgentState}
        self.agent_states: Dict[int, AgentState] = {}

        # Observation tracking: {timestep: ObservationSnapshot}
        self.observations: Dict[int, ObservationSnapshot] = {}

        # Action tracking: {timestep: ActionRecord}
        self.actions: Dict[int, ActionRecord] = {}

        # Cumulative seen objects
        self.historically_seen_objects: Set[str] = set()

        # Current timestep
        self.current_timestep: int = 0

        # Episode metadata
        self.episode_id: Optional[str] = None
        self.scene_name: Optional[str] = None
        self.task_instruction: Optional[str] = None

        # Scene static information (constant across episode)
        self.all_object_ids: Set[str] = set()
        self.object_types: Dict[str, str] = {}  # {object_id: object_type}

    def reset_episode(self, episode_id: str, scene_name: str,
                      task_instruction: str):
        """
        Reset tracker for a new episode.

        Args:
            episode_id: Unique identifier for the episode
            scene_name: Name of the scene (e.g., 'FloorPlan1')
            task_instruction: Natural language task instruction
        """
        self.timesteps = []
        self.object_history.clear()
        self.agent_states.clear()
        self.observations.clear()
        self.actions.clear()
        self.historically_seen_objects.clear()
        self.current_timestep = 0

        self.episode_id = episode_id
        self.scene_name = scene_name
        self.task_instruction = task_instruction

        # Note: object_types will be populated on first update

    def update_step(self,
                    env,
                    action_info: Dict[str, Any] = {},
                    reasoning: str = "",
                    language_skill_set: Optional[List[str]] = None):
        """
        Update tracker with current timestep information.

        Args:
            env: ThorConnector instance (has last_event, get_visible_objects, etc.)
            action_info: Dictionary containing action execution info
                - action_id: int or str
                - action_description: str
                - last_action_success: bool
                - env_feedback: str
                - env_step: int
            reasoning: Optional reasoning from the planner
            language_skill_set: Optional list of available language skills (actions)
                                 If not provided, will try to get from env
        """
        # Extract metadata from environment
        event_metadata = env.last_event.metadata
        current_step = action_info.get('env_step', self.current_timestep)

        self.timesteps.append(current_step)
        self.current_timestep = current_step

        # 1. Record object states
        self._update_object_states(event_metadata, env)

        # 2. Record agent state
        self._update_agent_state(event_metadata, current_step)

        # 3. Record observations
        self._update_observations(event_metadata, env, current_step)

        # 4. Record action
        self._update_action(action_info, env, reasoning, current_step,
                            language_skill_set)

    def _update_object_states(self, metadata: Dict, env):
        """Update object states from metadata."""
        # Get special state sets from environment
        cooled_objects = getattr(env, 'cooled_objects', set())
        heated_objects = getattr(env, 'heated_objects', set())
        cleaned_objects = getattr(env, 'cleaned_objects', set())

        for obj in metadata['objects']:
            obj_id = obj['objectId']

            # Initialize object type tracking
            if obj_id not in self.object_types:
                self.object_types[obj_id] = obj['objectType']
                self.all_object_ids.add(obj_id)

            # Create ObjectState snapshot
            state = ObjectState(
                object_id=obj_id,
                object_type=obj['objectType'],
                position=obj['position'],
                rotation=obj['rotation'],
                pickupable=obj.get('pickupable', False),
                receptacle=obj.get('receptacle', False),
                openable=obj.get('openable', False),
                toggleable=obj.get('toggleable', False),
                sliceable=obj.get('sliceable', False),
                dirtyable=obj.get('dirtyable', False),
                is_open=obj.get('isOpen', False),
                is_toggled_on=obj.get('isToggledOn', False),
                is_sliced=obj.get('isSliced', False),
                is_dirty=obj.get('isDirty', False),
                is_visible=obj.get('visible', False),
                distance=obj.get('distance', 0.0),
                is_heated=obj_id in heated_objects,
                is_cooled=obj_id in cooled_objects,
                is_cleaned=obj_id in cleaned_objects,
                parent_receptacles=obj.get('parentReceptacles', []),
                receptacle_object_ids=obj.get('receptacleObjectIds', []),
                loc_desc=obj.get('locDesc', obj['objectType']))

            self.object_history[obj_id][self.current_timestep] = state

    def _update_agent_state(self, metadata: Dict, timestep: int):
        """Update agent state from metadata."""
        agent_meta = metadata['agent']

        # Get facing direction
        facing_result = agent_meta.get('facingDirection', ('unknown', 0.0))
        if isinstance(facing_result, tuple) and len(facing_result) == 2:
            facing_direction, facing_angle = facing_result
        else:
            facing_direction = 'unknown'
            facing_angle = 0.0

        state = AgentState(timestep=timestep,
                           position=agent_meta['position'],
                           rotation=agent_meta['rotation'],
                           camera_horizon=agent_meta.get('cameraHorizon', 0.0),
                           facing_direction=facing_direction,
                           facing_angle=facing_angle,
                           inventory_objects=[{
                               'objectId': inv['objectId'],
                               'objectType': inv['objectType']
                           } for inv in agent_meta.get('inventoryObjects', [])
                                              ])

        self.agent_states[timestep] = state

    def _update_observations(self, metadata: Dict, env, timestep: int):
        """Update observation snapshot."""
        # Get visible and nearby objects
        visible_objs = set(env.get_visible_objects())
        nearby_objs = set(env.get_nearby_objects())

        # Update cumulative seen objects
        self.historically_seen_objects.update(visible_objs)

        observation = ObservationSnapshot(
            timestep=timestep,
            visible_objects=visible_objs,
            nearby_objects=nearby_objs,
            historically_seen_objects=self.historically_seen_objects.copy())

        self.observations[timestep] = observation

    def _update_action(self,
                       action_info: Dict,
                       env,
                       reasoning: str,
                       timestep: int,
                       language_skill_set: Optional[List[str]] = None):
        """Update action record."""
        # Get available actions from parameter or environment
        if language_skill_set is not None:
            available_actions = list(language_skill_set)
        elif hasattr(env, 'language_skill_set'):
            available_actions = list(env.language_skill_set)
        else:
            available_actions = []

        # Get action description
        action_id = action_info.get('action_id', -1)
        if language_skill_set is not None and isinstance(action_id, int):
            if action_id < len(language_skill_set):
                action_desc = language_skill_set[action_id]
            else:
                action_desc = action_info.get('action_description',
                                              str(action_id))
        elif isinstance(action_id, int) and hasattr(env, 'language_skill_set'):
            if action_id < len(env.language_skill_set):
                action_desc = env.language_skill_set[action_id]
            else:
                action_desc = str(action_id)
        else:
            action_desc = action_info.get('action_description', str(action_id))

        action = ActionRecord(timestep=timestep,
                              available_actions=available_actions,
                              executed_action=action_desc,
                              action_success=bool(
                                  action_info.get('last_action_success',
                                                  True)),
                              env_feedback=action_info.get('env_feedback', ''),
                              reasoning=reasoning,
                              resolved_target=action_info.get('resolved_target'))

        self.actions[timestep] = action

    # ==================== QUERY INTERFACE ====================

    def get_object_state_at_step(self, object_id: str,
                                 timestep: int) -> Optional[ObjectState]:
        """
        Get state of a specific object at a specific timestep.

        Args:
            object_id: Unique object identifier
            timestep: Timestep to query

        Returns:
            ObjectState if found, None if object not recorded at timestep
        """
        return self.object_history.get(object_id, {}).get(timestep)

    def get_object_location_history(
            self, object_id: str) -> Dict[int, Dict[str, float]]:
        """
        Get position history of an object across all timesteps.

        Args:
            object_id: Unique object identifier

        Returns:
            Dictionary mapping timestep to position {'x': float, 'y': float, 'z': float}
        """
        history = {}
        for ts, state in self.object_history.get(object_id, {}).items():
            history[ts] = state.position
        return history

    def get_object_state_changes(
            self, object_id: str) -> List[Tuple[int, str, Any, Any]]:
        """
        Get all state changes for an object across the episode.

        Args:
            object_id: Unique object identifier

        Returns:
            List of (timestep, change_type, old_value, new_value) tuples
        """
        changes = []
        object_states = self.object_history.get(object_id, {})

        if not object_states:
            return changes

        # Sort by timestep
        sorted_timesteps = sorted(object_states.keys())

        if len(sorted_timesteps) < 2:
            return changes

        # Compare consecutive states
        for i in range(1, len(sorted_timesteps)):
            prev_ts = sorted_timesteps[i - 1]
            curr_ts = sorted_timesteps[i]

            prev_state = object_states[prev_ts]
            curr_state = object_states[curr_ts]

            # Check for state changes
            if prev_state.is_open != curr_state.is_open:
                changes.append((curr_ts, 'isOpen', prev_state.is_open,
                                curr_state.is_open))

            if prev_state.is_toggled_on != curr_state.is_toggled_on:
                changes.append(
                    (curr_ts, 'isToggledOn', prev_state.is_toggled_on,
                     curr_state.is_toggled_on))

            if prev_state.is_sliced != curr_state.is_sliced:
                changes.append((curr_ts, 'isSliced', prev_state.is_sliced,
                                curr_state.is_sliced))

            if prev_state.is_dirty != curr_state.is_dirty:
                changes.append((curr_ts, 'isDirty', prev_state.is_dirty,
                                curr_state.is_dirty))

            if prev_state.is_heated != curr_state.is_heated:
                changes.append((curr_ts, 'isHeated', prev_state.is_heated,
                                curr_state.is_heated))

            if prev_state.is_cooled != curr_state.is_cooled:
                changes.append((curr_ts, 'isCooled', prev_state.is_cooled,
                                curr_state.is_cooled))

            if prev_state.is_cleaned != curr_state.is_cleaned:
                changes.append((curr_ts, 'isCleaned', prev_state.is_cleaned,
                                curr_state.is_cleaned))

            # Check for position changes
            if (prev_state.position['x'] != curr_state.position['x']
                    or prev_state.position['y'] != curr_state.position['y']
                    or prev_state.position['z'] != curr_state.position['z']):
                changes.append((curr_ts, 'position', prev_state.position,
                                curr_state.position))

            # Check for containment changes
            if prev_state.parent_receptacles != curr_state.parent_receptacles:
                changes.append((curr_ts, 'parentReceptacles',
                                prev_state.parent_receptacles,
                                curr_state.parent_receptacles))

        return changes

    def get_agent_state_at_step(self, timestep: int) -> Optional[AgentState]:
        """Get agent state at a specific timestep."""
        return self.agent_states.get(timestep)

    def get_agent_path(self) -> Dict[int, Dict[str, float]]:
        """Get agent position history across all timesteps."""
        path = {}
        for ts, state in self.agent_states.items():
            path[ts] = state.position
        return path

    def was_object_visible_at_step(self, object_id: str,
                                   timestep: int) -> bool:
        """Check if an object was visible at a specific timestep."""
        obs = self.observations.get(timestep)
        return obs is not None and object_id in obs.visible_objects

    def was_object_nearby_at_step(self, object_id: str, timestep: int) -> bool:
        """Check if an object was nearby at a specific timestep."""
        obs = self.observations.get(timestep)
        return obs is not None and object_id in obs.nearby_objects

    def get_first_seen_timestep(self, object_id: str) -> Optional[int]:
        """Get the first timestep when an object was seen."""
        for ts in sorted(self.observations.keys()):
            if object_id in self.observations[ts].historically_seen_objects:
                return ts
        return None

    def get_objects_of_type(self, object_type: str) -> List[str]:
        """Get all object IDs of a specific type."""
        return [
            obj_id for obj_id, obj_type in self.object_types.items()
            if obj_type == object_type
        ]

    def get_objects_in_receptacle(self, receptacle_id: str,
                                  timestep: int) -> List[str]:
        """Get all objects inside a receptacle at a specific timestep."""
        objects_in = []
        for obj_id in self.all_object_ids:
            state = self.get_object_state_at_step(obj_id, timestep)
            if state and receptacle_id in state.parent_receptacles:
                objects_in.append(obj_id)
        return objects_in

    def get_contained_objects(self, object_id: str,
                              timestep: int) -> List[str]:
        """Get all objects contained by a receptacle at a specific timestep."""
        state = self.get_object_state_at_step(object_id, timestep)
        return state.receptacle_object_ids if state else []

    def get_action_at_step(self, timestep: int) -> Optional[ActionRecord]:
        """Get action record at a specific timestep."""
        return self.actions.get(timestep)

    def find_object_pickup_steps(self, object_id: str) -> List[int]:
        """Find timesteps when an object was picked up."""
        pickup_steps = []

        for ts in sorted(self.agent_states.keys()):
            agent_state = self.agent_states[ts]
            # Check if object is in inventory
            for inv_obj in agent_state.inventory_objects:
                if inv_obj['objectId'] == object_id:
                    pickup_steps.append(ts)
                    break

        return pickup_steps

    def find_object_drop_steps(self, object_id: str) -> List[int]:
        """Find timesteps when an object was dropped."""
        drop_steps = []

        sorted_timesteps = sorted(self.agent_states.keys())

        for i in range(1, len(sorted_timesteps)):
            prev_ts = sorted_timesteps[i - 1]
            curr_ts = sorted_timesteps[i]

            prev_inv = self.agent_states[prev_ts].inventory_objects
            curr_inv = self.agent_states[curr_ts].inventory_objects

            # Check if object was in previous inventory but not current
            was_in_prev = any(obj['objectId'] == object_id for obj in prev_inv)
            not_in_curr = not any(obj['objectId'] == object_id
                                  for obj in curr_inv)

            if was_in_prev and not_in_curr:
                drop_steps.append(curr_ts)

        return drop_steps

    def get_objects_with_state_change(
            self, state_type: str) -> Dict[str, List[Tuple[int, Any, Any]]]:
        """
        Get all objects that had a specific state change.

        Args:
            state_type: Type of state change ('isOpen', 'isToggledOn', etc.)

        Returns:
            Dict mapping object_id to list of (timestep, old_value, new_value)
        """
        objects_with_changes = defaultdict(list)

        for obj_id in self.object_history.keys():
            changes = self.get_object_state_changes(obj_id)
            for change in changes:
                if change[1] == state_type:
                    objects_with_changes[obj_id].append(
                        (change[0], change[2], change[3]))

        return dict(objects_with_changes)

    def was_object_ever_held(self, object_id: str) -> bool:
        """Check if an object was ever in the agent's inventory."""
        for agent_state in self.agent_states.values():
            for inv_obj in agent_state.inventory_objects:
                if inv_obj['objectId'] == object_id:
                    return True
        return False

    def get_first_hold_step(self, object_id: str) -> Optional[int]:
        """Get the first timestep when an object appeared in agent inventory."""
        for ts in sorted(self.agent_states.keys()):
            for inv_obj in self.agent_states[ts].inventory_objects:
                if inv_obj['objectId'] == object_id:
                    return ts
        return None

    def get_hold_ordinal(self, object_id: str) -> Optional[int]:
        """
        Get 1-based ordinal of this object among same-type objects held by agent.

        For example, if agent held Apple|1 at step 5 and Apple|2 at step 10,
        Apple|1 has ordinal 1 and Apple|2 has ordinal 2.

        Returns None if the object was never held.
        """
        if not self.was_object_ever_held(object_id):
            return None

        obj_type = self.object_types.get(object_id)
        if obj_type is None:
            return None

        # Find all same-type objects that were held
        same_type_held = []
        for oid in self.all_object_ids:
            if self.object_types.get(oid) == obj_type:
                first_hold = self.get_first_hold_step(oid)
                if first_hold is not None:
                    same_type_held.append((oid, first_hold))

        # Sort by first hold time to determine order
        same_type_held.sort(key=lambda x: x[1])

        for i, (oid, _) in enumerate(same_type_held):
            if oid == object_id:
                return i + 1

        return None

    def is_unique_object_type(self, object_id: str) -> bool:
        """Check if this is the only object of its type in the scene."""
        obj_type = self.object_types.get(object_id)
        if obj_type is None:
            return False
        return sum(1 for t in self.object_types.values()
                   if t == obj_type) == 1

    # ==================== SERIALIZATION ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tracker data to dictionary for serialization.

        Returns:
            Dictionary containing all tracked data
        """
        return {
            'metadata': {
                'episodeId': self.episode_id,
                'sceneName': self.scene_name,
                'taskInstruction': self.task_instruction,
                'numTimesteps': len(self.timesteps),
                'numObjects': len(self.all_object_ids)
            },
            'objects': {
                obj_id: {
                    'objectType': self.object_types[obj_id],
                    'history': {
                        str(ts): state.to_dict()
                        for ts, state in sorted(states.items())
                    }
                }
                for obj_id, states in self.object_history.items()
            },
            'agent': {
                str(ts): state.to_dict()
                for ts, state in sorted(self.agent_states.items())
            },
            'observations': {
                str(ts): obs.to_dict()
                for ts, obs in sorted(self.observations.items())
            },
            'actions': {
                str(ts): action.to_dict()
                for ts, action in sorted(self.actions.items())
            },
            'summary': {
                'allObjectIds': list(self.all_object_ids),
                'objectTypes': dict(self.object_types),
                'timesteps': self.timesteps
            }
        }

    def save_to_file(self, filepath: str):
        """
        Save tracker data to JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        data = self.to_dict()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ThorTracker':
        """
        Load tracker data from JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            ThorTracker instance with loaded data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tracker = cls()

        # Load metadata
        metadata = data['metadata']
        tracker.episode_id = metadata['episodeId']
        tracker.scene_name = metadata['sceneName']
        tracker.task_instruction = metadata['taskInstruction']

        # Load objects
        for obj_id, obj_data in data['objects'].items():
            tracker.object_types[obj_id] = obj_data['objectType']
            tracker.all_object_ids.add(obj_id)

            for ts_str, state_dict in obj_data['history'].items():
                ts = int(ts_str)
                state = ObjectState(
                    object_id=state_dict['objectId'],
                    object_type=state_dict['objectType'],
                    position=state_dict['position'],
                    rotation=state_dict['rotation'],
                    pickupable=state_dict['pickupable'],
                    receptacle=state_dict['receptacle'],
                    openable=state_dict['openable'],
                    toggleable=state_dict['toggleable'],
                    sliceable=state_dict['sliceable'],
                    dirtyable=state_dict['dirtyable'],
                    is_open=state_dict['isOpen'],
                    is_toggled_on=state_dict['isToggledOn'],
                    is_sliced=state_dict['isSliced'],
                    is_dirty=state_dict['isDirty'],
                    is_visible=state_dict['isVisible'],
                    distance=state_dict['distance'],
                    is_heated=state_dict['isHeated'],
                    is_cooled=state_dict['isCooled'],
                    is_cleaned=state_dict['isCleaned'],
                    parent_receptacles=state_dict['parentReceptacles'],
                    receptacle_object_ids=state_dict['receptacleObjectIds'],
                    loc_desc=state_dict['locDesc'])
                tracker.object_history[obj_id][ts] = state

        # Load agent states
        for ts_str, state_dict in data['agent'].items():
            ts = int(ts_str)
            state = AgentState(
                timestep=state_dict['timestep'],
                position=state_dict['position'],
                rotation=state_dict['rotation'],
                camera_horizon=state_dict['cameraHorizon'],
                facing_direction=state_dict['facingDirection'],
                facing_angle=state_dict['facingAngle'],
                inventory_objects=state_dict['inventoryObjects'])
            tracker.agent_states[ts] = state

        # Load observations
        for ts_str, obs_dict in data['observations'].items():
            ts = int(ts_str)
            obs = ObservationSnapshot(
                timestep=obs_dict['timestep'],
                visible_objects=set(obs_dict['visibleObjects']),
                nearby_objects=set(obs_dict['nearbyObjects']),
                historically_seen_objects=set(
                    obs_dict['historicallySeenObjects']))
            tracker.observations[ts] = obs

        # Load actions
        for ts_str, action_dict in data['actions'].items():
            ts = int(ts_str)
            action = ActionRecord(
                timestep=action_dict['timestep'],
                available_actions=action_dict['availableActions'],
                executed_action=action_dict['executedAction'],
                action_success=action_dict['actionSuccess'],
                env_feedback=action_dict['envFeedback'],
                reasoning=action_dict.get('reasoning', ''),
                resolved_target=action_dict.get('resolvedTarget'))
            tracker.actions[ts] = action

        # Load summary
        tracker.timesteps = data['summary']['timesteps']
        tracker.current_timestep = max(
            tracker.timesteps) if tracker.timesteps else 0
        tracker.historically_seen_objects = set(tracker.all_object_ids)

        return tracker
