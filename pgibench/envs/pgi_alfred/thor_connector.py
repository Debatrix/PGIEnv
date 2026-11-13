import dis
import os, math, re
import textwrap
import json

import numpy as np
from scipy import spatial
from PIL import Image, ImageDraw, ImageFont
import logging

from pgibench.envs.pgi_alfred.env.thor_env import ThorEnv
from pgibench.envs.pgi_alfred.gen import constants
from pgibench.envs.pgi_alfred.gen.utils.game_util import get_objects_with_name_and_prop
from pgibench.envs.pgi_alfred.utils import natural_word_to_ithor_name

log = logging.getLogger(__name__)

log.setLevel(level=logging.ERROR)


class ThorConnector(ThorEnv):

    def __init__(self,
                 x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH,
                 visual_prompt=False):
        super().__init__(x_display, player_screen_height, player_screen_width,
                         quality, build_path, visual_prompt)
        self.font = ImageFont.truetype(
            os.path.join(os.path.dirname(__file__),
                         'media/DejaVuSansMono.ttf'), 24)
        self.agent_height = 0.9
        self.cur_receptacle = None
        self.reachable_positions, self.reachable_position_kdtree = None, None
        self.path_points, self.path_points_kdtree = None, None
        self.path_points_keys = None  # List of keys to maintain order for KD-tree index mapping
        # When False, predefined path points are NOT loaded; the unified list
        # starts empty and is filled only via SaveWaypoint (created_waypoint).
        self.load_predefined_path_points = True
        self.sliced = False
        self.task = None
        self.put_count_dict = {}
        self.proximity_distance = constants.PROXIMITY_DETECTION_DISTANCE
        self.best_positions_cache = {}

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        # print(object_poses)
        super().restore_scene(object_poses, object_toggles, dirty_and_empty)
        self.reachable_positions, self.reachable_position_kdtree = self.get_reachable_positions(
        )
        self.path_points, self.path_points_kdtree = self.load_path_points()

        self.cur_receptacle = None

    def load_path_points(self):
        """
        Load predefined path points from the layouts directory and build KD-tree.

        Predefined points are only loaded when ``self.load_predefined_path_points``
        is True (set by the env layer based on nav_mode). When False the unified
        path-point list starts empty and is populated solely by SaveWaypoint;
        this keeps predefined and user-created points in one structure that can
        be consumed uniformly.

        Returns:
            tuple: (path_points_dict, kdtree) or (None, None) if not loaded
                   path_points_dict: {loc_desc: {'x': x, 'y': y, 'z': z, 'rotation': rotation, 'horizon': horizon}}
        """
        # default_waypoint not active: start with an empty unified list so the
        # agent must create its own waypoints via SaveWaypoint.
        if not getattr(self, 'load_predefined_path_points', True):
            self.path_points_keys = None
            return None, None

        # Extract base scene name (remove any suffix like '-train')
        base_scene_name = self.last_event.metadata['sceneName'].split(
            '_')[0].split('-')[0]

        # Build path to the path-points.json file
        layouts_dir = os.path.join(os.path.dirname(__file__), "gen/layouts")
        path_points_file = os.path.join(layouts_dir,
                                        f"{base_scene_name}-path-points.json")

        if not os.path.exists(path_points_file):
            log.warning(f"Path points file not found: {path_points_file}")
            return None, None

        try:
            with open(path_points_file, 'r') as f:
                data = json.load(f)

            # Extract path points array from the 'path_points' key
            path_points_list = data.get('path_points', [])

            if not path_points_list:
                log.warning(f"No path points found in {path_points_file}")
                return None, None

            self.path_points_keys, path_points_dict = [], {}
            for idx, pt in enumerate(path_points_list):
                loc_desc = f'waypoint_#{idx:02d}'
                self.path_points_keys.append(loc_desc)
                path_points_dict[loc_desc] = {
                    'x': pt['x'],
                    'y': pt['y'],
                    'z': pt['z'],
                    'rotation': pt['rotation'],
                    'horizon': pt['horizon']
                }

            kd_tree = self._build_path_points_kdtree()

            log.info(
                f"Loaded {len(path_points_dict)} path points for scene {base_scene_name}"
            )

            return path_points_dict, kd_tree

        except Exception as e:
            log.error(
                f"Error loading path points from {path_points_file}: {e}")
            return None, None

    def _build_path_points_kdtree(self):
        """Rebuild the KD-tree over the current ``path_points``/``path_points_keys``.

        Called both after loading predefined points and after SaveWaypoint
        appends a new point, so nearest-neighbour queries stay consistent with
        the unified (predefined + user-created) list.
        """
        if not self.path_points_keys or self.path_points is None:
            return None
        pts = [self.path_points[k] for k in self.path_points_keys]
        arr = np.array([[p['x'], p['y'], p['z']] for p in pts])
        return spatial.KDTree(arr)

    def find_close_path_point(self, loc, k=1):
        """
        Find the k closest path points to the given location.

        Args:
            loc: Query location [x, y, z]
            k: Number of closest points to return (default: 1)

        Returns:
            dict or list: If k=1, returns a single path point dict with loc_desc as key.
                          If k>1, returns list of path point dicts.
                          Path point dict: {'x': x, 'y': y, 'z': z, 'rotation': rotation, 'horizon': horizon}
        """
        if self.path_points_kdtree is None:
            log.warning("Path points KD-tree not initialized. "
                        "Call load_path_points() first.")
            return None

        distances, indices = self.path_points_kdtree.query(loc, k=k)

        # Map indices back to dictionary keys
        if k == 1:
            loc_desc = self.path_points_keys[indices]
            return self.path_points[loc_desc]
        else:
            return [{
                self.path_points_keys[idx]:
                self.path_points[self.path_points_keys[idx]]
            } for idx in indices]

    def get_reachable_positions(self):
        free_positions = super().step(
            dict(action="GetReachablePositions")).metadata["actionReturn"]
        free_positions = np.array([[p['x'], p['y'], p['z']]
                                   for p in free_positions])
        kd_tree = spatial.KDTree(free_positions)
        return free_positions, kd_tree

    def write_step_on_img(self, cfg, idx, description):
        img = Image.fromarray(self.last_event.frame)
        text = str(idx) + ':' + description['action']
        lines = textwrap.wrap(text, width=20)
        y_text = 6
        draw = ImageDraw.Draw(img)
        for line in lines:
            width, height = self.font.getsize(line)
            draw.text((6, y_text), line, font=self.font, fill=(255, 255, 255))
            y_text += height
        if cfg is True:
            if not description['success']:
                text_msg = 'error : ' + description['message']
                lines = textwrap.wrap(text_msg, width=20)
                for line in lines:
                    width, height = self.font.getsize(line)
                    draw.text((6, y_text + 6),
                              line,
                              font=self.font,
                              fill=(255, 0, 0))
                    y_text += height
        return img

    def find_close_reachable_position(self, loc, nth=1):
        d, i = self.reachable_position_kdtree.query(loc, k=nth + 1)
        selected = i[nth - 1]
        return self.reachable_positions[selected]

    def llm_skill_interact(self, instruction: str):
        # Save current position and rotation for error recovery
        agent_pos = self.last_event.metadata['agent']['position']
        agent_rot = self.last_event.metadata['agent']['rotation']
        agent_horizon = self.last_event.metadata['agent']['cameraHorizon']

        if instruction.startswith("put down ") or instruction.startswith(
                "open "):
            pass
        else:
            self.cur_receptacle = None

        # Navigation and view control actions
        if instruction == "LookUp":
            ret = super().step(action=dict(
                action="LookUp")).metadata['errorMessage']
        elif instruction == "LookDown":
            ret = super().step(action=dict(
                action="LookDown")).metadata['errorMessage']
        elif instruction == "RotateRight":
            # ret = self.rotate_right()
            ret = super().step(action=dict(
                action="RotateRight")).metadata['errorMessage']
        elif instruction == "RotateLeft":
            # ret = self.rotate_left()
            ret = super().step(action=dict(
                action="RotateLeft")).metadata['errorMessage']
        elif instruction == "MoveAhead":
            ret = super().step(action=dict(
                action="MoveAhead")).metadata['errorMessage']
        elif instruction == "MoveBack":
            ret = super().step(action=dict(
                action="MoveBack")).metadata['errorMessage']
        elif instruction == "MoveRight":
            ret = super().step(action=dict(
                action="MoveRight")).metadata['errorMessage']
        elif instruction == "MoveLeft":
            ret = super().step(
                dict(action="MoveLeft")).metadata['errorMessage']
        elif instruction.startswith("GoTo"):
            # Parse GoTo (x,y,z) format
            try:
                # Extract coordinates from instruction like "GoTo (1.5,2.0,3.5)"
                coords_str = instruction.split('(')[1].split(')')[0]
                x, y, z = [float(c.strip()) for c in coords_str.split(',')]
                ret = self.go_to(x, y, z)
            except:
                ret = 'Invalid GoTo format. Use: GoTo (x,y,z)'
        elif instruction.startswith("goto "):
            # Navigate to path point
            path_point_loc_desc = instruction[5:].strip(
            )  # Remove 'goto ' prefix
            ret = self.go_to_path_point(path_point_loc_desc)
        elif instruction.startswith("navigate to "):
            # Navigate to object by objectId
            obj_name = instruction.replace('navigate to ', '')
            obj_name = natural_word_to_ithor_name(obj_name)
            ret = self.nav_obj_by_id(obj_name)
        # Object interaction actions
        elif instruction.startswith("pick up "):
            obj_name = instruction.replace('pick up ', '')
            obj_name = natural_word_to_ithor_name(obj_name)
            ret = self.pick_by_id(obj_name)
        elif instruction.startswith("put down "):
            # Check if instruction contains a receptacle ID (format: "put down <recep_id>")
            # Receptacle IDs contain '|' character, e.g., "Bowl|-03.52|+00.00|+02.72"
            if '|' in instruction:
                # Extract receptacle ID directly
                receptacle_id = instruction.replace('put down ', '').strip()
                ret = self.put_by_id(receptacle_id)
            else:
                # Check if instruction specifies a receptacle (new format: "put down <obj> in/on <receptacle>")
                # Try to match with "in" or "on" preposition
                m = re.match(r'put down (.+) (?:in|on) (.+)', instruction)
                if m:
                    obj_type = m.group(1).replace('the ', '')
                    receptacle = m.group(2).replace('the ', '')

                    # Set current receptacle and execute
                    self.cur_receptacle = receptacle

                    if self.cur_receptacle in self.put_count_dict:
                        self.put_count_dict[self.cur_receptacle] += 1
                    else:
                        self.put_count_dict[self.cur_receptacle] = 1

                    obj_name = natural_word_to_ithor_name(receptacle)
                    ret = self.nav_obj_by_id(obj_name)
                    ret = self.put_by_id(obj_name)

                    if len(ret) > 16 and self.put_count_dict[receptacle] >= 3:
                        # if put down failed, then drop the object
                        self.drop()
                        ret += f'. The robot dropped the object instead.'
                        self.last_event.metadata['lastActionSuccess'] = False
                else:
                    # Fallback for generic "put down the object in hand" or old format
                    if self.cur_receptacle is None:
                        ret = self.drop()
                    else:
                        m = re.match(r'put down (.+)', instruction)
                        obj = m.group(1).replace('the ', '')

                        if self.cur_receptacle in self.put_count_dict:
                            self.put_count_dict[self.cur_receptacle] += 1
                        else:
                            self.put_count_dict[self.cur_receptacle] = 1

                        receptacle = self.cur_receptacle
                        obj_name = natural_word_to_ithor_name(receptacle)
                        ret = self.nav_obj_by_id(obj_name)
                        ret = self.put_by_id(obj_name)

                        if len(ret
                               ) > 16 and self.put_count_dict[receptacle] >= 3:
                            # if put down failed, then drop the object
                            self.drop()
                            ret += f'. The robot dropped the object instead.'
                            self.last_event.metadata[
                                'lastActionSuccess'] = False

        elif instruction.startswith("open "):
            obj_name = instruction.replace('open ', '')
            obj_name = natural_word_to_ithor_name(obj_name)
            if obj_name not in self.get_visible_objects():
                return {
                    'action':
                    instruction,
                    'success':
                    False,
                    'message':
                    f'The object is not visible. Please navigate to it before opening.'
                }
            ret = self.nav_obj_by_id(obj_name)
            ret = self.open_by_id(obj_name)
        elif instruction.startswith("close "):
            obj_name = instruction.replace('close ', '')
            obj_name = natural_word_to_ithor_name(obj_name)
            if obj_name not in self.get_visible_objects():
                return {
                    'action':
                    instruction,
                    'success':
                    False,
                    'message':
                    f'The object is not visible. Please navigate to it before closeing.'
                }
            ret = self.nav_obj_by_id(obj_name)
            ret = self.close_by_id(obj_name)
        elif instruction.startswith("turn on "):
            obj_name = instruction.replace('turn on ', '')
            obj_name = natural_word_to_ithor_name(obj_name)
            if obj_name not in self.get_visible_objects():
                return {
                    'action': instruction,
                    'success': False,
                    'message':
                    'The object is not visible. Please navigate to it.'
                }
            ret = self.nav_obj_by_id(obj_name)
            ret = self.toggleon_by_id(obj_name)
        elif instruction.startswith("turn off "):
            obj_name = instruction.replace('turn off ', '')
            obj_name = natural_word_to_ithor_name(obj_name)
            if obj_name not in self.get_visible_objects():
                return {
                    'action': instruction,
                    'success': False,
                    'message':
                    'The object is not visible. Please navigate to it.'
                }
            ret = self.nav_obj_by_id(obj_name)
            ret = self.toggleoff_by_id(obj_name)
        elif instruction.startswith("slice "):
            obj_name = instruction.replace('slice ', '')
            obj_name = natural_word_to_ithor_name(obj_name)
            if obj_name not in self.get_visible_objects():
                return {
                    'action': instruction,
                    'success': False,
                    'message':
                    'The object is not visible. Please navigate to it.'
                }
            ret = self.nav_obj_by_id(obj_name)
            ret = self.slice_by_id(obj_name)
            self.sliced = True
        elif instruction.startswith("drop"):
            ret = self.drop()
        else:
            ret = 'instruction not supported'

        # Error recovery: if action failed, restore original position
        if not self.last_event.metadata['lastActionSuccess']:
            log.warning(f"llm_skill_interact failed")
            log.warning(
                f"errorMessage: {self.last_event.metadata['errorMessage']}")
            log.warning(f"returned msg: {ret}")
            log.warning("Attempting to restore original position...")

            # Restore original position
            super().step(
                dict(action="TeleportFull",
                     x=agent_pos['x'],
                     y=agent_pos['y'],
                     z=agent_pos['z'],
                     rotation=agent_rot['y'],
                     horizon=agent_horizon,
                     rotateOnTeleport=True))

            if not self.last_event.metadata['lastActionSuccess']:
                log.error("Failed to restore original position!")
            else:
                log.info("Successfully restored original position")
        else:
            log.info(f"Last action succeeded")

        ret_dict = {
            'action': instruction,
            'success': len(ret) <= 0,
            'message': ret
        }

        return ret_dict

    def get_object_prop(self, name, prop, metadata):
        for obj in metadata['objects']:
            if name in obj['objectId']:
                return obj[prop]
        return None

    @staticmethod
    def angle_diff(x, y):
        x = math.radians(x)
        y = math.radians(y)
        return math.degrees(math.atan2(math.sin(x - y), math.cos(x - y)))

    def nav_obj(self, target_obj: str, prefer_sliced=False):
        objects = self.last_event.metadata['objects']
        action_name = 'object navigation'
        ret_msg = ''
        print(f'{action_name} ({target_obj})')

        # get the object location
        if '|' in target_obj:
            obj_id = target_obj
            target_obj = target_obj.split('|')[0]
            tmp_id, tmp_obj_data = self.get_obj_id_from_name(
                target_obj,
                priority_in_visibility=True,
                priority_sliced=prefer_sliced)
            # if sliced object
            if 'Sliced' in tmp_id and obj_id in tmp_id:
                obj_id = tmp_id
                obj_data = tmp_obj_data
        else:
            obj_id, obj_data = self.get_obj_id_from_name(
                target_obj,
                priority_in_visibility=True,
                priority_sliced=prefer_sliced)

        # find object index from id
        obj_idx = -1
        for i, o in enumerate(objects):
            if o['objectId'] == obj_id:
                obj_idx = i
                break
        if obj_idx == -1:
            ret_msg = f'Cannot find {target_obj}. This object may not exist in this scene. Try to explore other instances instead.'
        else:
            # teleport sometimes fails even with reachable positions. if fails, repeat with the next closest reachable positions.
            max_attempts = 20
            teleport_success = False

            # get obj location
            loc = objects[obj_idx]['position']
            obj_rot = objects[obj_idx]['rotation']['y']

            # # do not move if the object is already visible and close
            # if objects[obj_idx]['visible'] and objects[obj_idx]['distance'] < 1.0:
            #     log.info('Object is already visible')
            #     max_attempts = 0
            #     teleport_success = True

            # try teleporting
            reachable_pos_idx = 0
            for i in range(max_attempts):
                reachable_pos_idx += 1
                if i == 10 and (target_obj == 'Fridge'
                                or target_obj == 'Microwave'):
                    reachable_pos_idx -= 10

                closest_loc = self.find_close_reachable_position(
                    [loc['x'], loc['y'], loc['z']], reachable_pos_idx)
                # calculate desired rotation angle (see https://github.com/allenai/ai2thor/issues/806)
                rot_angle = math.atan2(-(loc['x'] - closest_loc[0]),
                                       loc['z'] - closest_loc[2])
                if rot_angle > 0:
                    rot_angle -= 2 * math.pi
                rot_angle = -(180 / math.pi) * rot_angle  # in degrees

                if i < 10 and (
                        target_obj == 'Fridge' or target_obj == 'Microwave'
                ):  # not always correct, but better than nothing
                    angle_diff = abs(self.angle_diff(rot_angle, obj_rot))
                    if target_obj == 'Fridge' and \
                            not ((90 - 20 < angle_diff < 90 + 20) or (270 - 20 < angle_diff < 270 + 20)):
                        continue
                    if target_obj == 'Microwave' and \
                            not ((180 - 20 < angle_diff < 180 + 20) or (0 - 20 < angle_diff < 0 + 20)):
                        continue

                # calculate desired horizon angle
                camera_height = self.agent_height + constants.CAMERA_HEIGHT_OFFSET
                xz_dist = math.hypot(loc['x'] - closest_loc[0],
                                     loc['z'] - closest_loc[2])
                hor_angle = math.atan2((loc['y'] - camera_height), xz_dist)
                hor_angle = (180 / math.pi) * hor_angle  # in degrees
                hor_angle *= 0.9  # adjust angle for better view
                # hor_angle = -30
                # hor_angle = 0

                # teleport ### Full
                super().step(
                    dict(action="TeleportFull",
                         x=closest_loc[0],
                         y=self.agent_height,
                         z=closest_loc[2],
                         rotation=rot_angle,
                         horizon=-hor_angle))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.warning(
                        f"TeleportFull action failed: {self.last_event.metadata['errorMessage']}, trying again..."
                    )
                else:
                    teleport_success = True
                    break

            if not teleport_success:
                ret_msg = f'Cannot move to {target_obj}'

        return ret_msg

    def get_obj_id_from_name(self,
                             obj_name,
                             only_pickupable=False,
                             only_toggleable=False,
                             priority_sliced=False,
                             get_inherited=False,
                             parent_receptacle_penalty=True,
                             priority_in_visibility=False,
                             exclude_obj_id=None):
        obj_id = None
        obj_data = None
        min_distance = 1e+8

        if any(i.isdigit() for i in obj_name):
            for i in range(len(self.last_event.metadata['objects'])):
                if obj_name in self.last_event.metadata['objects'][i]['name']:
                    obj_id = self.last_event.metadata['objects'][i]['objectId']
                    obj_data = self.last_event.metadata['objects'][i]
                    break
            return obj_id, obj_data
        for obj in self.last_event.metadata['objects']:
            if obj['objectId'] == exclude_obj_id:
                continue

            if (only_pickupable is False or obj['pickupable']) and \
                    (only_toggleable is False or obj['toggleable']) and \
                    obj['objectId'].split('|')[0].casefold() == obj_name.casefold() and \
                    (get_inherited is False or len(obj['objectId'].split('|')) == 5):

                if obj["distance"] < min_distance:
                    penalty_advantage = 0  # low priority for objects in closable receptacles such as fridge, microwave
                    if parent_receptacle_penalty and obj['parentReceptacles']:
                        for p in obj['parentReceptacles']:
                            is_open = self.get_object_prop(
                                p, 'isOpen', self.last_event.metadata)
                            openable = self.get_object_prop(
                                p, 'openable', self.last_event.metadata)
                            if openable is True and is_open is False:
                                penalty_advantage += 100000
                                break

                    if obj_name.casefold() == 'stoveburner':
                        # try to find an empty stove
                        if len(obj['receptacleObjectIds']) > 0:
                            penalty_advantage += 10000

                    if priority_in_visibility and obj['visible'] is False:
                        penalty_advantage += 1000

                    if priority_sliced and '_Slice' in obj['name']:
                        penalty_advantage += -100  # prefer sliced objects; this prevents picking up non-sliced objects

                    if obj["distance"] + penalty_advantage < min_distance:
                        min_distance = obj["distance"] + penalty_advantage
                        obj_data = obj
                        obj_id = obj["objectId"]

        return obj_id, obj_data

    def get_nearby_objects(self):
        """
        Get objects within proximity distance of the agent.

        Returns:
            list: List of object types within proximity distance
        """
        nearby_objects = [
            obj['objectId'] for obj in self.last_event.metadata['objects']
            if obj['distance'] <= self.proximity_distance
        ]

        return nearby_objects

    def get_visible_objects(self,
                            use_inst_seg=True,
                            min_pixel_ratio=0.000005,
                            margin=1):
        """
        Get visible objects based on instance segmentation frame.
        Counts pixels for each object and only returns objects with pixels above threshold.

        Args:
            use_inst_seg: Whether to use instance segmentation (default: True)
            min_pixel_ratio: Minimum pixel ratio threshold
                           Only objects with pixel count ratio >= this threshold are considered visible

        Returns:
            list: List of visible object IDs
        """
        visible_objects = []
        if hasattr(self.last_event,
                   'instance_segmentation_frame') and use_inst_seg:
            seg_frame = np.array(self.last_event.instance_segmentation_frame)
            seg_frame = seg_frame[margin:-margin, margin:-margin, :]

            h, w = seg_frame.shape[:2]
            total_pixels = h * w
            min_pixel_threshold = max(5, int(total_pixels * min_pixel_ratio))

            # Get all valid object IDs from metadata
            valid_object_ids = {
                obj['objectId']
                for obj in self.last_event.metadata['objects']
            }

            # Build reverse mapping: color -> object_id, but only for valid objects
            color_to_valid_obj_id = {
                color: obj_id
                for color, obj_id in
                self.last_event.color_to_object_id.items()
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

            # Filter objects by pixel count threshold
            for obj_id, pixel_count in object_pixel_counts.items():
                if pixel_count >= min_pixel_threshold:
                    visible_objects.append(obj_id)
        else:
            for obj in self.last_event.metadata['objects']:
                if obj['visible']:
                    visible_objects.append(obj['objectId'])

        return visible_objects

    def nav_obj_by_id(self, target_obj_id: str):
        """
        Navigate to a specific object by its objectId.

        Args:
            target_obj_id: The exact objectId to navigate to

        Returns:
            str: Empty string if successful, error message if failed
        """
        objects = self.last_event.metadata['objects']
        ret_msg = ''

        # Find the target object by objectId
        target_obj = None
        for obj in objects:
            if obj['objectId'] == target_obj_id:
                target_obj = obj
                break

        if target_obj is None:
            ret_msg = f'Cannot find object with ID {target_obj_id}. This object may not exist in this scene.'
            print(ret_msg)
            return ret_msg

        # Teleport to the best interaction position first
        ret_msg = self.teleport_to_best_interaction_position(target_obj_id)
        if len(ret_msg) == 0:
            return ret_msg

        # Extract object location and rotation
        loc = target_obj['position']
        obj_rot = target_obj['rotation']['y']

        # Teleport to the object with multiple attempts
        max_attempts = 20
        teleport_success = False
        reachable_pos_idx = 0

        for i in range(max_attempts):
            reachable_pos_idx += 1

            # Get closest reachable position
            closest_loc = self.find_close_reachable_position(
                [loc['x'], loc['y'], loc['z']], reachable_pos_idx)

            # Calculate desired rotation angle
            rot_angle = math.atan2(-(loc['x'] - closest_loc[0]),
                                   loc['z'] - closest_loc[2])
            if rot_angle > 0:
                rot_angle -= 2 * math.pi
            rot_angle = -(180 / math.pi) * rot_angle  # in degrees

            # Calculate desired horizon angle
            camera_height = self.agent_height + constants.CAMERA_HEIGHT_OFFSET
            xz_dist = math.hypot(loc['x'] - closest_loc[0],
                                 loc['z'] - closest_loc[2])
            hor_angle = math.atan2((loc['y'] - camera_height), xz_dist)
            hor_angle = (180 / math.pi) * hor_angle  # in degrees
            hor_angle *= 0.9  # adjust angle for better view

            # Execute teleport
            super().step(
                dict(action="TeleportFull",
                     x=closest_loc[0],
                     y=self.agent_height,
                     z=closest_loc[2],
                     rotation=rot_angle,
                     horizon=-hor_angle))

            if self.last_event.metadata['agent']['position']['y'] < 0:
                log.warning(
                    "Agent fell below ground level after teleportation. Retrying..."
                )
                for _y in [closest_loc[1], 0.9, 0.8, 1, 0.7, 1.1]:
                    super().step(
                        dict(action="TeleportFull",
                             x=closest_loc[0],
                             y=_y,
                             z=closest_loc[2],
                             rotation=rot_angle,
                             horizon=-hor_angle))
                    if self.last_event.metadata['agent']['position']['y'] >= 0:
                        teleport_success = True
                        break
                if not teleport_success:
                    return "Agent fell below ground level after teleportation."

            if self.last_event.metadata['lastActionSuccess']:
                teleport_success = True
                break
            else:
                log.warning(
                    f"TeleportFull action failed: {self.last_event.metadata['errorMessage']}, trying again..."
                )

        if teleport_success:
            ret_msg = ''
        else:
            ret_msg = f'Cannot move to object {target_obj_id}'

        return ret_msg

    def load_best_positions(self, scene_name):
        """Load best interaction positions for the given scene."""
        if scene_name in self.best_positions_cache:
            return self.best_positions_cache[scene_name]

        # Extract scene number from scene name (e.g., "FloorPlan1" -> 1)
        scene_num = scene_name.replace("FloorPlan", "")
        layout_file = os.path.join(os.path.dirname(__file__), "gen/layouts",
                                   f"FloorPlan{scene_num}-openable.json")

        best_positions = {}
        if os.path.exists(layout_file):
            with open(layout_file, 'r') as f:
                best_positions = json.load(f)

        self.best_positions_cache[scene_name] = best_positions
        return best_positions

    def teleport_to_best_interaction_position(self, object_id):
        """
        Teleport to the best interaction position for the given object.

        Args:
            object_id: The exact objectId of the object to interact with

        Returns:
            str: Empty string if successful, error message if failed
        """
        # Get current scene name from last event
        scene_name = self.last_event.metadata['sceneName'].split('_')[0]

        # Load best positions for this scene
        best_positions = self.load_best_positions(scene_name)

        if object_id not in best_positions:
            return f"No best interaction position found for object {object_id}"

        # Get the best position: [x, z, rotation, horizon]
        position_data = best_positions[object_id]
        x, z, rotation, horizon = position_data

        # Get agent height
        agent_height = self.agent_height

        # Execute teleport to the best position
        super().step(
            dict(action="TeleportFull",
                 x=x,
                 y=agent_height,
                 z=z,
                 rotation=rotation,
                 horizon=horizon,
                 rotateOnTeleport=True))

        if self.last_event.metadata['agent']['position']['y'] < 0:
            log.warning(
                "Agent fell below ground level after teleportation. Retrying..."
            )
            for _y in [0.9, 0.8, 1, 0.7, 1.1]:
                super().step(
                    dict(action="TeleportFull",
                         x=x,
                         y=_y,
                         z=z,
                         rotation=rotation,
                         horizon=horizon,
                         rotateOnTeleport=True))
                if self.last_event.metadata['agent']['position']['y'] >= 0:
                    teleport_success = True
                    break
            if not teleport_success:
                return "Agent fell below ground level after teleportation."

        if not self.last_event.metadata['lastActionSuccess']:
            return f"Failed to teleport to best interaction position for {object_id}: {self.last_event.metadata['errorMessage']}"

        return ""

    def pick(self, obj_name):
        obj_id, obj_data = self.get_obj_id_from_name(
            obj_name,
            only_pickupable=True,
            priority_in_visibility=True,
            priority_sliced=self.sliced)

        ret_msg = ''
        log.info(f'pick {obj_id}')

        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to pick up. Find the object before picking up it'
        else:
            if obj_data['visible'] is False and obj_data[
                    'parentReceptacles'] is not None and len(
                        obj_data['parentReceptacles']) > 0:
                # recep_name = obj_data["parentReceptacles"][0].split('|')[0]
                recep_name = obj_data["parentReceptacles"][0]
                ret_msg = f'{obj_name} is not visible because it is in {recep_name}. Note: multiple instances of {recep_name} may exist'

                # try anyway
                super().step(
                    dict(action="PickupObject",
                         objectId=obj_id,
                         forceAction=False))
            else:
                super().step(
                    dict(action="PickupObject",
                         objectId=obj_id,
                         forceAction=False))

                if not self.last_event.metadata['lastActionSuccess']:
                    if len(self.last_event.metadata['inventoryObjects']) == 0:
                        ret_msg = f'Robot is not holding any object'
                    else:
                        # check if the agent is holding the object
                        holding_obj_id = self.last_event.metadata[
                            'inventoryObjects'][0]['objectId']
                        holding_obj_type = self.last_event.metadata[
                            'inventoryObjects'][0]['objectType']
                        ret_msg = f'Robot is currently holding {holding_obj_type}'

            if self.last_event.metadata['lastActionSuccess']:
                ret_msg = ''

        return ret_msg

    def pick_by_id(self, obj_id):
        """
        Pick up an object by its objectId with multi-layer retry mechanism.

        Args:
            obj_id: The exact objectId of the object to pick up

        Returns:
            str: Empty string if successful, error message if failed
        """
        ret_msg = ''
        log.info(f'pick_by_id {obj_id}')

        # Find the object by objectId
        obj_data = None
        for obj in self.last_event.metadata['objects']:
            if obj['objectId'] == obj_id:
                obj_data = obj
                break

        if obj_data is None:
            return f'Cannot find object with ID {obj_id} to pick up'
        elif not obj_data.get('pickupable', False):
            return f'The {obj_id} is not pickupable'

        # Check if agent is already holding an object
        if len(self.last_event.metadata['inventoryObjects']) > 0:
            holding_obj_type = self.last_event.metadata['inventoryObjects'][0][
                'objectType']
            return f'Robot is already holding {holding_obj_type}. Cannot pick up another object.'

        receptacle_id = None
        if obj_data.get('parentReceptacles') and len(
                obj_data['parentReceptacles']) > 0:
            receptacle_id = obj_data['parentReceptacles'][0]

        if obj_id not in self.get_visible_objects():
            return f"The {obj_data['objectType']} is not visible. Please navigate to the object before picking it up."

        if receptacle_id:
            # Get container object
            receptacle_obj = None
            for obj in self.last_event.metadata['objects']:
                if obj['objectId'] == receptacle_id:
                    receptacle_obj = obj
                    break

            # Check if container is openable and closed
            if (receptacle_obj and receptacle_obj.get('openable', False)
                    and not receptacle_obj.get('isOpen', False)):
                return f"The {obj_data['objectType']} is in closed container {receptacle_obj['objectType']}. Please open the container first."

        original_pos = self.last_event.metadata['agent']['position']
        original_rot = self.last_event.metadata['agent']['rotation']
        original_horizon = self.last_event.metadata['agent']['cameraHorizon']

        try:
            # Layer 1: navigation and basic pickup
            log.info(f'Navigating to object {obj_id}')
            ret_msg = self.nav_obj_by_id(obj_id)
            if len(ret_msg) > 0:
                log.warning(f'Navigation to {obj_id} failed: {ret_msg}')
                return f'Cannot navigate to object {obj_id}: {ret_msg}'

            # Basic pickup attempt
            log.info(f'Attempting basic pickup for {obj_id}')
            super().step(
                dict(action="PickupObject", objectId=obj_id, forceAction=True))

            if self.last_event.metadata['lastActionSuccess']:
                log.info(f'Successfully picked up {obj_id}')
                return ''

            # If the object is inside a container, navigate to the container first and try picking from there
            if receptacle_id:
                log.info(
                    f'Object {obj_id} is in container {receptacle_id}, navigating to container first'
                )
                ret_msg = self.nav_obj_by_id(receptacle_id)
                if len(ret_msg) > 0:
                    log.warning(
                        f'Navigation to container {receptacle_id} failed: {ret_msg}'
                    )
                else:
                    # Navigated to the container successfully; attempt the interaction
                    log.info(
                        f'Attempting pickup after navigating to container {receptacle_id}'
                    )
                    super().step(
                        dict(action="PickupObject",
                             objectId=obj_id,
                             forceAction=True))

                    if self.last_event.metadata['lastActionSuccess']:
                        log.info(
                            f'Successfully picked up {obj_id} from container {receptacle_id}'
                        )
                        return ''  # success
                    else:
                        log.info(
                            f'Pickup after navigating to container failed, will try other strategies'
                        )

            # Layer 2: retry with 6 strategies
            halt = False
            for j in range(6):
                # Execute strategy adjustment
                if j == 1:
                    super().step(dict(action="LookUp"))
                    super().step(dict(action="LookUp"))
                elif j == 2:
                    super().step(dict(action="LookDown"))
                    super().step(dict(action="LookDown"))
                    super().step(dict(action="LookDown"))
                    super().step(dict(action="LookDown"))
                elif j == 3:
                    super().step(dict(action="LookUp"))
                    super().step(dict(action="LookUp"))
                    super().step(dict(action="MoveBack"))
                elif j == 4:
                    super().step(dict(action="MoveAhead"))
                    for r in range(4):
                        super().step(dict(action="MoveRight"))
                elif j == 5:
                    for r in range(8):
                        super().step(dict(action="MoveLeft"))

                # Attempt pickup
                super().step(
                    dict(action="PickupObject",
                         objectId=obj_id,
                         forceAction=True))

                if self.last_event.metadata['lastActionSuccess']:
                    ret_msg = ''
                    halt = True
                    break
                else:
                    log.warning(
                        f"Strategy {j} failed: {self.last_event.metadata['errorMessage']}"
                    )

            if not halt:
                ret_msg = f"Failed to pick up {obj_data['objectType']}:{self.last_event.metadata['errorMessage']}"
                super().step(
                    dict(action="TeleportFull",
                         x=original_pos['x'],
                         y=original_pos['y'],
                         z=original_pos['z'],
                         rotation=original_rot['y'],
                         horizon=original_horizon,
                         rotateOnTeleport=True))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.error("Failed to restore original position!")

            return ret_msg

        finally:
            # On failure, restore the agent position
            if not self.last_event.metadata['lastActionSuccess']:
                super().step(
                    dict(action="TeleportFull",
                         x=original_pos['x'],
                         y=original_pos['y'],
                         z=original_pos['z'],
                         rotation=original_rot['y'],
                         horizon=original_horizon,
                         rotateOnTeleport=True))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.error("Failed to restore original position!")

    def put(self, receptacle_name):
        # assume the agent always put the object currently holding
        ret_msg = ''
        orig_receptacle_name = receptacle_name

        if len(self.last_event.metadata['inventoryObjects']) == 0:
            ret_msg = f'Robot is not holding any object'
            return ret_msg
        else:
            holding_obj_id = self.last_event.metadata['inventoryObjects'][0][
                'objectId']

        halt = False
        last_recep_id = None
        exclude_obj_id = None
        for k in range(2):  # try closest and next closest one
            for j in range(7):  # move/look around or rotate obj
                for i in range(
                        2
                ):  # try inherited receptacles too (e.g., sink basin, bath basin)
                    if k == 1 and exclude_obj_id is None:
                        exclude_obj_id = last_recep_id  # previous recep id

                    # for the second round, find another receptacle
                    if k == 0 and '|' in orig_receptacle_name:
                        if i == 1:
                            continue
                        recep_id = orig_receptacle_name
                        receptacle_name = orig_receptacle_name.split('|')[0]
                    else:
                        if 'Sink' in receptacle_name or 'Bathtub' in receptacle_name:  # sink base
                            if i == 0:
                                recep_id, _ = self.get_obj_id_from_name(
                                    receptacle_name,
                                    get_inherited=True,
                                    exclude_obj_id=exclude_obj_id)
                            else:
                                recep_id, _ = self.get_obj_id_from_name(
                                    receptacle_name,
                                    exclude_obj_id=exclude_obj_id)
                        else:
                            if i == 0:
                                recep_id, _ = self.get_obj_id_from_name(
                                    receptacle_name,
                                    exclude_obj_id=exclude_obj_id)
                            else:
                                recep_id, _ = self.get_obj_id_from_name(
                                    receptacle_name,
                                    get_inherited=True,
                                    exclude_obj_id=exclude_obj_id)

                    if not recep_id:
                        ret_msg = f'Putting the object on {receptacle_name} failed. First check whether the receptacle is open or not. Also try other instances of the receptacle'
                        continue

                    print(f'put {holding_obj_id} on {recep_id}')

                    # look up (put action fails when a receptacle is not visible)
                    if j == 1:
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="LookUp"))
                    elif j == 2:
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                    elif j == 3:
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="MoveBack"))
                    elif j == 4:
                        super().step(dict(action="MoveAhead"))
                        for r in range(4):
                            super().step(dict(action="MoveRight"))
                    elif j == 5:
                        for r in range(8):
                            super().step(dict(action="MoveLeft"))
                    elif j == 6:
                        for r in range(4):
                            super().step(dict(action="MoveRight"))
                        super().step(
                            dict(  # this somehow make putobject success in some cases
                                action="RotateHand",
                                x=40))

                    super().step(
                        dict(action="PutObject",
                             objectId=holding_obj_id,
                             receptacleObjectId=recep_id,
                             forceAction=True))
                    last_recep_id = recep_id

                    if not self.last_event.metadata['lastActionSuccess']:
                        logging.warning(
                            f"PutObject action failed: {self.last_event.metadata['errorMessage']}, trying again..."
                        )
                        ret_msg = f'Putting the object on {receptacle_name} failed. First check the receptacle is open or not. Also try other instances of the receptacle'
                    else:
                        ret_msg = ''
                        halt = True
                        break
                if halt:
                    break
            if halt:
                break

        return ret_msg

    def put_by_id(self, receptacle_id):
        """
        Put object on a receptacle by its objectId.

        Args:
            receptacle_id: The exact objectId of the receptacle

        Returns:
            str: Empty string if successful, error message if failed
        """
        # assume the agent always put the object currently holding
        ret_msg = ''

        if len(self.last_event.metadata['inventoryObjects']) == 0:
            ret_msg = f'Robot is not holding any object'
            return ret_msg
        else:
            holding_obj_id = self.last_event.metadata['inventoryObjects'][0][
                'objectId']

        # Find the receptacle by objectId
        receptacle_obj = None
        for obj in self.last_event.metadata['objects']:
            if obj['objectId'] == receptacle_id:
                receptacle_obj = obj
                break

        if receptacle_obj is None:
            ret_msg = f"Cannot find receptacle with ID {receptacle_id} to put object on"
            return ret_msg

        if receptacle_id not in self.get_visible_objects():
            # Check if container is openable and closed
            if (receptacle_obj and receptacle_obj.get('openable', False)
                    and not receptacle_obj.get('isOpen', False)):
                return f'Receptacle {receptacle_obj["objectType"]} is in closed. Please open the receptacle first.'
            else:
                return f'Receptacle {receptacle_obj["objectType"]} is not visible. Please navigate to the receptacle first.'

        ret_msg = self.nav_obj_by_id(receptacle_id)
        if len(ret_msg) > 0:
            log.warning(
                f'Navigation to container {receptacle_id} failed: {ret_msg}')
        else:
            # Navigated to the container successfully; attempt the interaction
            log.info(
                f'Attempting pickup after navigating to container {receptacle_id}'
            )
            super().step(
                dict(action="PutObject",
                     objectId=holding_obj_id,
                     receptacleObjectId=receptacle_id,
                     forceAction=True))

            if self.last_event.metadata['lastActionSuccess']:
                log.info(
                    f'Successfully put {holding_obj_id} to container {receptacle_id}'
                )
                return ''  # success
            else:
                log.info(
                    f'putdown after navigating to container failed, will try other strategies'
                )

        for j in range(7):  # move/look around or rotate obj
            log.info(f'put {holding_obj_id} on {receptacle_id}')

            # look up (put action fails when a receptacle is not visible)
            if j == 1:
                super().step(dict(action="LookUp"))
                super().step(dict(action="LookUp"))
            elif j == 2:
                super().step(dict(action="LookDown"))
                super().step(dict(action="LookDown"))
                super().step(dict(action="LookDown"))
                super().step(dict(action="LookDown"))
            elif j == 3:
                super().step(dict(action="LookUp"))
                super().step(dict(action="LookUp"))
                super().step(dict(action="MoveBack"))
            elif j == 4:
                super().step(dict(action="MoveAhead"))
                for r in range(4):
                    super().step(dict(action="MoveRight"))
            elif j == 5:
                for r in range(8):
                    super().step(dict(action="MoveLeft"))
            elif j == 6:
                for r in range(4):
                    super().step(dict(action="MoveRight"))
                super().step(
                    dict(  # this somehow make putobject success in some cases
                        action="RotateHand",
                        x=40))

            super().step(
                dict(action="PutObject",
                     objectId=holding_obj_id,
                     receptacleObjectId=receptacle_id,
                     forceAction=True))

            if not self.last_event.metadata['lastActionSuccess']:
                log.warning(
                    f"PutObject action failed: {self.last_event.metadata['errorMessage']}, trying again..."
                )
                ret_msg = f'Putting object on {receptacle_obj["objectType"]} failed. First check whether the receptacle is open or not.'
            else:
                ret_msg = ''
                break

        return ret_msg

    def drop(self):
        log.info(f'drop')
        ret_msg = ''
        super().step(dict(action="DropHandObject", forceAction=True))

        if not self.last_event.metadata['lastActionSuccess']:
            if len(self.last_event.metadata['inventoryObjects']) == 0:
                ret_msg = f'Robot is not holding any object'
            else:
                ret_msg = f"Drop action failed"
        else:
            ret_msg = ''

        return ret_msg

    def get_held_object(self):
        """
        Get the object currently held by the agent.

        Returns:
            tuple: (object_id, object_type) if holding an object, else (None, None)
        """
        if len(self.last_event.metadata['inventoryObjects']) == 0:
            return None, None
        obj = self.last_event.metadata['inventoryObjects'][0]
        return obj['objectId'], obj['objectType']

    def get_valid_receptacles_for_object(self, object_type, visible_only=True):
        """
        Get receptacles that can accept the given object type.

        Args:
            object_type: Type of object being held (e.g., 'Apple', 'AppleSliced')
            visible_only: If True, only include visible/nearby receptacles. If False, include all.

        Returns:
            list: List of tuples (receptacle_id, loc_desc, receptacle_type) for valid visible/nearby receptacles
        """
        from pgibench.envs.pgi_alfred.gen.constants import VAL_RECEPTACLE_OBJECTS

        # Find all receptacle types that can accept this object
        valid_receptacle_types = set()
        for recep_type, valid_objects in VAL_RECEPTACLE_OBJECTS.items():
            if object_type in valid_objects:
                valid_receptacle_types.add(recep_type)

        # If object type not found in VAL_RECEPTACLE_OBJECTS, return empty list
        if not valid_receptacle_types:
            return []

        # Filter visible/nearby objects by valid receptacle types
        valid_receptacles = []

        if visible_only:
            visible_objs = set(self.get_visible_objects())
            nearby_objs = set(self.get_nearby_objects())

        for obj in self.last_event.metadata['objects']:
            if obj['objectType'] in valid_receptacle_types:
                # Check if visible or nearby (if filtering enabled)
                if visible_only:
                    if obj['objectId'] not in (visible_objs | nearby_objs):
                        continue

                loc_desc = obj.get('locDesc', obj['objectType'])
                valid_receptacles.append(
                    (obj['objectId'], loc_desc, obj['objectType']))

        return valid_receptacles

    def open(self, obj_name):
        log.info(f'open {obj_name}')
        ret_msg = ''
        # obj_id, _ = self.get_obj_id_from_name(obj_name)
        # get the object location
        if '|' in obj_name:
            obj_id = obj_name
            obj_name = obj_name.split('|')[0]
        else:
            obj_id, _ = self.get_obj_id_from_name(obj_name)

        if obj_id is None:
            ret_msg = f"Cannot find {obj_name} to open. Find the object before opening it"
        else:
            open_flag = False
            for ob in self.last_event.metadata['objects']:
                if ob['objectId'] == obj_id and ob['openable'] and ob['isOpen']:
                    open_flag = True
                    break

            for i in range(4):
                super().step(dict(
                    action="OpenObject",
                    objectId=obj_id,
                ))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.warning(
                        f"OpenObject action failed: {self.last_event.metadata['errorMessage']}, moving backward and trying again..."
                    )
                    if open_flag:
                        ret_msg = f"Open action failed. The {obj_name} is already open"
                    else:
                        ret_msg = f"Open action failed."

                    # move around to avoid self-collision
                    if i == 0:
                        super().step(dict(action="MoveBack"))
                    elif i == 1:
                        super().step(dict(action="MoveBack"))
                        super().step(dict(action="MoveRight"))
                    elif i == 2:
                        super().step(dict(action="MoveLeft"))
                        super().step(dict(action="MoveLeft"))
                else:
                    ret_msg = ''
                    break

        return ret_msg

    def _sync_state(self, action_name, obj_id):
        """Mirror ThorEnv.step's update_states/check_post_conditions bookkeeping.

        The _by_id primitives call super().step() directly, bypassing ThorEnv.step,
        so toggling a faucet/microwave or closing a fridge would leave the
        heated/cooled/cleaned sets stale (BUG#5). Call this after such actions.
        """
        action = {"action": action_name, "objectId": obj_id}
        self.update_states(action)
        self.check_post_conditions(action)

    def open_by_id(self, obj_id):
        """
        Open an object by its objectId.

        Args:
            obj_id: The exact objectId of the object to open

        Returns:
            str: Empty string if successful, error message if failed
        """
        log.info(f'open_by_id {obj_id}')
        ret_msg = ''

        # Find the object by objectId
        obj = None
        for o in self.last_event.metadata['objects']:
            if o['objectId'] == obj_id:
                obj = o
                break

        if obj is None:
            ret_msg = f"Cannot find object with ID {obj_id} to open"
        elif not obj.get('openable', False):
            ret_msg = f"Object {obj_id} is not openable"
        else:
            if obj_id not in self.get_visible_objects():
                return f'The object is not visible. Please navigate to it before opening.'
            open_flag = obj.get('isOpen', False)

            for i in range(4):
                super().step(dict(
                    action="OpenObject",
                    objectId=obj_id,
                ))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.warning(
                        f"OpenObject action failed: {self.last_event.metadata['errorMessage']}, moving backward and trying again..."
                    )
                    if open_flag:
                        ret_msg = f"Open action failed. The object {obj_id} is already open"
                    else:
                        ret_msg = f"Open action failed: {self.last_event.metadata['errorMessage']}."

                    # move around to avoid self-collision
                    if i == 0:
                        super().step(dict(action="MoveBack"))
                    elif i == 1:
                        super().step(dict(action="MoveBack"))
                        super().step(dict(action="MoveRight"))
                    elif i == 2:
                        super().step(dict(action="MoveLeft"))
                        super().step(dict(action="MoveLeft"))
                else:
                    ret_msg = ''
                    break

        return ret_msg

    def close(self, obj_name):
        log.info(f'close {obj_name}')
        ret_msg = ''
        if '|' in obj_name:
            obj_id = obj_name
            obj_name = obj_name.split('|')[0]
        else:
            obj_id, _ = self.get_obj_id_from_name(obj_name)

        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to close'
        else:
            super().step(dict(
                action="CloseObject",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Close action failed"

                for ob in self.last_event.metadata['objects']:
                    if ob['objectId'] == obj_id and ob[
                            'openable'] and not ob['isOpen']:
                        ret_msg += f". The {obj_name} is already closed"
                        break

        return ret_msg

    def close_by_id(self, obj_id):
        """
        Close an object by its objectId.

        Args:
            obj_id: The exact objectId of the object to close

        Returns:
            str: Empty string if successful, error message if failed
        """
        log.info(f'close_by_id {obj_id}')
        ret_msg = ''

        # Find the object by objectId
        obj = None
        for o in self.last_event.metadata['objects']:
            if o['objectId'] == obj_id:
                obj = o
                break

        if obj is None:
            ret_msg = f'Cannot find object with ID {obj_id} to close'
        elif not obj.get('openable', False):
            ret_msg = f"The {obj['objectType']} is not closeable"
        else:
            super().step(dict(
                action="CloseObject",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Close action failed"

                # Check if the object is already closed
                if not obj.get('isOpen', False):
                    ret_msg += f". The object {obj_id} is already closed"
            self._sync_state("CloseObject", obj_id)

        return ret_msg

    def toggleon(self, obj_name):
        log.info(f'toggle on {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, only_toggleable=True)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to turn on'
        else:
            try:
                super().step(dict(
                    action="ToggleObjectOn",
                    objectId=obj_id,
                ))
                if not self.last_event.metadata['lastActionSuccess']:
                    ret_msg = f"Turn on action failed"
            except:
                ret_msg = f"Turn on action failed"
                self.last_event.metadata['lastActionSuccess'] = False

        return ret_msg

    def toggleon_by_id(self, obj_id):
        """
        Turn on an object by its objectId.

        Args:
            obj_id: The exact objectId of the object to turn on

        Returns:
            str: Empty string if successful, error message if failed
        """
        log.info(f'toggle on by_id {obj_id}')
        ret_msg = ''

        # Find the object by objectId
        obj = None
        for o in self.last_event.metadata['objects']:
            if o['objectId'] == obj_id:
                obj = o
                break

        if obj is None:
            ret_msg = f'Cannot find object with ID {obj_id} to turn on'
        elif not obj.get('toggleable', False):
            ret_msg = f"The {obj['objectType']} is not toggleable"
        else:
            try:
                super().step(
                    dict(action="ToggleObjectOn",
                         objectId=obj_id,
                         forceAction=True))
                if not self.last_event.metadata['lastActionSuccess']:
                    ret_msg = f"Turn on action failed"
            except:
                ret_msg = f"Turn on action failed"
                self.last_event.metadata['lastActionSuccess'] = False
            # sync heated (microwave) / cleaned (faucet) state sets — see _sync_state
            self._sync_state("ToggleObjectOn", obj_id)

        return ret_msg

    def toggleoff(self, obj_name):
        log.info(f'toggle off {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, only_toggleable=True)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to turn off'
        else:
            super().step(dict(
                action="ToggleObjectOff",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Turn off action failed"

        return ret_msg

    def toggleoff_by_id(self, obj_id):
        """
        Turn off an object by its objectId.

        Args:
            obj_id: The exact objectId of the object to turn off

        Returns:
            str: Empty string if successful, error message if failed
        """
        log.info(f'toggle off by_id {obj_id}')
        ret_msg = ''

        # Find the object by objectId
        obj = None
        for o in self.last_event.metadata['objects']:
            if o['objectId'] == obj_id:
                obj = o
                break

        if obj is None:
            ret_msg = f'Cannot find object with ID {obj_id} to turn off'
        elif not obj.get('toggleable', False):
            ret_msg = f"The {obj['objectType']} is not toggleable"
        else:
            super().step(
                dict(action="ToggleObjectOff",
                     objectId=obj_id,
                     forceAction=True))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Turn off action failed"

        return ret_msg

    def slice(self, obj_name):
        log.info(f'slice {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to slice'
        else:
            super().step(dict(
                action="SliceObject",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Slice action failed"

        return ret_msg

    def slice_by_id(self, obj_id):
        """
        Slice an object by its objectId.

        Args:
            obj_id: The exact objectId of the object to slice

        Returns:
            str: Empty string if successful, error message if failed
        """
        log.info(f'slice_by_id {obj_id}')
        ret_msg = ''

        # Find the object by objectId
        obj = None
        for o in self.last_event.metadata['objects']:
            if o['objectId'] == obj_id:
                obj = o
                break

        if obj is None:
            ret_msg = f'Cannot find object with ID {obj_id} to slice'
        elif not obj.get('sliceable', False):
            ret_msg = f"The {obj['objectType']} is not sliceable"
        else:
            super().step(
                dict(action="SliceObject", objectId=obj_id, forceAction=True))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Slice action failed"
            self._sync_state("SliceObject", obj_id)

        return ret_msg

    def go_to(self, x, y, z, rotation=None, horizon=None):
        """
        Teleport to given XYZ coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            rotation: Horizontal rotation angle (optional, auto-calculated if None)
            horizon: Vertical horizon angle (optional, 0 if None)

        Returns:
            str: Empty string if successful, error message if failed
        """
        log.info(f'go_to ({x:.2f}, {y:.2f}, {z:.2f})')

        # Get current position
        current_pos = self.last_event.metadata['agent']['position']

        # Calculate rotation angle if not provided
        if rotation is None:
            # Point from current position to target position
            dx = x - current_pos['x']
            dz = z - current_pos['z']
            rotation = math.degrees(math.atan2(-dx, dz))

        # Set default horizon if not provided
        if horizon is None:
            horizon = 0

        # Execute teleport
        super().step(
            dict(action="TeleportFull",
                 x=x,
                 y=y,
                 z=z,
                 rotation=rotation,
                 horizon=horizon,
                 rotateOnTeleport=True))
        if self.last_event.metadata['agent']['position']['y'] < 0:
            log.warning(
                "Agent fell below ground level after teleportation. Retrying..."
            )
            for _y in [0.9, 0.8, 1, 0.7, 1.1]:
                super().step(
                    dict(action="TeleportFull",
                         x=x,
                         y=_y,
                         z=y,
                         rotation=rotation,
                         horizon=-horizon))
                if self.last_event.metadata['agent']['position']['y'] >= 0:
                    teleport_success = True
                    break
            if not teleport_success:
                return "Agent fell below ground level after teleportation."

        if not self.last_event.metadata['lastActionSuccess']:
            return f"Failed to teleport to ({x:.2f}, {y:.2f}, {z:.2f}): {self.last_event.metadata['errorMessage']}"

        return ""

    def go_to_path_point(self, path_point_loc_desc):
        """
        Navigate to a specific path point by its loc_desc.
        Similar to nav_obj_by_id, tries to teleport to the exact path point location first,
        then attempts nearby reachable positions if the direct teleport fails.

        Args:
            path_point_loc_desc: The loc_desc of the path point to navigate to

        Returns:
            str: Empty string if successful, error message if failed
        """
        log.info(f'go_to_path_point ({path_point_loc_desc})')

        # Check if path points are loaded
        if self.path_points is None or path_point_loc_desc not in self.path_points:
            return f'Path point {path_point_loc_desc} not found'

        # Get path point data
        path_point = self.path_points[path_point_loc_desc]
        x, y, z = path_point['x'], path_point['y'], path_point['z']
        target_rotation = path_point['rotation']
        target_horizon = path_point['horizon']

        ret_msg = ''

        # Try direct teleport first with exact path point position and orientation
        log.info(
            f'Trying direct teleport to path point: ({x:.2f}, {y:.2f}, {z:.2f})'
        )

        super().step(
            dict(action="TeleportFull",
                 x=x,
                 y=self.agent_height,
                 z=z,
                 rotation=target_rotation,
                 horizon=target_horizon,
                 rotateOnTeleport=True))

        _pos = self.last_event.metadata['agent']['position']
        if self.last_event.metadata['lastActionSuccess'] and _pos['y'] > 0:
            log.info(
                f'Successfully teleported to path point {path_point_loc_desc} ({str(_pos)})'
            )
            return ret_msg
        elif self.last_event.metadata['lastActionSuccess'] and _pos['y'] < 0:
            log.warning(
                "Agent fell below ground level after teleportation. Retrying..."
            )
            for _y in [0.9, 0.8, 1, 0.7, 1.1]:
                super().step(
                    dict(action="TeleportFull",
                         x=x,
                         y=_y,
                         z=z,
                         rotation=target_rotation,
                         horizon=-target_horizon))
                if self.last_event.metadata['agent']['position']['y'] >= 0:
                    return ret_msg

        # If direct teleport failed, try nearby reachable positions
        log.warning(
            f'Direct teleport failed, trying nearby reachable positions...')
        max_attempts = 20
        teleport_success = False
        reachable_pos_idx = 0

        for i in range(max_attempts):
            reachable_pos_idx += 1

            # Get closest reachable position
            closest_loc = self.find_close_reachable_position([x, y, z],
                                                             reachable_pos_idx)

            # Use the path point's rotation and horizon
            # Execute teleport
            super().step(
                dict(action="TeleportFull",
                     x=closest_loc[0],
                     y=self.agent_height,
                     z=closest_loc[2],
                     rotation=target_rotation,
                     horizon=target_horizon,
                     rotateOnTeleport=True))

            _pos = self.last_event.metadata['agent']['position']
            if self.last_event.metadata['lastActionSuccess'] and _pos['y'] > 0:
                teleport_success = True
                log.info(
                    f'Successfully teleported to nearby position of path point {path_point_loc_desc}({str(_pos)})'
                )
                break
            elif self.last_event.metadata['lastActionSuccess'] and _pos[
                    'y'] < 0:
                log.warning(
                    "Agent fell below ground level after teleportation. Retrying..."
                )
                for _y in [0.9, 0.8, 1, 0.7, 1.1]:
                    super().step(
                        dict(action="TeleportFull",
                             x=closest_loc[0],
                             y=_y,
                             z=closest_loc[2],
                             rotation=target_rotation,
                             horizon=-target_horizon))
                    _pos = self.last_event.metadata['agent']['position']
                    if _pos['y'] >= 0:
                        teleport_success = True
                        log.info(
                            f'Successfully teleported to nearby position of path point {path_point_loc_desc}({str(_pos)})'
                        )
                        break
                if teleport_success:
                    break
            else:
                log.warning(
                    f"TeleportFull action failed: {self.last_event.metadata['errorMessage']}, trying again..."
                )

        if not teleport_success:
            ret_msg = f'Cannot move to path point {path_point_loc_desc}'

        return ret_msg

    def look_up(self, angle=30):
        """
        Look up by specified angle.

        Args:
            angle: Angle to look up in degrees (default: 30)

        Returns:
            str: Empty string if successful, error message if failed
        """
        log.info(f'look_up({angle})')

        super().step(dict(action="LookUp", degrees=angle))

        if not self.last_event.metadata['lastActionSuccess']:
            return f"look_up failed: {self.last_event.metadata['errorMessage']}"

        return ""

    def look_down(self, angle=30):
        """
        Look down by specified angle.

        Args:
            angle: Angle to look down in degrees (default: 30)

        Returns:
            str: Empty string if successful, error message if failed
        """
        log.info(f'look_down({angle})')

        super().step(dict(action="LookDown", degrees=angle))

        if not self.last_event.metadata['lastActionSuccess']:
            return f"look_down failed: {self.last_event.metadata['errorMessage']}"

        return ""

    def get_agent_position(self):
        """
        Get agent position and facing direction.

        Coordinate system definition:
        - Y-axis: Vertical (up/down)
        - XZ-plane: Horizontal directions
          - +Z direction = North
          - +X direction = East
          - -Z direction = South
          - -X direction = West
        - When rotation['y'] = 0°, agent faces +Z direction (North)
        - Angle increases clockwise

        Returns:
            dict: Contains rotation_y, direction, position, closest_path_point, and is_on_path_point
        """
        agent = self.last_event.metadata['agent']
        rotation_y = agent['rotation']['y'] % 360

        # 0° faces North (+Z), angle increases clockwise
        # (aligned with thor_env.get_agent_facing_direction)
        if 337.5 <= rotation_y or rotation_y < 22.5:
            direction = "north"
        elif 22.5 <= rotation_y < 67.5:
            direction = "northeast"
        elif 67.5 <= rotation_y < 112.5:
            direction = "east"
        elif 112.5 <= rotation_y < 157.5:
            direction = "southeast"
        elif 157.5 <= rotation_y < 202.5:
            direction = "south"
        elif 202.5 <= rotation_y < 247.5:
            direction = "southwest"
        elif 247.5 <= rotation_y < 292.5:
            direction = "west"
        else:  # 292.5 <= rotation_y < 337.5
            direction = "northwest"

        # Find closest path point
        closest_path_point = None
        is_on_path_point = False
        distance_to_closest = None
        relative_direction_to_path_point = None

        if self.path_points_kdtree is not None:
            agent_loc = [
                agent['position']['x'], agent['position']['y'],
                agent['position']['z']
            ]
            distance, index = self.path_points_kdtree.query(agent_loc, k=1)
            closest_path_point = self.path_points_keys[index]
            distance_to_closest = distance
            is_on_path_point = distance < 0.2

            # If not on path point, calculate relative direction from path point to agent
            if not is_on_path_point:
                path_point = self.path_points[closest_path_point]
                dx = agent['position']['x'] - path_point['x']
                dz = agent['position']['z'] - path_point['z']

                # Calculate angle from path point to agent
                # atan2(dz, dx) returns angle where 0°=+X (East), 90°=+Z (North)
                # Convert to North=0° clockwise system (aligned with get_spatial_relation)
                angle_rad = math.atan2(dz, dx)
                angle_deg = math.degrees(angle_rad)
                angle = (90 - angle_deg) % 360

                # Convert angle to 8 cardinal directions (0°=North, clockwise)
                if 337.5 <= angle or angle < 22.5:
                    relative_direction_to_path_point = "north"
                elif 22.5 <= angle < 67.5:
                    relative_direction_to_path_point = "northeast"
                elif 67.5 <= angle < 112.5:
                    relative_direction_to_path_point = "east"
                elif 112.5 <= angle < 157.5:
                    relative_direction_to_path_point = "southeast"
                elif 157.5 <= angle < 202.5:
                    relative_direction_to_path_point = "south"
                elif 202.5 <= angle < 247.5:
                    relative_direction_to_path_point = "southwest"
                elif 247.5 <= angle < 292.5:
                    relative_direction_to_path_point = "west"
                else:  # 292.5 <= angle < 337.5
                    relative_direction_to_path_point = "northwest"

        return {
            'position': agent['position'],
            'horizon': agent['cameraHorizon'],
            'rotation_y': rotation_y,
            'direction': direction,
            'closest_path_point': closest_path_point,
            'is_on_path_point': is_on_path_point,
            'distance_to_closest_path_point': distance_to_closest,
            'relative_direction_to_path_point':
            relative_direction_to_path_point,
        }

    def rotate_right(self):
        """
        Rotate right to face one of the 8 cardinal directions.
        Will rotate at most 45 degrees to align with the nearest direction.

        Returns:
            str: Empty string if successful, error message if failed
        """
        log.info('rotate_right')

        # Get current position and rotation
        current_pos = self.last_event.metadata['agent']['position']
        current_rotation = self.last_event.metadata['agent']['rotation']['y']

        # Target cardinal directions (angles in degrees)
        # 0°=North, clockwise: aligned with get_agent_position / get_agent_facing_direction
        cardinal_directions = {
            'north': 0,
            'northeast': 45,
            'east': 90,
            'southeast': 135,
            'south': 180,
            'southwest': 225,
            'west': 270,
            'northwest': 315
        }

        # Find the nearest direction to the right (within 45 degrees)
        target_rotation = None
        min_diff = float('inf')
        epsilon = 1e-5  # Small tolerance for floating point comparison

        for direction_name, direction_angle in cardinal_directions.items():
            # Calculate angle difference considering wrap-around
            diff = (direction_angle - current_rotation)

            # Only consider directions to the right (positive difference, epsilon-45 degrees)
            # Exclude directions that are essentially the same as current (diff < epsilon)
            # Use epsilon tolerance on upper bound too for floating point errors
            if epsilon < diff <= (45 + epsilon):
                if diff < min_diff:
                    min_diff = diff
                    target_rotation = direction_angle
                    log.info(
                        f"rotate_right: Found {direction_name} at {direction_angle}°, diff={diff}"
                    )

        if target_rotation is None:
            return "No valid right rotation target found within 45 degrees"

        # Use TeleportFull to set the rotation directly
        super().step(
            dict(action='TeleportFull',
                 x=float(current_pos['x']),
                 y=float(current_pos['y']),
                 z=float(current_pos['z']),
                 rotation=target_rotation,
                 horizon=0,
                 rotateOnTeleport=True))

        if not self.last_event.metadata['lastActionSuccess']:
            return f"rotate_right failed: {self.last_event.metadata['errorMessage']}"

        return ""

    def rotate_left(self):
        """
        Rotate left to face one of the 8 cardinal directions.
        Will rotate at most 45 degrees to align with the nearest direction.

        Returns:
            str: Empty string if successful, error message if failed
        """
        log.info('rotate_left')

        # Get current position and rotation
        current_pos = self.last_event.metadata['agent']['position']
        current_rotation = self.last_event.metadata['agent']['rotation']['y']

        # Target cardinal directions (angles in degrees)
        # 0°=North, clockwise: aligned with get_agent_position / get_agent_facing_direction
        cardinal_directions = {
            'north': 0,
            'northeast': 45,
            'east': 90,
            'southeast': 135,
            'south': 180,
            'southwest': 225,
            'west': 270,
            'northwest': 315
        }

        # Find the nearest direction to the left (within 45 degrees)
        target_rotation = None
        min_diff = float('inf')
        epsilon = 1e-5  # Small tolerance for floating point comparison

        for direction_name, direction_angle in cardinal_directions.items():
            # Calculate angle difference considering wrap-around
            diff = (current_rotation - direction_angle) % 360

            # Only consider directions to the left (positive difference, epsilon-45 degrees)
            # Exclude directions that are essentially the same as current (diff < epsilon)
            # Use epsilon tolerance on upper bound too for floating point errors
            if epsilon < diff <= (45 + epsilon):
                if diff < min_diff:
                    min_diff = diff
                    target_rotation = direction_angle
                    log.info(
                        f"rotate_left: Found {direction_name} at {direction_angle}°, diff={diff}"
                    )

        if target_rotation is None:
            log.info(
                f"rotate_left: No valid target found at current_rotation={current_rotation}"
            )
            return "No valid left rotation target found within 45 degrees"

        log.info(
            f"rotate_left: Target rotation = {target_rotation}°, current = {current_rotation}°"
        )

        # Use TeleportFull to set the rotation directly
        super().step(
            dict(action='TeleportFull',
                 x=float(current_pos['x']),
                 y=float(current_pos['y']),
                 z=float(current_pos['z']),
                 rotation=target_rotation,
                 horizon=0,
                 rotateOnTeleport=True))

        if not self.last_event.metadata['lastActionSuccess']:
            return f"rotate_left failed: {self.last_event.metadata['errorMessage']}"

        actual_rotation = self.last_event.metadata['agent']['rotation']['y']
        log.info(
            f"rotate_left: Actual rotation after teleport = {actual_rotation}°"
        )

        return ""
