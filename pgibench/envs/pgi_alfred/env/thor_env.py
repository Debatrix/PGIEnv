import cv2
import copy
import pgibench.envs.pgi_alfred.gen.constants as constants
import numpy as np
from collections import Counter, OrderedDict, defaultdict
from pgibench.envs.pgi_alfred.env.tasks import get_task
from ai2thor.controller import Controller
import pgibench.envs.pgi_alfred.gen.utils.image_util as image_util
from pgibench.envs.pgi_alfred.gen.utils import game_util
from pgibench.envs.pgi_alfred.gen.utils.game_util import get_objects_of_type, get_obj_of_type_closest_to_obj

DEFAULT_RENDER_SETTINGS = {
    'renderImage': True,
    'renderDepthImage': False,
    'renderClassImage': False,
    'renderObjectImage': False,
}


class ThorEnv(Controller):
    '''
    an extension of ai2thor.controller.Controller for ALFRED tasks
    '''

    def __init__(self,
                 x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH,
                 visual_prompt=False):
        self.task = None

        super().__init__(quality=quality)
        self.local_executable_path = build_path
        self.start(x_display=x_display,
                   player_screen_height=player_screen_height,
                   player_screen_width=player_screen_width)
        self.task = None

        # internal states
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

        # intemediate states for CoolObject Subgoal
        self.cooled_reward = False
        self.reopen_reward = False

        # location description generation mappings
        self.surface_objects = {
            'CounterTop', 'DiningTable', 'CoffeeTable', 'SideTable',
            'StoveBurner', 'Shelf', 'Bed', 'Sofa', 'ArmChair', 'Ottoman',
            'Dresser', 'Desk', 'TVStand'
        }

        self.container_objects = constants.RECEPTACLES - self.surface_objects

        self.ordinal_numbers = [
            'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
            'eighth', 'ninth', 'tenth'
        ]
        self.visual_prompt = visual_prompt
        self.id_2_name = {}
        self.all_object_type = defaultdict(list)

        print("ThorEnv started.")

    def reset(self,
              scene_name_or_num,
              grid_size=constants.AGENT_STEP_SIZE /
              constants.RECORD_SMOOTHING_FACTOR,
              camera_y=constants.CAMERA_HEIGHT_OFFSET,
              render_image=constants.RENDER_IMAGE,
              render_depth_image=constants.RENDER_DEPTH_IMAGE,
              render_class_image=constants.RENDER_CLASS_IMAGE,
              render_object_image=constants.RENDER_OBJECT_IMAGE,
              visibility_distance=constants.VISIBILITY_DISTANCE):
        '''
        reset scene and task states
        '''
        print("Resetting ThorEnv")

        if type(scene_name_or_num) == str:
            scene_name = scene_name_or_num
        else:
            scene_name = 'FloorPlan%d' % scene_name_or_num
        super().reset(scene_name)
        event = super().step(
            dict(
                action='Initialize',
                gridSize=grid_size,
                cameraY=camera_y,
                renderImage=render_image,
                renderDepthImage=render_depth_image,
                renderClassImage=render_class_image,
                renderObjectImage=render_object_image,
                visibility_distance=visibility_distance,
                makeAgentsVisible=False,
            ))

        # reset task if specified
        if self.task is not None:
            self.task.reset()

        # clear object state changes
        self.reset_states()

        # update location descriptions after reset
        self.update_loc_descs()

        return event

    def reset_states(self):
        '''
        clear state changes
        '''
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()
        # Reset intermediate reward flags for CoolObjectAction
        self.cooled_reward = False
        self.reopen_reward = False

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        '''
        restore object locations and states
        '''
        super().step(
            dict(
                action='Initialize',
                gridSize=constants.AGENT_STEP_SIZE /
                constants.RECORD_SMOOTHING_FACTOR,
                cameraY=constants.CAMERA_HEIGHT_OFFSET,
                renderImage=constants.RENDER_IMAGE,
                renderDepthImage=constants.RENDER_DEPTH_IMAGE,
                renderClassImage=constants.RENDER_CLASS_IMAGE,
                renderObjectImage=constants.RENDER_OBJECT_IMAGE,
                visibility_distance=constants.VISIBILITY_DISTANCE,
                makeAgentsVisible=False,
            ))
        if len(object_toggles) > 0:
            super().step((dict(action='SetObjectToggles',
                               objectToggles=object_toggles)))

        if dirty_and_empty:
            super().step(
                dict(action='SetStateOfAllObjects',
                     StateChange="CanBeDirty",
                     forceAction=True))
            super().step(
                dict(action='SetStateOfAllObjects',
                     StateChange="CanBeFilled",
                     forceAction=False))
        super().step((dict(action='SetObjectPoses', objectPoses=object_poses)))

    def set_task(self,
                 traj,
                 args,
                 reward_type='sparse',
                 max_episode_length=2000):
        '''
        set the current task type (one of 7 tasks)
        '''
        task_type = traj['task_type']
        self.task = get_task(task_type,
                             traj,
                             self,
                             args,
                             reward_type=reward_type,
                             max_episode_length=max_episode_length)

    def step(self, action, smooth_nav=False):
        '''
        overrides ai2thor.controller.Controller.step() for smooth navigation and goal_condition updates
        '''
        if type(action) == str:
            action = {'action': action}
        if smooth_nav:
            if "MoveAhead" in action['action']:
                self.smooth_move_ahead(action)
            elif "Rotate" in action['action']:
                self.smooth_rotate(action)
            elif "Look" in action['action']:
                self.smooth_look(action)
            else:
                super().step(action)
        else:
            if "LookUp" in action['action']:
                self.look_angle(-constants.AGENT_HORIZON_ADJ)
            elif "LookDown" in action['action']:
                self.look_angle(constants.AGENT_HORIZON_ADJ)
            else:
                super().step(action)

        event = self.update_states(action)
        self.check_post_conditions(action)
        return event

    def check_post_conditions(self, action):
        '''
        handle special action post-conditions
        '''
        if action['action'] == 'ToggleObjectOn':
            self.check_clean(action['objectId'])

    def update_states(self, action):
        '''
        extra updates to metadata after step
        '''
        # add 'cleaned' to all object that were washed in the sink
        event = self.last_event
        if event.metadata['lastActionSuccess']:
            # clean
            if action['action'] == 'ToggleObjectOn' and "Faucet" in action[
                    'objectId']:
                sink_basin = get_obj_of_type_closest_to_obj(
                    'SinkBasin', action['objectId'], event.metadata)
                cleaned_object_ids = sink_basin['receptacleObjectIds']
                self.cleaned_objects = self.cleaned_objects | set(
                    cleaned_object_ids
                ) if cleaned_object_ids is not None else set()
            # heat
            if action['action'] == 'ToggleObjectOn' and "Microwave" in action[
                    'objectId']:
                microwave = get_objects_of_type('Microwave', event.metadata)[0]
                heated_object_ids = microwave['receptacleObjectIds']
                self.heated_objects = self.heated_objects | set(
                    heated_object_ids
                ) if heated_object_ids is not None else set()
            # cool
            if action['action'] == 'CloseObject' and "Fridge" in action[
                    'objectId']:
                fridge = get_objects_of_type('Fridge', event.metadata)[0]
                cooled_object_ids = fridge['receptacleObjectIds']
                self.cooled_objects = self.cooled_objects | set(
                    cooled_object_ids
                ) if cooled_object_ids is not None else set()

        # update location descriptions for all objects
        self.update_loc_descs()

        # update agent facing direction
        event.metadata['agent'][
            'facingDirection'] = self.get_agent_facing_direction()

        return event

    def get_agent_facing_direction(self):
        """
        Calculate the agent's facing direction based on rotation angle (8 directions).

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
            tuple: (direction_description, angle_value)
                direction_description: One of "north", "northeast", "east", "southeast",
                                        "south", "southwest", "west", "northwest"
                angle_value: Rotation angle in degrees [0, 360)
        """
        if not hasattr(self.last_event,
                       'metadata') or 'agent' not in self.last_event.metadata:
            return "unknown", 0.0

        rotation_y = self.last_event.metadata['agent']['rotation']['y']
        # Normalize angle to 0-360 range
        angle = rotation_y % 360

        # Determine direction based on angle (8 directions)
        # 0° faces North (+Z), angle increases clockwise
        # North: 337.5-360 or 0-22.5
        if 337.5 <= angle or angle < 22.5:
            direction = 'north'
        # Northeast: 22.5-67.5 (45° clockwise)
        elif 22.5 <= angle < 67.5:
            direction = 'northeast'
        # East: 67.5-112.5 (90° clockwise)
        elif 67.5 <= angle < 112.5:
            direction = 'east'
        # Southeast: 112.5-157.5 (135° clockwise)
        elif 112.5 <= angle < 157.5:
            direction = 'southeast'
        # South: 157.5-202.5 (180° clockwise)
        elif 157.5 <= angle < 202.5:
            direction = 'south'
        # Southwest: 202.5-247.5 (225° clockwise)
        elif 202.5 <= angle < 247.5:
            direction = 'southwest'
        # West: 247.5-292.5 (270° clockwise)
        elif 247.5 <= angle < 292.5:
            direction = 'west'
        # Northwest: 292.5-337.5 (315° clockwise)
        else:  # 292.5 <= angle < 337.5
            direction = 'northwest'

        return direction, angle

    def get_transition_reward(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for transition_reward")
        else:
            return self.task.transition_reward(self.last_event)

    def get_goal_satisfied(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_satisfied(self.last_event)

    def get_goal_conditions_met(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_conditions_met(self.last_event)

    def get_subgoal_idx(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for subgoal_idx")
        else:
            return self.task.get_subgoal_idx()

    def get_subtask_details(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for subtask_details")
        else:
            return self.task.get_subtask_details(self.last_event)

    def noop(self):
        '''
        do nothing
        '''
        super().step(dict(action='Pass'))

    def smooth_move_ahead(self, action, render_settings=None):
        '''
        smoother MoveAhead
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        smoothing_factor = constants.RECORD_SMOOTHING_FACTOR
        new_action = copy.deepcopy(action)
        new_action[
            'moveMagnitude'] = constants.AGENT_STEP_SIZE / smoothing_factor

        new_action['renderImage'] = render_settings['renderImage']
        new_action['renderClassImage'] = render_settings['renderClassImage']
        new_action['renderObjectImage'] = render_settings['renderObjectImage']
        new_action['renderDepthImage'] = render_settings['renderDepthImage']

        events = []
        for xx in range(smoothing_factor - 1):
            event = super().step(new_action)
            if event.metadata['lastActionSuccess']:
                events.append(event)

        event = super().step(new_action)
        if event.metadata['lastActionSuccess']:
            events.append(event)
        return events

    def smooth_rotate(self, action, render_settings=None):
        '''
        smoother RotateLeft and RotateRight
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        if action['action'] == 'RotateLeft':
            end_rotation = (start_rotation - 90)
        else:
            end_rotation = (start_rotation + 90)

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action':
                    'TeleportFull',
                    'rotation':
                    np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x':
                    position['x'],
                    'z':
                    position['z'],
                    'y':
                    position['y'],
                    'horizon':
                    horizon,
                    'tempRenderChange':
                    True,
                    'renderNormalsImage':
                    False,
                    'renderImage':
                    render_settings['renderImage'],
                    'renderClassImage':
                    render_settings['renderClassImage'],
                    'renderObjectImage':
                    render_settings['renderObjectImage'],
                    'renderDepthImage':
                    render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action':
                    'TeleportFull',
                    'rotation':
                    np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x':
                    position['x'],
                    'z':
                    position['z'],
                    'y':
                    position['y'],
                    'horizon':
                    horizon,
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def smooth_look(self, action, render_settings=None):
        '''
        smoother LookUp and LookDown
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + constants.AGENT_HORIZON_ADJ * (
            1 - 2 * int(action['action'] == 'LookUp'))
        position = event.metadata['agent']['position']

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action':
                    'TeleportFull',
                    'rotation':
                    rotation,
                    'x':
                    position['x'],
                    'z':
                    position['z'],
                    'y':
                    position['y'],
                    'horizon':
                    np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                    'tempRenderChange':
                    True,
                    'renderNormalsImage':
                    False,
                    'renderImage':
                    render_settings['renderImage'],
                    'renderClassImage':
                    render_settings['renderClassImage'],
                    'renderObjectImage':
                    render_settings['renderObjectImage'],
                    'renderDepthImage':
                    render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action':
                    'TeleportFull',
                    'rotation':
                    rotation,
                    'x':
                    position['x'],
                    'z':
                    position['z'],
                    'y':
                    position['y'],
                    'horizon':
                    np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def look_angle(self, angle, render_settings=None):
        '''
        look at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + angle
        position = event.metadata['agent']['position']

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': rotation,
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': np.round(end_horizon, 3),
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        return event

    def rotate_angle(self, angle, render_settings=None):
        '''
        rotate at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        end_rotation = start_rotation + angle

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': np.round(end_rotation, 3),
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': horizon,
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        return event

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        # TODO: parametrized navigation commands

        if "RotateLeft" in action:
            action = dict(action="RotateLeft", forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "RotateRight" in action:
            action = dict(action="RotateRight", forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "MoveAhead" in action:
            action = dict(action="MoveAhead", forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookUp" in action:
            action = dict(action="LookUp", forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookDown" in action:
            action = dict(action="LookDown", forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "OpenObject" in action:
            action = dict(action="OpenObject",
                          objectId=object_id,
                          moveMagnitude=1.0)
            event = self.step(action)
        elif "CloseObject" in action:
            action = dict(action="CloseObject",
                          objectId=object_id,
                          forceAction=True)
            event = self.step(action)
        elif "PickupObject" in action:
            action = dict(action="PickupObject", objectId=object_id)
            event = self.step(action)
        elif "PutObject" in action:
            inventory_object_id = self.last_event.metadata['inventoryObjects'][
                0]['objectId']
            action = dict(action="PutObject",
                          objectId=inventory_object_id,
                          receptacleObjectId=object_id,
                          forceAction=True,
                          placeStationary=True)
            event = self.step(action)
        elif "ToggleObjectOn" in action:
            action = dict(action="ToggleObjectOn", objectId=object_id)
            event = self.step(action)

        elif "ToggleObjectOff" in action:
            action = dict(action="ToggleObjectOff", objectId=object_id)
            event = self.step(action)
        elif "SliceObject" in action:
            # check if agent is holding knife in hand
            inventory_objects = self.last_event.metadata['inventoryObjects']
            if len(inventory_objects
                   ) == 0 or 'Knife' not in inventory_objects[0]['objectType']:
                raise Exception(
                    "Agent should be holding a knife before slicing.")

            action = dict(action="SliceObject", objectId=object_id)
            event = self.step(action)
        else:
            raise Exception(
                "Invalid action. Conversion to THOR API failed! (action='" +
                str(action) + "')")

        return event, action

    def check_clean(self, object_id):
        '''
        Handle special case when Faucet is toggled on.
        In this case, we need to execute a `CleanAction` in the simulator on every object in the corresponding
        basin. This is to clean everything in the sink rather than just things touching the stream.
        '''
        event = self.last_event
        if event.metadata['lastActionSuccess'] and 'Faucet' in object_id:
            # Need to delay one frame to let `isDirty` update on stream-affected.
            event = self.step({'action': 'Pass'})
            sink_basin_obj = game_util.get_obj_of_type_closest_to_obj(
                "SinkBasin", object_id, event.metadata)
            for in_sink_obj_id in sink_basin_obj['receptacleObjectIds']:
                if (game_util.get_object(in_sink_obj_id,
                                         event.metadata)['dirtyable']
                        and game_util.get_object(in_sink_obj_id,
                                                 event.metadata)['isDirty']):
                    event = self.step({
                        'action': 'CleanObject',
                        'objectId': in_sink_obj_id
                    })
        return event

    def prune_by_any_interaction(self, instances_ids):
        '''
        ignores any object that is not interactable in anyway
        '''
        pruned_instance_ids = []
        for obj in self.last_event.metadata['objects']:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj[
                        'openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id)

        ordered_instance_ids = [
            id for id in instances_ids if id in pruned_instance_ids
        ]
        return ordered_instance_ids

    def va_interact(self,
                    action,
                    interact_mask=None,
                    smooth_nav=True,
                    mask_px_sample=1,
                    debug=False):
        '''
        interact mask based action call
        '''

        all_ids = []

        if type(interact_mask) is str and interact_mask == "NULL":
            raise Exception("NULL mask.")
        elif interact_mask is not None:
            # ground-truth instance segmentation mask from THOR
            instance_segs = np.array(
                self.last_event.instance_segmentation_frame)
            color_to_object_id = self.last_event.color_to_object_id

            # get object_id for each 1-pixel in the interact_mask
            nz_rows, nz_cols = np.nonzero(interact_mask)
            instance_counter = Counter()
            for i in range(0, len(nz_rows), mask_px_sample):
                x, y = nz_rows[i], nz_cols[i]
                instance = tuple(instance_segs[x, y])
                instance_counter[instance] += 1
            if debug:
                print("action_box", "instance_counter", instance_counter)

            # iou scores for all instances
            iou_scores = {}
            for color_id, intersection_count in instance_counter.most_common():
                union_count = np.sum(
                    np.logical_or(np.all(instance_segs == color_id, axis=2),
                                  interact_mask.astype(bool)))
                iou_scores[color_id] = intersection_count / float(union_count)
            iou_sorted_instance_ids = list(
                OrderedDict(
                    sorted(iou_scores.items(),
                           key=lambda x: x[1],
                           reverse=True)))

            # get the most common object ids ignoring the object-in-hand
            inv_obj = self.last_event.metadata['inventoryObjects'][0]['objectId'] \
                if len(self.last_event.metadata['inventoryObjects']) > 0 else None
            all_ids = [
                color_to_object_id[color_id]
                for color_id in iou_sorted_instance_ids
                if color_id in color_to_object_id
                and color_to_object_id[color_id] != inv_obj
            ]

            # print all ids
            if debug:
                print("action_box", "all_ids", all_ids)

            # print instance_ids
            instance_ids = [
                inst_id for inst_id in all_ids if inst_id is not None
            ]
            if debug:
                print("action_box", "instance_ids", instance_ids)

            # prune invalid instances like floors, walls, etc.
            instance_ids = self.prune_by_any_interaction(instance_ids)

            # cv2 imshows to show image, segmentation mask, interact mask
            if debug:
                print("action_box", "instance_ids", instance_ids)
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                instance_seg *= 255

                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.last_event.frame[:, :, ::-1])
                cv2.waitKey(0)

            if len(instance_ids) == 0:
                err = "Bad interact mask. Couldn't locate target object"
                success = False
                return success, None, None, err, None

            target_instance_id = instance_ids[0]
        else:
            target_instance_id = ""

        if debug:
            print("taking action: " + str(action) + " on target_instance_id " +
                  str(target_instance_id))
        try:
            event, api_action = self.to_thor_api_exec(action,
                                                      target_instance_id,
                                                      smooth_nav)
        except Exception as err:
            success = False
            return success, None, None, err, None

        if not event.metadata['lastActionSuccess']:
            if interact_mask is not None and debug:
                print("Failed to execute action!", action, target_instance_id)
                print("all_ids inside BBox: " + str(all_ids))
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.last_event.frame[:, :, ::-1])
                cv2.waitKey(0)
                print(event.metadata['errorMessage'])
            success = False
            return success, event, target_instance_id, event.metadata[
                'errorMessage'], api_action

        success = True
        return success, event, target_instance_id, '', api_action

    @staticmethod
    def bbox_to_mask(bbox):
        return image_util.bbox_to_mask(bbox)

    @staticmethod
    def point_to_mask(point):
        return image_util.point_to_mask(point)

    @staticmethod
    def decompress_mask(compressed_mask):
        return image_util.decompress_mask(compressed_mask)

    def get_preposition(self, parent_object_type):
        """
        Determine the appropriate preposition based on container or surface type.
        """
        if parent_object_type in self.surface_objects:
            return 'on'
        elif parent_object_type in self.container_objects:
            return 'in'
        else:
            return 'near'

    def group_objects_by_proximity_and_container(self, objects):
        """
        Group objects of the same type by container and location.
        Objects in the same group are of the same type and either in the same container,
        same container type, or have at least one coordinate (xyz) difference < 0.15.
        """
        groups = defaultdict(list)
        processed = set()

        # Process by object type
        for obj_type in set(obj['objectType'] for obj in objects):
            # Get all objects of the same type
            same_type_objects = [
                obj for obj in objects if obj['objectType'] == obj_type
            ]

            # First group by container type (objects in same container type go together)
            container_groups = defaultdict(list)
            for obj in same_type_objects:
                if obj['parentReceptacles']:
                    # Group by container type, not specific container instance
                    parent_id = obj['parentReceptacles'][0]
                    parent_obj = next(
                        (o for o in objects if o['objectId'] == parent_id),
                        None)
                    if parent_obj:
                        container_type = parent_obj['objectType']
                        container_groups[container_type].append(obj)

            # Create group for each container type
            for container_type, container_objects in container_groups.items():
                key = ('container_type', container_type, obj_type)
                groups[key].extend(container_objects)
                processed.update(obj['objectId'] for obj in container_objects)

            # Then group remaining ungrouped objects by location proximity
            for obj in same_type_objects:
                if obj['objectId'] in processed:
                    continue

                obj_pos = obj['position']
                group_key = (
                    'position',
                    f"{obj_type}_{obj_pos['x']:.2f}_{obj_pos['y']:.2f}_{obj_pos['z']:.2f}"
                )

                # Add current object
                groups[group_key].append(obj)
                processed.add(obj['objectId'])

                # Find other objects of the same type nearby
                for other_obj in same_type_objects:
                    if other_obj['objectId'] not in processed:
                        other_pos = other_obj['position']

                        # Check if at least one coordinate differs by less than 0.15
                        x_close = abs(obj_pos['x'] - other_pos['x']) < 0.15
                        y_close = abs(obj_pos['y'] - other_pos['y']) < 0.15
                        z_close = abs(obj_pos['z'] - other_pos['z']) < 0.15

                        if x_close or y_close or z_close:
                            groups[group_key].append(other_obj)
                            processed.add(other_obj['objectId'])

        return groups

    def find_reference_object(self,
                              obj,
                              all_objects,
                              exclude_objects=None,
                              prefer_unique=True):
        """
        Find a reference object for the given object.
        """
        if exclude_objects is None:
            exclude_objects = set()

        obj_pos = obj['position']
        best_ref = None
        min_distance = float('inf')

        # If prefer_unique is True, prioritize unique objects as reference
        if prefer_unique:
            # First collect all unique non-pickupable objects
            unique_objects = []
            for ref_obj in all_objects:
                if ref_obj['objectId'] in exclude_objects:
                    continue
                if ref_obj['objectId'] == obj['objectId']:
                    continue
                if not ref_obj['pickupable']:
                    # Check if this object type is unique
                    same_type_count = sum(
                        1 for o in all_objects
                        if o['objectType'] == ref_obj['objectType'])
                    if same_type_count == 1:
                        unique_objects.append(ref_obj)

            # If there are unique non-pickupable objects, select the closest one
            if unique_objects:
                for ref_obj in unique_objects:
                    ref_pos = ref_obj['position']
                    distance = np.sqrt((obj_pos['x'] - ref_pos['x'])**2 +
                                       (obj_pos['y'] - ref_pos['y'])**2 +
                                       (obj_pos['z'] - ref_pos['z'])**2)

                    if distance < min_distance:
                        min_distance = distance
                        best_ref = ref_obj

        # If no unique object found, or prefer_unique is False, use original logic
        if best_ref is None:
            for ref_obj in all_objects:
                if ref_obj['objectId'] in exclude_objects:
                    continue
                if ref_obj['objectId'] == obj['objectId']:
                    continue

                # Prioritize non-pickupable objects as reference
                if not ref_obj['pickupable']:
                    ref_pos = ref_obj['position']
                    distance = np.sqrt((obj_pos['x'] - ref_pos['x'])**2 +
                                       (obj_pos['y'] - ref_pos['y'])**2 +
                                       (obj_pos['z'] - ref_pos['z'])**2)

                    if distance < min_distance:
                        min_distance = distance
                        best_ref = ref_obj

        return best_ref

    def get_spatial_relation(self, obj, ref_obj):
        """
        Determine the spatial relationship between two objects using absolute directions (8 directions).

        Coordinate system definition:
        - Y-axis: Vertical (up/down)
        - XZ-plane: Horizontal directions
          - +Z direction = North
          - +X direction = East
          - -Z direction = South
          - -X direction = West

        Args:
            obj: Target object
            ref_obj: Reference object

        Returns:
            str: Spatial relationship description (e.g., 'to the north of', 'to the east of',
                 'above', 'below', 'near')
        """
        obj_pos = obj['position']
        ref_pos = ref_obj['position']

        # Calculate relative position difference (using X and Z axes, Y axis is height)
        dx = obj_pos['x'] - ref_pos['x']  # East-West (+X=East)
        dy = obj_pos['y'] - ref_pos['y']  # Height difference
        dz = obj_pos['z'] - ref_pos['z']  # North-South (+Z=North)

        # Calculate horizontal distance
        horizontal_distance = np.sqrt(dx**2 + dz**2)

        # Prioritize vertical relationships
        if horizontal_distance < 0.1:
            if dy > 0.1:
                return 'above'
            elif dy < -0.1:
                return 'below'
            else:
                return 'near'

        # Calculate angle to determine 8 directions
        # arctan2(dz, dx) returns angle where 0°=+X (East), 90°=+Z (North)
        # Need to convert to agent's rotation system (0°=North, clockwise increase)
        angle_rad = np.arctan2(dz, dx)
        angle_deg = np.degrees(angle_rad)

        # Convert reference system: from "East=0°" to "North=0°"
        # 90° (North) -> 0°, 0° (East) -> 90°
        angle = (90 - angle_deg) % 360

        # Determine direction based on angle (0°=North, clockwise increase)
        # North: 337.5-360 or 0-22.5
        if 337.5 <= angle or angle < 22.5:
            return 'to the north of'
        # Northeast: 22.5-67.5
        elif 22.5 <= angle < 67.5:
            return 'to the northeast of'
        # East: 67.5-112.5
        elif 67.5 <= angle < 112.5:
            return 'to the east of'
        # Southeast: 112.5-157.5
        elif 112.5 <= angle < 157.5:
            return 'to the southeast of'
        # South: 157.5-202.5
        elif 157.5 <= angle < 202.5:
            return 'to the south of'
        # Southwest: 202.5-247.5
        elif 202.5 <= angle < 247.5:
            return 'to the southwest of'
        # West: 247.5-292.5
        elif 247.5 <= angle < 292.5:
            return 'to the west of'
        # Northwest: 292.5-337.5
        else:  # 292.5 <= angle < 337.5
            return 'to the northwest of'

    def _name_unique_non_pickupable_object(self, obj, obj_type, used_names):
        """
        Name unique non-pickupable objects
        Use objectType directly as the name
        """
        name = f"the {obj_type.lower()}"
        if name not in used_names:
            return name
        return None

    def _name_unique_pickupable_object(self, obj, obj_type, all_objects,
                                       used_names):
        """
        Name unique pickupable objects
        Use objectType as base name and find a reference object to modify it
        """
        # Prioritize using container information to modify
        if obj['parentReceptacles']:
            parent_id = obj['parentReceptacles'][0]
            parent_obj = next(
                (o for o in all_objects if o['objectId'] == parent_id), None)
            if parent_obj:
                parent_type = parent_obj['objectType']
                preposition = self.get_preposition(parent_type)
                name = f"the {obj_type.lower()} {preposition} the {parent_type.lower()}"
                if name not in used_names:
                    return name

        # Rule 2.2: if the object stands alone, describe it by the nearest unique non-pickupable object
        ref_obj = self.find_reference_object(obj, all_objects)
        if ref_obj:
            spatial_relation = self.get_spatial_relation(obj, ref_obj)
            ref_type = ref_obj['objectType']
            name = f"the {obj_type.lower()} {spatial_relation} the {ref_type.lower()}"
            if name not in used_names:
                return name

        return None

    def _name_multiple_objects_by_group(self,
                                        obj,
                                        obj_type,
                                        all_objects,
                                        used_names,
                                        obj_group=None,
                                        common_ref=None):
        """
        Name multiple objects of the same type
        After grouping by location/container, distinguish with ordinal numbers within groups
        """
        # Determine which group the object belongs to
        current_group = obj_group
        group_common_ref = common_ref

        # If no group information provided, group by location/container
        if not current_group:
            # Get all objects of the same type
            same_type_objects = [
                o for o in all_objects if o['objectType'] == obj_type
            ]
            # Build temporary proximity groups
            temp_groups = self._group_same_type_objects_by_proximity(
                same_type_objects)
            for group in temp_groups.values():
                if obj in group:
                    current_group = group
                    group_common_ref = self.find_common_reference_for_group(
                        group, all_objects)
                    break

        # If still no group, use all same-type objects as group
        if not current_group:
            current_group = [
                o for o in all_objects if o['objectType'] == obj_type
            ]

        # Count same-type objects in current group
        same_type_in_group = [
            o for o in current_group if o['objectType'] == obj_type
        ]
        group_size = len(same_type_in_group)

        # Rule 3.1: if its group holds a single object, fall back to rule 2
        if group_size == 1:
            # Prioritize using container information
            if obj['parentReceptacles']:
                parent_id = obj['parentReceptacles'][0]
                parent_obj = next(
                    (o for o in all_objects if o['objectId'] == parent_id),
                    None)
                if parent_obj:
                    parent_type = parent_obj['objectType']
                    preposition = self.get_preposition(parent_type)
                    name = f"the {obj_type.lower()} {preposition} the {parent_type.lower()}"
                    if name not in used_names:
                        return name

            # Use common reference to ensure consistency with other objects in group
            if group_common_ref:
                # Get unified spatial relation conjunction (if group has other object types)
                unified_relation = self.get_unified_spatial_relation_for_group(
                    current_group, group_common_ref)
                if unified_relation:
                    ref_type = group_common_ref['objectType']
                    name = f"the {obj_type.lower()} {unified_relation} the {ref_type.lower()}"
                    if name not in used_names:
                        return name
                else:
                    # If this is the only object in the group, use its own spatial relation
                    spatial_relation = self.get_spatial_relation(
                        obj, group_common_ref)
                    ref_type = group_common_ref['objectType']
                    name = f"the {obj_type.lower()} {spatial_relation} the {ref_type.lower()}"
                    if name not in used_names:
                        return name

            # Find reference object
            ref_obj = self.find_reference_object(obj, all_objects)
            if ref_obj:
                spatial_relation = self.get_spatial_relation(obj, ref_obj)
                ref_type = ref_obj['objectType']
                name = f"the {obj_type.lower()} {spatial_relation} the {ref_type.lower()}"
                if name not in used_names:
                    return name

        # Rule 3.2: If group has multiple objects, number by spatial position within the group
        elif group_size > 1:
            # Sort by position within group to get ordinal number
            same_type_in_group.sort(key=lambda o: (o['position']['x'], o[
                'position']['y'], o['position']['z']))
            obj_index = same_type_in_group.index(obj)
            ordinal = self.ordinal_numbers[obj_index] if obj_index < len(
                self.ordinal_numbers) else f"{obj_index + 1}th"

            # Check if all objects in group are in same-type containers
            if all(obj['parentReceptacles'] for obj in same_type_in_group):
                # Get all container types
                container_types = set()
                for group_obj in same_type_in_group:
                    parent_id = group_obj['parentReceptacles'][0]
                    parent_obj = next(
                        (o for o in all_objects if o['objectId'] == parent_id),
                        None)
                    if parent_obj:
                        container_types.add(parent_obj['objectType'])

                # If all objects are in same-type containers, use unified container description
                if len(container_types) == 1:
                    container_type = list(container_types)[0]
                    preposition = self.get_preposition(container_type)
                    name = f"the {ordinal} {obj_type.lower()} {preposition} the {container_type.lower()}"
                    if name not in used_names:
                        return name

            # Prioritize using common reference to ensure same conjunction for group
            if group_common_ref:
                # Get unified spatial relation conjunction
                unified_relation = self.get_unified_spatial_relation_for_group(
                    same_type_in_group, group_common_ref)
                if unified_relation:
                    ref_type = group_common_ref['objectType']
                    name = f"the {ordinal} {obj_type.lower()} {unified_relation} the {ref_type.lower()}"
                    if name not in used_names:
                        return name

            # Use current object's container information (if different containers)
            if obj['parentReceptacles']:
                parent_id = obj['parentReceptacles'][0]
                parent_obj = next(
                    (o for o in all_objects if o['objectId'] == parent_id),
                    None)
                if parent_obj:
                    parent_type = parent_obj['objectType']
                    preposition = self.get_preposition(parent_type)
                    name = f"the {ordinal} {obj_type.lower()} {preposition} the {parent_type.lower()}"
                    if name not in used_names:
                        return name

            # If there is common reference but cannot unify conjunction, still use common reference but allow different conjunction
            if group_common_ref:
                spatial_relation = self.get_spatial_relation(
                    obj, group_common_ref)
                ref_type = group_common_ref['objectType']
                name = f"the {ordinal} {obj_type.lower()} {spatial_relation} the {ref_type.lower()}"
                if name not in used_names:
                    return name

            # Only now look for a new reference object (this should rarely happen)
            ref_obj = self.find_reference_object(obj, all_objects)
            if ref_obj:
                spatial_relation = self.get_spatial_relation(obj, ref_obj)
                ref_type = ref_obj['objectType']
                name = f"the {ordinal} {obj_type.lower()} {spatial_relation} the {ref_type.lower()}"
                if name not in used_names:
                    return name

        return None

    def _group_same_type_objects_by_proximity(self, same_type_objects):
        """
        Group objects of the same type by location proximity
        """
        groups = defaultdict(list)
        processed = set()

        for obj in same_type_objects:
            if obj['objectId'] in processed:
                continue

            obj_pos = obj['position']
            group_key = f"group_{obj_pos['x']:.2f}_{obj_pos['y']:.2f}_{obj_pos['z']:.2f}"
            groups[group_key].append(obj)
            processed.add(obj['objectId'])

            # Find other objects nearby
            for other_obj in same_type_objects:
                if other_obj['objectId'] not in processed:
                    other_pos = other_obj['position']

                    # Check if at least one coordinate differs by less than 0.15
                    x_close = abs(obj_pos['x'] - other_pos['x']) < 0.15
                    y_close = abs(obj_pos['y'] - other_pos['y']) < 0.15
                    z_close = abs(obj_pos['z'] - other_pos['z']) < 0.15

                    if x_close or y_close or z_close:
                        groups[group_key].append(other_obj)
                        processed.add(other_obj['objectId'])

        return groups

    def generate_unique_loc_desc(self,
                                 obj,
                                 all_objects,
                                 used_names,
                                 obj_group=None,
                                 common_ref=None):
        """
        Generate unique location description for an object
        Following the required process:
        1. Non-pickupable and no same-type objects -> Use objectType directly
        2. Pickupable and no same-type objects -> Use reference object to modify
        3. Multiple same-type objects -> Process after grouping
        """
        obj_type = obj['objectType']
        is_pickupable = obj['pickupable']

        # Get same-type objects
        same_type_objects = [
            o for o in all_objects if o['objectType'] == obj_type
        ]

        # Rule 1: non-pickupable with no same-type objects
        if not is_pickupable and len(same_type_objects) == 1:
            result = self._name_unique_non_pickupable_object(
                obj, obj_type, used_names)
            if result:
                return result

        # Rule 2: pickupable with no same-type objects
        if is_pickupable and len(same_type_objects) == 1:
            result = self._name_unique_pickupable_object(
                obj, obj_type, all_objects, used_names)
            if result:
                return result

        # Rule 3: multiple same-type objects
        if len(same_type_objects) > 1:
            result = self._name_multiple_objects_by_group(
                obj, obj_type, all_objects, used_names, obj_group, common_ref)
            if result:
                return result

        # Final fallback (should not reach here)
        same_type_objects.sort(key=lambda o: (o['position']['x'], o['position']
                                              ['y'], o['position']['z']))
        obj_index = same_type_objects.index(obj)
        ordinal = self.ordinal_numbers[obj_index] if obj_index < len(
            self.ordinal_numbers) else f"{obj_index + 1}th"
        name = f"the {ordinal} {obj_type.lower()}"
        return name

    def has_spatial_relation_to(self, obj, ref_obj, relation):
        """
        Check if object has a specific spatial relationship with reference object
        """
        actual_relation = self.get_spatial_relation(obj, ref_obj)
        return actual_relation == relation

    def find_common_reference_for_group(self, group_objects, all_objects):
        """
        Find a common reference object for a group, prioritizing references that ensure unified conjunctions
        """
        if not group_objects:
            return None

        # Collect all possible unique reference objects
        unique_references = []
        for ref_obj in all_objects:
            if not ref_obj['pickupable']:
                # Check if this object type is unique
                same_type_count = sum(
                    1 for o in all_objects
                    if o['objectType'] == ref_obj['objectType'])
                if same_type_count == 1:
                    # Ensure reference is not in the group
                    if ref_obj not in group_objects:
                        unique_references.append(ref_obj)

        # If no unique reference, consider all non-pickupable objects
        if not unique_references:
            for ref_obj in all_objects:
                if not ref_obj['pickupable'] and ref_obj not in group_objects:
                    unique_references.append(ref_obj)

        if not unique_references:
            return None

        # Calculate score for each reference: consider distance and conjunction consistency
        best_ref = None
        best_score = -1

        for ref_obj in unique_references:
            total_distance = 0
            spatial_relations = set()

            # Calculate total distance and spatial relations to all objects in group
            for group_obj in group_objects:
                group_pos = group_obj['position']
                ref_pos = ref_obj['position']
                distance = np.sqrt((group_pos['x'] - ref_pos['x'])**2 +
                                   (group_pos['y'] - ref_pos['y'])**2 +
                                   (group_pos['z'] - ref_pos['z'])**2)
                total_distance += distance

                # Calculate spatial relation
                spatial_relation = self.get_spatial_relation(
                    group_obj, ref_obj)
                spatial_relations.add(spatial_relation)

            avg_distance = total_distance / len(group_objects)

            # Calculate score: closer distance = higher score, fewer conjunctions = higher score
            distance_score = 1.0 / (1.0 + avg_distance)
            consistency_score = 1.0 / len(spatial_relations)  # fewer conjunctions is better
            total_score = distance_score * 0.7 + consistency_score * 0.3  # distance matters more

            if total_score > best_score:
                best_score = total_score
                best_ref = ref_obj

        return best_ref

    def get_unified_spatial_relation_for_group(self, group_objects, ref_obj):
        """
        Find a unified spatial relation conjunction for a group of objects
        """
        if not group_objects or not ref_obj:
            return None

        # Collect spatial relations of all objects in group relative to reference
        spatial_relations = []
        for group_obj in group_objects:
            spatial_relation = self.get_spatial_relation(group_obj, ref_obj)
            spatial_relations.append(spatial_relation)

        # If all objects use the same conjunction, return directly
        if len(set(spatial_relations)) == 1:
            return spatial_relations[0]

        # Otherwise, select the most common spatial relation
        from collections import Counter
        relation_counts = Counter(spatial_relations)
        most_common_relation = relation_counts.most_common(1)[0][0]
        return most_common_relation

    def _get_location_description(self,
                                  obj,
                                  all_objects,
                                  group_objects=None,
                                  common_ref=None):
        """
        Get the location description of an object
        """
        # Prioritize using container information
        if obj['parentReceptacles']:
            parent_id = obj['parentReceptacles'][0]
            parent_obj = next(
                (o for o in all_objects if o['objectId'] == parent_id), None)
            if parent_obj:
                parent_type = parent_obj['objectType']
                preposition = self.get_preposition(parent_type)
                return f"{preposition} the {parent_type.lower()}"

        # If there is a common reference, use it
        if common_ref:
            spatial_relation = self.get_spatial_relation(obj, common_ref)
            ref_type = common_ref['objectType']
            return f"{spatial_relation} the {ref_type.lower()}"

        # Find the nearest immovable reference object
        ref_obj = self.find_reference_object(obj, all_objects)
        if ref_obj:
            spatial_relation = self.get_spatial_relation(obj, ref_obj)
            ref_type = ref_obj['objectType']
            return f"{spatial_relation} the {ref_type.lower()}"

        # Use coordinate position (rarely used)
        pos = obj['position']
        if pos['x'] < -2:
            return "on the left side"
        elif pos['x'] > 2:
            return "on the right side"
        elif pos['z'] < 0:
            return "at the back"
        else:
            return "at the front"

    def update_loc_descs(self):
        """
        Update locDesc property for all objects in the environment
        Highest priority rule: If object is in agent's hand, use "in hand" as location description
        """
        if not hasattr(
                self.last_event,
                'metadata') or 'objects' not in self.last_event.metadata:
            return

        objects = self.last_event.metadata['objects']

        # Rebuild id_2_name / all_object_type from scratch every call. This
        # mirrors a fresh object-detection pass each frame: the labels only have
        # to be self-consistent within the current observation (image overlay,
        # '## Visible Objects' catalog, and the id-mode resolver all read the
        # same snapshot), and need not be stable across steps or episodes.
        # Indexing within a type is the enumeration order of this frame, so
        # '<type>_#<n>' restarts at _#0 per type per frame.
        self.id_2_name = {}
        self.all_object_type = defaultdict(list)
        for x in objects:
            self.all_object_type[x['objectType']].append(x['objectId'])
        for x in objects:
            index = self.all_object_type[x['objectType']].index(x['objectId'])
            # ID format '<type>_#<n>' mirrors the waypoint_#NN convention,
            # so the same '<type>_#<n>' label is used both as the bbox label
            # drawn on the image and as the object reference accepted in
            # command-mode 'pick up mug_#0' etc.
            self.id_2_name[x['objectId']] = f"{x['objectType'].lower()}_#{index}"

        if self.visual_prompt:
            for obj in objects:
                obj['locDesc'] = self.id_2_name[obj['objectId']]
        else:
            # Get list of object IDs in agent's hand
            inventory_object_ids = set()
            if 'inventoryObjects' in self.last_event.metadata:
                inventory_object_ids = set(
                    obj['objectId']
                    for obj in self.last_event.metadata['inventoryObjects'])

            # Generate locDesc for each object, ensuring uniqueness
            used_names = set()

            # First handle objects in agent's hand (highest priority)
            for obj_id in inventory_object_ids:
                obj = next((o for o in objects if o['objectId'] == obj_id),
                           None)
                if obj:
                    obj_type = obj['objectType'].lower()
                    loc_desc = f"the {obj_type} in hand"
                    obj['locDesc'] = loc_desc
                    used_names.add(loc_desc)

            # Group remaining objects by container and location (excluding objects in agent's hand)
            remaining_objects = [
                obj for obj in objects
                if obj['objectId'] not in inventory_object_ids
            ]
            groups = self.group_objects_by_proximity_and_container(
                remaining_objects)

            # Find common reference for each group
            group_references = {}
            for group_key, group_objects in groups.items():
                common_ref = self.find_common_reference_for_group(
                    group_objects, remaining_objects)
                group_references[group_key] = common_ref

            # Generate locDesc for remaining objects
            for obj in remaining_objects:
                # Find the group that the object belongs to
                obj_group = None
                common_ref = None
                for group_key, group_objects in groups.items():
                    if obj in group_objects:
                        obj_group = group_objects
                        common_ref = group_references[group_key]
                        break

                loc_desc = self.generate_unique_loc_desc(
                    obj, remaining_objects, used_names, obj_group, common_ref)
                obj['locDesc'] = loc_desc
                used_names.add(loc_desc)
