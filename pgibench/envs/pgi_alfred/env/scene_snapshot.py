#!/usr/bin/env python3
"""Scene snapshot + deterministic random initialization for task creation.

snapshot_scene() serializes the live AI2-THOR world state into the object_poses /
object_toggles / dirty_and_empty / init_action block that ThorEnv.restore_scene()
consumes (objectName uses the per-instance name (o['name']); SetObjectPoses matches
spawned pickupables by instance name, so each instance is restored individually).

random_initialize_scene() deterministically spawns the instruction objects plus
random distractors via InitialRandomSpawn, then snapshots the result.

Consumers: id_trajectory_generator (scripts/), compose-alfred-task skill.
"""

from typing import Dict, Optional
import random as _random

# Representative objects per scene type for RandomSpawn (migrated from position_provider.py).
# Membership rule: pickupable types with ~100% inherent coverage across that room's
# FloorPlans (verified via gen/layouts/FloorPlan*-objects.json). Sub-100% types are
# excluded to avoid InitialRandomSpawn silent failures — InitialRandomSpawn can only
# add instances of types already inherent to the chosen FloorPlan.
REPRESENTATIVE_OBJECTS = {
    "Kitchen": [
        "Apple", "Bread", "Cup", "Egg", "Fork", "Knife", "Mug", "Plate", "Pot", "Pan",
        "Bowl", "Spoon", "Tomato", "Lettuce", "Potato",
        "SaltShaker", "PepperShaker", "DishSponge", "SoapBottle", "Spatula", "ButterKnife",
    ],
    "LivingRoom": [
        "Book", "CellPhone", "Laptop", "Newspaper", "Pillow", "RemoteControl", "Vase",
        "Box", "CreditCard", "KeyChain",
    ],
    "Bedroom": [
        "Book", "CellPhone", "Laptop", "Newspaper", "Pillow", "Watch", "AlarmClock",
        "CD", "CreditCard", "KeyChain", "Pen", "Pencil",
    ],
    "Bathroom": [
        "Cloth", "DishSponge", "SoapBar", "SoapBottle", "SprayBottle", "TissueBox",
        "Candle", "Towel", "ToiletPaper", "HandTowel", "Plunger", "ScrubBrush",
    ],
}


def snapshot_scene(thor_env) -> Dict:
    """Capture the current world state into a restore_scene-compatible scene block.

    object_poses contains PICKUPABLE instances only (matched to spawned instances
    by o['name']); fixed furniture is NOT frozen here — it is FloorPlan-inherent
    and already correct after env.reset(FloorPlan). Including fixed furniture made
    SetObjectPoses return lastActionSuccess=False on restore (AI2-THOR refuses to
    re-pose non-pickupables). object_toggles still captures ALL toggleables,
    including fixed lamps (DeskLamp/FloorLamp) whose on/off state must be restored.
    """
    md = thor_env.last_event.metadata
    agent = md['agent']
    object_poses = []
    object_toggles = []
    for o in md['objects']:
        if not o.get('pickupable'):
            continue   # only freeze spawned pickupable instances; fixed furniture is FloorPlan-inherent
        object_poses.append({
            "objectName": o['name'],
            "position": {k: float(o['position'][k]) for k in ('x', 'y', 'z')},
            "rotation": {k: float(o['rotation'][k]) for k in ('x', 'y', 'z')},
        })
        if o.get('toggleable'):
            object_toggles.append({
                "objectType": o['objectType'],
                "isOn": bool(o.get('isToggled', False)),
            })
    # Fixed toggleables (e.g. DeskLamp/FloorLamp) were skipped above; capture them here.
    for o in md['objects']:
        if o.get('pickupable') and o.get('toggleable'):
            continue  # already added above
        if o.get('toggleable'):
            object_toggles.append({
                "objectType": o['objectType'],
                "isOn": bool(o.get('isToggled', False)),
            })
    init_action = {
        "action": "TeleportFull",
        "x": float(agent['position']['x']),
        "y": float(agent['position']['y']),
        "z": float(agent['position']['z']),
        "rotation": float(agent['rotation']['y']),
        "horizon": float(agent['cameraHorizon']),
        "standing": True,
    }
    return {
        "object_poses": object_poses,
        "object_toggles": object_toggles,
        "dirty_and_empty": False,
        "init_action": init_action,
    }


def _ensure_nav_graph(thor_env) -> None:
    """Build reachable-positions + path-points nav graph after a scene change.
    ThorConnector.restore_scene does this, but random_initialize_scene goes through
    ThorEnv.reset which skips it; without this, nav_obj_by_id hits a None KDTree."""
    if hasattr(thor_env, 'get_reachable_positions'):
        thor_env.reachable_positions, thor_env.reachable_position_kdtree = \
            thor_env.get_reachable_positions()
    if hasattr(thor_env, 'load_path_points'):
        thor_env.path_points, thor_env.path_points_kdtree = thor_env.load_path_points()


def sample_distractors(room_type: str, n: int, seed: int,
                       exclude: Optional[set] = None,
                       allow_repeat: bool = True) -> Dict[str, int]:
    """Pick n random irrelevant object types for the room (deterministic via seed).
    Returns {objectType: count} suitable for InitialRandomSpawn numRepeats.

    allow_repeat=True (default) samples WITH replacement, so the same type can
    appear multiple times (e.g. {Mug: 3}) — multiple look-alike instances raise
    the agent's discrimination difficulty. allow_repeat=False keeps the legacy
    at-most-one-per-type behavior."""
    exclude = exclude or set()
    pool = [o for o in REPRESENTATIVE_OBJECTS.get(room_type, []) if o not in exclude]
    rng = _random.Random(seed)
    if not pool:
        return {}
    if allow_repeat:
        chosen = rng.choices(pool, k=n)
    else:
        chosen = rng.sample(pool, min(n, len(pool)))
    counts: Dict[str, int] = {}
    for ot in chosen:
        counts[ot] = counts.get(ot, 0) + 1
    return counts


def inherent_pickupable_types(thor_env) -> set:
    """Pickupable object types inherently present in the current scene.

    InitialRandomSpawn can only add instances of types the FloorPlan already
    contains; non-inherent types spawn silently (returns lastActionSuccess=True but
    the object never appears). Use this to filter instruction objects up front.
    """
    return {o['objectType'] for o in thor_env.last_event.metadata['objects']
            if o.get('pickupable')}


# Scene-fixed facility object types that state-changing subtasks depend on.
_FACILITY_TYPES = (
    'Microwave', 'Fridge', 'SinkBasin', 'Faucet', 'DeskLamp', 'FloorLamp',
)


def layout_summary(thor_env) -> Dict:
    """Summarize the live scene for task design: where each pickupable landed,
    which state-changing facilities exist, and which receptacle types are
    present (with the lowest Y per type, so place-targets above the agent's
    reach can be filtered out before trajectory generation). Consumed by the
    agent (and the compose-alfred-task skill) after spawn to design subtasks +
    micro instruction.
    """
    md = thor_env.last_event.metadata
    objects = []
    for o in md['objects']:
        if not o.get('pickupable'):
            continue
        parent = o.get('parentReceptacles') or []
        objects.append({
            'name': o['name'],
            'type': o['objectType'],
            'receptacle': parent[0] if parent else None,
            'position': {k: float(o['position'][k]) for k in ('x', 'y', 'z')},
        })
    present_types = {o['objectType'] for o in md['objects']}
    facilities = {ft: (ft in present_types) for ft in _FACILITY_TYPES}
    # Receptacle types present in this FloorPlan, keyed to the lowest instance Y.
    # Put subtasks target a receptacle instance — agent standing height ~0.9,
    # so instances above ~1.05 are put-unreachable ("look_up" adjustment fails).
    receptacles = {}
    for o in md['objects']:
        if not o.get('receptacle') or o.get('pickupable'):
            continue
        ot = o['objectType']
        y = float(o['position']['y'])
        if ot not in receptacles or y < receptacles[ot]:
            receptacles[ot] = y
    return {'objects': objects, 'facilities': facilities,
            'receptacles': receptacles}


def random_initialize_scene(thor_env, scene_num: int, seed: int,
                            object_counts: Dict[str, int],
                            distractors: Optional[Dict[str, int]] = None,
                            min_free: Optional[Dict[str, int]] = None,
                            max_retries: int = 5) -> Dict:
    """Reset to FloorPlan<scene_num>, deterministically spawn object_counts
    (instruction objects) plus distractors via InitialRandomSpawn, then snapshot.
    Retries with seed+1 when the spawn reports failure. Returns the scene block.

    Note: passes through ThorEnv.step so the heated/cooled/cleaned sets and loc
    descs stay consistent; InitialRandomSpawn does not itself mutate those sets.
    """
    counts = dict(object_counts)
    if distractors:
        for ot, c in distractors.items():
            counts[ot] = counts.get(ot, 0) + c
    num_repeats = [{"objectType": ot, "count": c} for ot, c in counts.items() if c > 0]

    thor_env.reset(f"FloorPlan{scene_num}")

    event = None
    for attempt in range(max_retries):
        action = dict(action="InitialRandomSpawn",
                      randomSeed=seed + attempt,
                      forceVisible=True,
                      numRepeats=num_repeats)
        if min_free:
            action["minFreePerReceptacleType"] = [
                {"objectType": ot, "count": c} for ot, c in min_free.items()
            ]
        event = thor_env.step(action)
        if event.metadata.get("lastActionSuccess"):
            _ensure_nav_graph(thor_env)
            return snapshot_scene(thor_env)
    err = event.metadata.get('errorMessage', 'unknown') if event else 'no event'
    raise RuntimeError(
        f"InitialRandomSpawn failed in FloorPlan{scene_num} after {max_retries} seeds: {err}")


# Closed-container types where hiding an instruction object creates a search task.
# Drawer / Cabinet / Safe / Box are room furniture; Fridge/Microwave excluded
# (state-changing, would heat/cool the hidden object).
HIDDEN_CONTAINER_TYPES = ('Cabinet', 'Drawer', 'Safe', 'Box')


def spawn_into_container(thor_env, obj_type: str, container_type: str,
                         max_containers: int = 5) -> Optional[Dict]:
    """Force-place one instance of obj_type inside a closed container.

    Called AFTER random_initialize_scene (which spawns pickupables randomly).
    Finds an existing obj_type instance, opens a target container, picks the
    object up, puts it into the container, then closes it. Returns
    {'object_id':..., 'container_id':..., 'container_name':...} on success,
    None on failure (no instance / no container / put rejected).

    Used by search-and-stow / search-use-later archetypes so the agent must
    OPEN a closed receptacle to find the hidden object during verification.
    """
    md = thor_env.last_event.metadata
    cands = [o for o in md['objects']
             if o['objectType'] == obj_type and o.get('pickupable')
             and not o.get('isHeld')]
    containers = [o for o in md['objects']
                  if o['objectType'] == container_type and o.get('openable')
                  and o.get('receptacle')]
    if not cands or not containers:
        return None

    # forceAction alone doesn't bypass OpenObject's distance/orientation
    # check — teleport the agent near each container (and facing it) first.
    thor_env.step({'action': 'GetReachablePositions'})
    reachable = thor_env.last_event.metadata.get('actionReturn', [])

    def _teleport_near(pos):
        import math
        if not reachable:
            return
        closest = min(reachable, key=lambda p: (p['x']-pos['x'])**2 + (p['z']-pos['z'])**2)
        dx, dz = pos['x']-closest['x'], pos['z']-closest['z']
        rot = (90 - math.degrees(math.atan2(dz, dx))) % 360
        thor_env.step({'action': 'TeleportFull',
                       'x': closest['x'], 'y': closest['y'], 'z': closest['z'],
                       'rotation': rot, 'horizon': 0, 'standing': True})

    cand = cands[0]
    for cont in containers[:max_containers]:
        cid = cont['objectId']
        _teleport_near(cont['position'])
        if not cont.get('isOpen'):
            thor_env.step({'action': 'OpenObject', 'objectId': cid,
                           'forceAction': True})
            if not thor_env.last_event.metadata.get('lastActionSuccess'):
                continue
        # pick + put into container
        thor_env.step({'action': 'PickupObject', 'objectId': cand['objectId'],
                       'forceAction': True})
        if not thor_env.last_event.metadata.get('lastActionSuccess'):
            thor_env.step({'action': 'CloseObject', 'objectId': cid,
                           'forceAction': True})
            continue
        thor_env.step({'action': 'PutObject', 'objectId': cand['objectId'],
                       'receptacleObjectId': cid, 'forceAction': True})
        put_ok = thor_env.last_event.metadata.get('lastActionSuccess')
        thor_env.step({'action': 'CloseObject', 'objectId': cid,
                       'forceAction': True})
        if put_ok:
            return {'object_id': cand['objectId'],
                    'container_id': cid,
                    'container_name': cont['name']}
    return None


# --- Curated scene positions: scatter objects at verified poses -------------
#
# Why this exists: InitialRandomSpawn only places objects ON receptacles — it
# cannot scatter them on the FLOOR or at arbitrary verified spots. The
# scene_positions/FloorPlan<N>.json dataset holds every object pose ever seen
# in an existing ALFRED task for that FloorPlan (collected from executed
# trajectories), so each pose is already proven reachable/placeable. We read it
# to place objects at verified positions — including floor-level (y<0.3) poses
# that give "pick up from the floor" semantics without any physics simulation.
#
# Scene-appropriateness is inherent: the dataset is per-FloorPlan, so a Bathroom
# scene only offers poses for objects that actually appeared in bathroom tasks.
# No vegetables appear on a bedroom floor because no bedroom task ever placed
# them there. Cross-type matching (e.g. Mug in Bedroom) works when the dataset
# has the pose, regardless of InitialRandomSpawn's inherent-type restriction.

import os as _os
# scene_snapshot.py is at pgibench/envs/pgi_alfred/env/; scene_positions lives
# under pgibench/envs/pgi_alfred/data/scene_positions/ (2 dirnames up from __file__).
_SCENE_POSITIONS_DIR = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    'data', 'scene_positions')


def load_scene_positions(scene_num):
    """Load the curated pose dataset for FloorPlan<scene_num>.

    Returns {objectType: [{objectName, position, rotation}, ...], 'scene_type'}.
    objectName is the BARE type (e.g. 'Apple'), matching the legacy object_poses
    convention — SetObjectPoses accepts bare-type names via type matching. Each
    pose came from an executed ALFRED trajectory, so it is already proven
    valid for that FloorPlan. Returns None if the file is absent.
    """
    path = _os.path.join(_SCENE_POSITIONS_DIR, f'FloorPlan{scene_num}.json')
    if not _os.path.exists(path):
        return None
    with open(path) as f:
        import json as _json
        return _json.load(f)


def sample_scene_pose(scene_positions, otype, rng, floor_only=False):
    """Pick one verified pose for otype from the curated dataset.

    floor_only=True restricts to floor-level poses (y<0.3) — use this for
    "scattered on the floor" semantics. Returns the pose dict (objectName with
    bare type, position, rotation) or None if no matching pose exists.
    """
    poses = scene_positions.get(otype) or []
    if floor_only:
        poses = [p for p in poses if p['position']['y'] < 0.3]
    return rng.choice(poses) if poses else None


def scatter_from_positions(thor_env, specs, seed):
    """Place objects at curated verified poses via SetObjectPoses.

    Called AFTER random_initialize_scene (which spawned inherent objects on
    receptacles). specs is a list of {objectType, count, floor_only: bool}.
    For each spec, picks `count` verified poses from scene_positions and
    SetObjectPoses-matches them onto existing instances of that type. Objects
    NOT yet in the scene (neither inherent nor spawned) are skipped with a
    warning — SetObjectPoses can only re-pose existing instances.

    Returns (placed_poses, skipped) where placed_poses is the list of pose
    dicts actually applied (objectName here uses the live instance name) and
    skipped lists {objectType, reason} entries.
    """
    import random as _r
    rng = _r.Random(seed)
    # sceneName is like 'FloorPlan1' or 'FloorPlan1_physics' on physics builds
    sn_meta = thor_env.last_event.metadata.get('sceneName')
    scene_num = None
    if isinstance(sn_meta, str) and sn_meta.startswith('FloorPlan'):
        scene_num = int(sn_meta[len('FloorPlan'):].split('_')[0])
    sp = load_scene_positions(scene_num)
    if sp is None:
        return [], [{'reason': f'no scene_positions for FloorPlan{scene_num}'}]

    placed, skipped = [], []
    for spec in specs:
        otype = spec['objectType']
        count = int(spec.get('count', 1))
        floor_only = bool(spec.get('floor_only', False))
        instances = [o for o in thor_env.last_event.metadata['objects']
                     if o['objectType'] == otype and o.get('pickupable')]
        if not instances:
            skipped.append({'objectType': otype,
                            'reason': 'no live instance to re-pose'})
            continue
        poses = []
        for _ in range(count):
            p = sample_scene_pose(sp, otype, rng, floor_only=floor_only)
            if p is None:
                continue
            poses.append(p)
        if not poses:
            skipped.append({'objectType': otype,
                            'reason': 'no curated pose' +
                                      (' (floor_only)' if floor_only else '')})
            continue
        # apply: re-pose up to len(instances) objects; if more poses than
        # instances, extras are dropped (can't clone via SetObjectPoses)
        object_poses = []
        for p, inst in zip(poses, instances):
            object_poses.append({
                'objectName': inst['name'],
                'position': {k: float(p['position'][k]) for k in ('x', 'y', 'z')},
                'rotation': {k: float(p['rotation'][k]) for k in ('x', 'y', 'z')},
            })
            placed.append({
                'objectName': inst['name'], 'objectType': otype,
                'position': object_poses[-1]['position'],
                'floor_only': floor_only,
            })
        if object_poses:
            thor_env.step({'action': 'SetObjectPoses',
                           'objectPoses': object_poses})
    return placed, skipped
