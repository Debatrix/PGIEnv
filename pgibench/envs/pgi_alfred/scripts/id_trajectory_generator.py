#!/usr/bin/env python3
"""Object-id driven trajectory generator + feasibility verifier.

Replaces the trajectory-splicing composite_generator. Given a task
(task.yaml), start AI2-THOR, restore the scene, then drive each subtask via
ThorConnector's _by_id primitives (nav_obj_by_id + pick/put/open/close/toggleon/
...). Each subtask is ~"navigate + interact" high-level steps, with no atomic
navigation expansion — cheap, and enough to verify solvability.

State sets (heated/cooled/cleaned) stay correct because the _by_id primitives now
call _sync_state (BUG#5 fix), which mirrors ThorEnv.step's bookkeeping.

After execution, CompositeTask.goal_satisfied / goal_conditions_met decide
feasibility. Feasible tasks get a trajectory.json (exec log + placeholder plan).

Requires X server: `python -m pgibench.envs.pgi_alfred.scripts.startx 1`
Run:
    python -m pgibench.envs.pgi_alfred.scripts.id_trajectory_generator --task-dir <dir>
"""

import os
import json
import argparse

# op -> ThorConnector method
_BY_ID = {
    'pick': 'pick_by_id', 'put': 'put_by_id', 'drop': 'drop',
    'open': 'open_by_id', 'close': 'close_by_id',
    'toggleon': 'toggleon_by_id', 'toggleoff': 'toggleoff_by_id',
    'slice': 'slice_by_id',
}

# Ops where small movement adjustments can recover a failed interaction after nav.
# 'put' is excluded from raw-retry: the held object would be dropped by movement
# in some scenes; put_by_id already has its own 7-step movement retry built in.
_RETRYABLE_OPS = {'pick', 'toggleon', 'toggleoff', 'open', 'close', 'slice', 'put'}

# Raw AI2-THOR action name for each op (direct re-issue after movement, no nav).
_RAW_ACTION = {
    'pick': 'PickupObject', 'put': 'PutObject',
    'toggleon': 'ToggleObjectOn', 'toggleoff': 'ToggleObjectOff',
    'open': 'OpenObject', 'close': 'CloseObject',
    'slice': 'SliceObject',
}

# Small movement sequences tried when the initial nav+interact fails.
# Single-step perturbations first, then a few useful combinations. These nudge
# the agent a grid step / rotation / look angle away from the teleported spot
# so the target re-enters the reachable/visible fringe.
#
# Total capped at 25 (per design constraint). The last 6 entries target
# DeskLamp toggle (needs precise look_up angle) and small-object pick
# (CreditCard/CellPhone often hidden behind furniture).
_MOVEMENT_ADJUSTMENTS = [
    ('look_up',),
    ('look_down',),
    ('rotate_left',),
    ('rotate_right',),
    ('move_ahead',),
    ('move_back',),
    ('move_left',),
    ('move_right',),
    ('look_up', 'look_up'),
    ('look_down', 'look_down'),
    ('rotate_left', 'move_ahead'),
    ('rotate_right', 'move_ahead'),
    ('move_ahead', 'rotate_left'),
    ('move_ahead', 'rotate_right'),
    ('move_back', 'look_up'),
    ('rotate_left', 'rotate_left'),
    ('rotate_right', 'rotate_right'),
    # New: aggressive view-angle combos for DeskLamp toggle failures.
    ('look_up', 'look_up', 'look_up'),
    ('look_up', 'rotate_left', 'look_up'),
    ('look_up', 'rotate_right', 'look_up'),
    # New: dense search for small-object pick failures.
    ('move_ahead', 'move_left', 'rotate_left'),
    ('move_ahead', 'move_right', 'rotate_right'),
    ('rotate_left', 'move_ahead', 'look_up'),
    ('rotate_right', 'move_ahead', 'look_up'),
    ('move_back', 'rotate_left', 'look_up'),
]

# recording context for frame + simulator-log capture during verification
_REC = {'frames_dir': None, 'sim_log': None, 'frame_idx': 0}


def _snapshot_rec(thor_env, op, obj_id, ret, idx, log):
    """Append an exec_log entry, save a frame, and append a simulator-log entry."""
    ok = ret == ''
    log.append({'subtask': idx, 'op': op, 'target': obj_id,
                'ok': ok, 'err': ret if ret else ''})
    if _REC['frames_dir']:
        frame = getattr(thor_env.last_event, 'frame', None)
        if frame is not None:
            fn = os.path.join(_REC['frames_dir'],
                              f"{_REC['frame_idx']:04d}_sub{idx}_{op}.png")
            try:
                import numpy as np
                from PIL import Image
                if isinstance(frame, np.ndarray):
                    Image.fromarray(frame).save(fn)
                else:
                    frame.save(fn)
            except Exception:
                pass
    if _REC['sim_log'] is not None:
        md = thor_env.last_event.metadata
        _REC['sim_log'].append({
            'step': _REC['frame_idx'], 'subtask': idx, 'op': op, 'target': obj_id,
            'ok': ok, 'err': ret if ret else '',
            'last_action': md.get('lastAction'),
            'last_action_success': md.get('lastActionSuccess'),
            'error_message': md.get('errorMessage', ''),
            'agent_position': md['agent']['position'],
            'agent_rotation': md['agent']['rotation']['y'],
            'held_object_types': [o['objectType'] for o in md.get('inventoryObjects', [])],
        })
    _REC['frame_idx'] += 1


# ---- object resolution helpers ------------------------------------------------

def _find_obj_id(thor_env, obj_type, exclude=None):
    """First objectId of a given type (prefer pickupable/visible), excluding a set."""
    exclude = exclude or set()
    objs = [o for o in thor_env.last_event.metadata['objects']
            if o['objectType'] == obj_type and o['objectId'] not in exclude]
    for pref in (lambda o: o.get('pickupable') and o.get('visible'),
                 lambda o: o.get('pickupable'),
                 lambda o: True):
        for o in objs:
            if pref(o):
                return o['objectId']
    return None


def _find_recep_id(thor_env, recep_type):
    """A receptacle of the given type (prefer visible)."""
    md = thor_env.last_event.metadata['objects']
    for o in md:
        if o['objectType'] == recep_type and o.get('receptacle') and o.get('visible'):
            return o['objectId']
    for o in md:
        if o['objectType'] == recep_type and o.get('receptacle'):
            return o['objectId']
    return None


def _open_container_if_closed(thor_env, obj_id, log, idx):
    """If obj_id sits in a closed openable receptacle, open it first (so it can be picked)."""
    o = next((x for x in thor_env.last_event.metadata['objects']
              if x['objectId'] == obj_id), None)
    if not o:
        return
    for pid in (o.get('parentReceptacles') or []):
        p = next((x for x in thor_env.last_event.metadata['objects']
                  if x['objectId'] == pid), None)
        if p and p.get('openable') and not p.get('isOpen'):
            _do(thor_env, 'open', pid, idx, log)
            break


def _basic_move(thor_env, name):
    """Execute one basic movement by short name. Goes through ThorEnv.step so state
    bookkeeping (update_states / check_post_conditions) stays consistent."""
    if name in ('look_up', 'look_down'):
        # ThorConnector exposes look_up/look_down with an angle arg; use a small
        # 30-degree nudge (the AI2-THOR default horizon step).
        getattr(thor_env, name)(30)
        return
    action_map = {
        'move_ahead': 'MoveAhead', 'move_back': 'MoveBack',
        'move_left': 'MoveLeft', 'move_right': 'MoveRight',
        'rotate_left': 'RotateLeft', 'rotate_right': 'RotateRight',
    }
    thor_env.step({'action': action_map[name]})


def _raw_interact(thor_env, op, obj_id):
    """Directly issue the AI2-THOR interaction action via ThorEnv.step (so
    update_states / check_post_conditions run, e.g. faucet->clean sync). No nav,
    no visibility gate, forceAction=True. Returns '' on success, else a message."""
    if op == 'put':
        inv = thor_env.last_event.metadata['inventoryObjects']
        if not inv:
            return 'not holding any object'
        payload = {'action': 'PutObject', 'forceAction': True,
                   'objectId': inv[0]['objectId'],
                   'receptacleObjectId': obj_id}
    else:
        payload = {'action': _RAW_ACTION[op], 'objectId': obj_id,
                   'forceAction': True}
    thor_env.step(payload)
    if thor_env.last_event.metadata['lastActionSuccess']:
        return ''
    return (thor_env.last_event.metadata.get('errorMessage', '').strip()
            or f"{payload['action']} failed")


# Error substrings that movement cannot recover from (receptacle full, object
# gone, already-in-target-state). When the raw retry returns one of these we
# stop perturbing — more movement is just wasted sim steps.
_UNRECOVERABLE = (
    'No valid Receptacle found',
    'No valid positions to place object found',
    'already on',
    'already off',
    'is already open',
    'is already closed',
    'not holding',
    'is not pickupable',
    'is not toggleable',
    'is not openable',
)


def _force_clear_one_obstacle(thor_env, recep_id, idx, log, exclude_types=None):
    """Force-move one pickupable obstacle off the receptacle via SetObjectPoses.

    Last-resort fallback for put failures when min_free reservation didn't take
    effect (rare): find a pickupable sitting on the target receptacle, teleport
    it to the floor near the agent. Returns cleared objectId on success, else None.
    Only clears ONE obstacle to avoid runaway side effects.

    exclude_types: objectTypes that must NOT be moved (instruction objects placed
    by earlier subtasks — without this, clearing would silently undo completed
    subtask state and cause later resolve to fail with 'object not found')."""
    exclude_types = exclude_types or set()
    md = thor_env.last_event.metadata
    inv = md.get('inventoryObjects', [])
    held_id = inv[0]['objectId'] if inv else None
    recep = next((o for o in md['objects'] if o['objectId'] == recep_id), None)
    if recep is None:
        return None
    obstacles = [o for o in md['objects']
                 if o['objectId'] != held_id and o.get('pickupable')
                 and o['objectType'] not in exclude_types
                 and recep_id in (o.get('parentReceptacles') or [])]
    if not obstacles:
        return None
    obs = obstacles[0]
    agent_pos = md['agent']['position']
    thor_env.step({
        'action': 'SetObjectPoses',
        'objectPoses': [{
            'objectName': obs['name'],
            'position': {'x': float(agent_pos['x']) + 0.5,
                         'y': 0.0,
                         'z': float(agent_pos['z']) + 0.5},
            'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        }],
    })
    ok = thor_env.last_event.metadata.get('lastActionSuccess', False)
    _snapshot_rec(thor_env, 'clear_obstacle', obs['objectId'],
                  'moved to floor' if ok else 'set_poses_failed', idx, log)
    return obs['objectId'] if ok else None


def _retry_with_movement(thor_env, op, obj_id, idx, log, prev_ret):
    """After nav+_by_id failed, try small basic movements from the current (near-
    target) position and directly re-issue the raw AI2-THOR action. Does NOT
    re-navigate, so the adjustment is preserved against the target. Returns '' on
    success or the last error message. Aborts early on unrecoverable errors
    (receptacle full, already-in-state) where movement cannot help."""
    last_ret = prev_ret
    for adj in _MOVEMENT_ADJUSTMENTS:
        # Skip remaining adjustments if the last error is unrecoverable.
        if any(s in last_ret for s in _UNRECOVERABLE):
            return last_ret
        for mv in adj:
            _basic_move(thor_env, mv)
        _snapshot_rec(thor_env, 'adjust', obj_id, '+'.join(adj), idx, log)
        ret = _raw_interact(thor_env, op, obj_id)
        _snapshot_rec(thor_env, op, obj_id, ret, idx, log)
        if not ret:
            return ''
        last_ret = ret
    return last_ret


def _do(thor_env, op, obj_id, idx, log):
    """Navigate to the target, then invoke a _by_id primitive. Each step is recorded
    (exec_log + optional frame + simulator log). The _by_id primitives require the
    object to be visible (they do not nav themselves), so we nav_obj_by_id first.

    If nav succeeded but the interaction failed (object not visible from the
    teleported spot, or the raw action collided), retry with small basic movement
    adjustments and directly re-issue the raw AI2-THOR action — see
    _retry_with_movement. For put failures on cluttered receptacles, first try
    force-clearing one obstacle via SetObjectPoses (_force_clear_one_obstacle)."""
    if op != 'drop' and obj_id is None:
        # A None id means the object/receptacle could not be resolved in the
        # current scene (e.g. a seed-specific spawn/restore race). Log it as a
        # clean failure instead of crashing the whole verification run.
        _snapshot_rec(thor_env, op, None, 'object/receptacle id resolved to None', idx, log)
        return 'object/receptacle id resolved to None'
    if obj_id is not None and op != 'drop':
        nav_ret = thor_env.nav_obj_by_id(obj_id)
        _snapshot_rec(thor_env, 'nav', obj_id, nav_ret, idx, log)
    method = getattr(thor_env, _BY_ID[op])
    ret = method(obj_id) if obj_id is not None else method()
    _snapshot_rec(thor_env, op, obj_id, ret, idx, log)

    # put-specific fallback: clear one obstacle from the receptacle before
    # falling into movement-based retry. min_free reservation handles most
    # cases at spawn time; this catches the rare cluttered-recep residual.
    # Exclude instruction object types so we don't move completed-subtask state.
    if ret and op == 'put' and obj_id is not None:
        inst_types = set()
        task = getattr(thor_env, 'task', None)
        if task:
            for st in task.traj.get('pddl_params', {}).get('subtasks', []):
                obj = st.get('pddl_params', {}).get('object_target')
                if obj:
                    inst_types.add(obj)
        cleared = _force_clear_one_obstacle(thor_env, obj_id, idx, log,
                                            exclude_types=inst_types)
        if cleared is not None:
            ret = _raw_interact(thor_env, 'put', obj_id)
            _snapshot_rec(thor_env, op, obj_id, ret, idx, log)

    if ret and obj_id is not None and op in _RETRYABLE_OPS:
        ret = _retry_with_movement(thor_env, op, obj_id, idx, log, ret)
    return ret


def _nearest_sink_for_faucet(md, faucet):
    """Same decision as game_util.get_obj_of_type_closest_to_obj('SinkBasin', faucet):
    the SinkBasin with minimal Euclidean distance to this faucet. Mirrored here so
    the generator can predict which sink a faucet will clean without stepping."""
    sinks = [o for o in md['objects'] if o['objectType'] == 'SinkBasin']
    if not sinks:
        return None
    fp = faucet['position']
    return min(sinks, key=lambda s: sum(
        (s['position'][k] - fp[k]) ** 2 for k in ('x', 'y', 'z')) ** 0.5)


def _try_clean_in_sink(thor_env, sink_id, obj_id, idx, log):
    """Toggle faucets in priority order until obj_id enters thor_env.cleaned_objects.

    thor_env.update_states marks cleaned the contents of the SinkBasin nearest to the
    toggled faucet. So a faucet whose nearest sink is NOT `sink_id` will never clean
    `obj_id`. We therefore try faucets whose nearest sink IS sink_id first (sorted by
    distance to that sink), then fall back to all other faucets. After each toggle
    cycle we check cleaned_objects and short-circuit on success. Returns True iff the
    object got cleaned."""
    if obj_id in thor_env.cleaned_objects:
        return True
    md = thor_env.last_event.metadata
    sink = next((o for o in md['objects'] if o['objectId'] == sink_id), None)
    if sink is None:
        return False

    def _dist_to_sink(f):
        return sum((sink['position'][k] - f['position'][k]) ** 2
                   for k in ('x', 'y', 'z')) ** 0.5

    primary, secondary = [], []
    for f in (o for o in md['objects'] if o['objectType'] == 'Faucet'):
        nearest = _nearest_sink_for_faucet(md, f)
        entry = (f['objectId'], _dist_to_sink(f))
        (primary if nearest is not None and nearest['objectId'] == sink_id
         else secondary).append(entry)
    primary.sort(key=lambda e: e[1])
    secondary.sort(key=lambda e: e[1])

    for fid, _ in primary + secondary:
        _do(thor_env, 'toggleon', fid, idx, log)
        _do(thor_env, 'toggleoff', fid, idx, log)
        if obj_id in thor_env.cleaned_objects:
            return True
    return False


# ---- per-task-type execution -------------------------------------------------

def _exec_subtask(thor_env, task_type, pp, idx, log):
    obj = pp.get('object_target') or None
    parent = pp.get('parent_target') or None
    toggle = pp.get('toggle_target') or None
    mrecep = pp.get('mrecep_target') or None
    sliced = bool(pp.get('object_sliced'))

    def pick_and_place(obj_type, recep_type, exclude=None):
        oid = _find_obj_id(thor_env, obj_type, exclude=exclude)
        if oid is None:
            log.append({'subtask': idx, 'op': 'resolve', 'target': obj_type,
                        'ok': False, 'err': 'object not found'})
            return
        _open_container_if_closed(thor_env, oid, log, idx)
        if sliced:
            # slice on the surface (AI2-THOR rejects slicing a held object),
            # then pick the resulting <obj>Sliced piece and place it.
            _do(thor_env, 'slice', oid, idx, log)
            sid = _find_obj_id(thor_env, obj_type + 'Sliced')
            if sid is None:
                log.append({'subtask': idx, 'op': 'resolve',
                            'target': obj_type + 'Sliced',
                            'ok': False, 'err': 'sliced object not found'})
                return
            _do(thor_env, 'pick', sid, idx, log)
        else:
            _do(thor_env, 'pick', oid, idx, log)
        rid = _find_recep_id(thor_env, recep_type)
        if rid is None:
            log.append({'subtask': idx, 'op': 'resolve', 'target': recep_type,
                        'ok': False, 'err': 'receptacle not found'})
            return
        _do(thor_env, 'put', rid, idx, log)

    if task_type == 'pick_and_place_simple':
        pick_and_place(obj, parent)

    elif task_type == 'pick_two_obj_and_place':
        placed = set()
        for _ in range(2):
            oid = _find_obj_id(thor_env, obj, exclude=placed)
            if oid is None:
                break
            _open_container_if_closed(thor_env, oid, log, idx)
            _do(thor_env, 'pick', oid, idx, log)
            rid = _find_recep_id(thor_env, parent)
            _do(thor_env, 'put', rid, idx, log)
            placed.add(oid)

    elif task_type == 'look_at_obj_in_light':
        oid = _find_obj_id(thor_env, obj)
        _open_container_if_closed(thor_env, oid, log, idx)
        _do(thor_env, 'pick', oid, idx, log)
        lid = _find_obj_id(thor_env, toggle)
        _do(thor_env, 'toggleon', lid, idx, log)

    elif task_type == 'pick_heat_then_place_in_recep':
        oid = _find_obj_id(thor_env, obj)
        _open_container_if_closed(thor_env, oid, log, idx)
        _do(thor_env, 'pick', oid, idx, log)
        mw = _find_obj_id(thor_env, 'Microwave')
        _do(thor_env, 'open', mw, idx, log)
        _do(thor_env, 'put', mw, idx, log)
        _do(thor_env, 'close', mw, idx, log)
        _do(thor_env, 'toggleon', mw, idx, log)   # heats contents (_sync_state)
        _do(thor_env, 'toggleoff', mw, idx, log)
        if oid not in thor_env.heated_objects:
            # One microwave per kitchen — no fallback. Surface the failure so the
            # verify report shows 'heat not applied' rather than a cryptic put/nav error.
            log.append({'subtask': idx, 'op': 'heat', 'target': obj,
                        'ok': False, 'err': 'object not heated after microwave cycle'})
        _do(thor_env, 'open', mw, idx, log)
        _do(thor_env, 'pick', oid, idx, log)      # take out (heated set keeps it)
        _do(thor_env, 'close', mw, idx, log)
        rid = _find_recep_id(thor_env, parent)
        _do(thor_env, 'put', rid, idx, log)

    elif task_type == 'pick_cool_then_place_in_recep':
        oid = _find_obj_id(thor_env, obj)
        _open_container_if_closed(thor_env, oid, log, idx)
        _do(thor_env, 'pick', oid, idx, log)
        fr = _find_obj_id(thor_env, 'Fridge')
        _do(thor_env, 'open', fr, idx, log)
        _do(thor_env, 'put', fr, idx, log)
        _do(thor_env, 'close', fr, idx, log)      # cools contents (_sync_state)
        if oid not in thor_env.cooled_objects:
            log.append({'subtask': idx, 'op': 'cool', 'target': obj,
                        'ok': False, 'err': 'object not cooled after fridge cycle'})
        _do(thor_env, 'open', fr, idx, log)
        _do(thor_env, 'pick', oid, idx, log)
        rid = _find_recep_id(thor_env, parent)
        _do(thor_env, 'put', rid, idx, log)

    elif task_type == 'pick_clean_then_place_in_recep':
        oid = _find_obj_id(thor_env, obj)
        _open_container_if_closed(thor_env, oid, log, idx)
        _do(thor_env, 'pick', oid, idx, log)
        sink = _find_obj_id(thor_env, 'SinkBasin')
        put_err = _do(thor_env, 'put', sink, idx, log)
        if put_err:
            # SinkBasin is one of AI2-THOR's unreliable put targets; if the put
            # never landed, the basin is empty, no faucet can clean the object,
            # and the post-clean pick would try to grab an object we still hold
            # (cascading 20+ useless retries). Short-circuit: skip clean + pick.
            log.append({'subtask': idx, 'op': 'clean', 'target': obj,
                        'ok': False, 'err': f'put-into-sink failed: {put_err}'})
        else:
            # _try_clean_in_sink prefers faucets whose nearest SinkBasin is the one
            # we just put the object into, falling back to all others. Without this,
            # a multi-sink scene would silently fail because update_states() only
            # cleans the sink nearest to the toggled faucet.
            if not _try_clean_in_sink(thor_env, sink, oid, idx, log):
                log.append({'subtask': idx, 'op': 'clean', 'target': obj,
                            'ok': False, 'err': 'object not cleaned after all faucets'})
            # object is sitting in the sink — pick it back up before final place
            _do(thor_env, 'pick', oid, idx, log)
        rid = _find_recep_id(thor_env, parent)
        _do(thor_env, 'put', rid, idx, log)

    elif task_type == 'pick_and_place_with_movable_recep':
        mid = _find_obj_id(thor_env, mrecep)
        _open_container_if_closed(thor_env, mid, log, idx)
        rid = _find_recep_id(thor_env, parent)
        # Carry the movable receptacle to the open parent surface FIRST, then
        # put the object into it. AI2-THOR cannot reliably place an object into
        # a movable receptacle that is still sitting inside a (possibly closed)
        # spawn container — the put returns "receptacle is open or not".
        _do(thor_env, 'pick', mid, idx, log)          # pick mrecep
        _do(thor_env, 'put', rid, idx, log)           # mrecep -> parent (open)
        oid = _find_obj_id(thor_env, obj)
        _open_container_if_closed(thor_env, oid, log, idx)
        _do(thor_env, 'pick', oid, idx, log)          # pick object
        _do(thor_env, 'put', mid, idx, log)           # object -> mrecep (on parent)

    elif task_type == 'pick_slice_then_place_in_recep':
        # Slice the object on the surface (AI2-THOR rejects slicing a held
        # object), then pick the resulting <obj>Sliced piece and place it.
        oid = _find_obj_id(thor_env, obj)
        if oid is None:
            log.append({'subtask': idx, 'op': 'resolve', 'target': obj,
                        'ok': False, 'err': 'object not found'})
            return
        _open_container_if_closed(thor_env, oid, log, idx)
        _do(thor_env, 'slice', oid, idx, log)
        sid = _find_obj_id(thor_env, obj + 'Sliced')
        if sid is None:
            log.append({'subtask': idx, 'op': 'resolve', 'target': obj + 'Sliced',
                        'ok': False, 'err': 'sliced object not found'})
            return
        _do(thor_env, 'pick', sid, idx, log)
        rid = _find_recep_id(thor_env, parent)
        if rid is None:
            log.append({'subtask': idx, 'op': 'resolve', 'target': parent,
                        'ok': False, 'err': 'receptacle not found'})
            return
        _do(thor_env, 'put', rid, idx, log)

    elif task_type == 'toggle_object_on':
        lid = _find_obj_id(thor_env, toggle)
        _do(thor_env, 'toggleon', lid, idx, log)
    elif task_type == 'toggle_object_off':
        lid = _find_obj_id(thor_env, toggle)
        _do(thor_env, 'toggleoff', lid, idx, log)
    elif task_type == 'open_receptacle':
        rid = _find_recep_id(thor_env, parent)
        _do(thor_env, 'open', rid, idx, log)
    elif task_type == 'close_receptacle':
        rid = _find_recep_id(thor_env, parent)
        _do(thor_env, 'close', rid, idx, log)
    elif task_type == 'slice_in_place':
        oid = _find_obj_id(thor_env, obj)
        _do(thor_env, 'slice', oid, idx, log)
    else:
        log.append({'subtask': idx, 'op': 'dispatch', 'target': task_type,
                    'ok': False, 'err': 'unknown task_type'})


# ---- main verify loop --------------------------------------------------------

def generate_and_verify(thor_env, frames_dir=None):
    """Drive every subtask, then read CompositeTask goal state.
    `thor_env` must already have its scene restored and task set.
    If frames_dir is set, save a frame + simulator-log entry per step into _REC.
    Returns a report dict."""
    _REC['frames_dir'] = frames_dir
    _REC['sim_log'] = []
    _REC['frame_idx'] = 0
    task = thor_env.task
    subtasks = task.traj['pddl_params']['subtasks']
    exec_log = []
    for i, st in enumerate(subtasks):
        # Ensure the hand is empty before each subtask: a failed put in a prior
        # subtask can leave an object held, cascading into "already holding" errors
        # that would silently fail every later subtask.
        _held_id, _ = thor_env.get_held_object()
        if _held_id is not None:
            _do(thor_env, 'drop', None, i, exec_log)
        _exec_subtask(thor_env, st['task_type'], st['pddl_params'], i, exec_log)
        # per-subtask goal snapshot
        s_i, ts_i = task._subtasks[i].goal_conditions_met(thor_env.last_event)
        exec_log.append({'subtask': i, 'task_type': st['task_type'],
                         'goal_s': s_i, 'goal_ts': ts_i,
                         'ok': s_i == ts_i})

    satisfied = thor_env.get_goal_satisfied()
    met, total = thor_env.get_goal_conditions_met()
    feasible = bool(satisfied) and total > 0 and met == total
    return {
        'feasible': feasible,
        'success': bool(satisfied),
        'goal_conditions_met': f'{met}/{total}',
        'num_subtasks': len(subtasks),
        'exec_log': exec_log,
        'sim_log': _REC['sim_log'],
    }


def _start_env_for_task(traj_data, resolution=300, x_display='1'):
    """Boot a ThorConnector, restore the scene, set the task. Returns the connector."""
    from pgibench.envs.pgi_alfred.thor_connector import ThorConnector
    from pgibench.envs.pgi_alfred.PGIAlfEnv import ALFRED_REWARD_PATH
    from pgibench.envs.pgi_alfred import utils

    scene = traj_data['scene']
    env = ThorConnector(x_display=x_display,
                        player_screen_height=resolution,
                        player_screen_width=resolution)
    rs = scene.get('random_spawn')
    if rs:
        # newly composed task: rebuild the spawned layout deterministically
        from pgibench.envs.pgi_alfred.env.scene_snapshot import random_initialize_scene
        random_initialize_scene(env, scene['scene_num'], rs['seed'],
                                rs['object_counts'], rs.get('distractors'),
                                min_free=rs.get('min_free'))
        # two-stage restore: force each spawned instance to its frozen pose
        if scene.get('object_poses'):
            env.step({'action': 'SetObjectPoses', 'objectPoses': scene['object_poses']})
    else:
        # legacy migrated task: restore exact object_poses
        env.reset(scene['floor_plan'])
        env.restore_scene(scene['object_poses'], scene['object_toggles'],
                          scene['dirty_and_empty'])
        init_action = dict(scene['init_action'])
        if init_action.get('action') == 'TeleportFull':
            init_action.pop('rotateOnTeleport', None)
            init_action['standing'] = True
        env.step(init_action)
    args = utils.dotdict({
        'pframe': 300, 'use_templated_goals': False,
        'reward_config': ALFRED_REWARD_PATH, 'max_steps': 1000,
        'pp_folder': 'pp', 'dout': 'exp/model', 'fast_epoch': False,
    })
    env.set_task(traj_data, args, reward_type='dense', max_episode_length=100)
    return env


def run(task_dir, out_dir='running/composite_verify', resolution=300, x_display='1',
        write_trajectory=True, save_frames=False):
    """Load a task, verify it in AI2-THOR. Writes report.json +
    sim_log.json (+ frames/ subdir when save_frames) under <out_dir>/<name>/."""
    from pgibench.envs.pgi_alfred.env.task_format import load_task_dir

    traj = load_task_dir(task_dir)
    name = traj.get('task_id') or os.path.basename(task_dir.rstrip('/'))
    task_out = os.path.join(out_dir, name)
    os.makedirs(task_out, exist_ok=True)
    frames_dir = os.path.join(task_out, 'frames') if save_frames else None
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    env = _start_env_for_task(traj, resolution=resolution, x_display=x_display)
    try:
        report = generate_and_verify(env, frames_dir=frames_dir)
    finally:
        try:
            env.stop()
        except Exception:
            pass

    report['task_id'] = traj['task_id']
    report['task_dir'] = task_dir

    with open(os.path.join(task_out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(task_out, 'sim_log.json'), 'w') as f:
        json.dump(report.get('sim_log', []), f, indent=2, ensure_ascii=False)

    if write_trajectory and report['feasible']:
        trajectory = {
            'exec_log': report['exec_log'],
            'high_pddl': [{'high_idx': 0, 'discrete_action': {'action': 'End', 'args': []},
                           'planner_action': {'action': 'End', 'value': 1}}],
            'low_actions': [],
            'images': [],
        }
        with open(os.path.join(task_dir, 'trajectory.json'), 'w') as f:
            json.dump(trajectory, f, indent=2, ensure_ascii=False)

    summary = {k: v for k, v in report.items() if k not in ('exec_log', 'sim_log')}
    summary['frames_dir'] = frames_dir
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'report -> {os.path.join(task_out, "report.json")}')
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Generate + verify a task via id-driven primitives')
    parser.add_argument('--task-dir', required=True, help='task directory (contains task.yaml)')
    parser.add_argument('--out-dir', default='running/composite_verify')
    parser.add_argument('--resolution', type=int, default=300)
    parser.add_argument('--x-display', default='1')
    parser.add_argument('--no-trajectory', action='store_true',
                        help='do not write trajectory.json even if feasible')
    parser.add_argument('--save-frames', action='store_true',
                        help='save per-step RGB frames under <out-dir>/<name>/frames/')
    args = parser.parse_args()
    run(args.task_dir, args.out_dir, args.resolution, args.x_display,
        write_trajectory=not args.no_trajectory, save_frames=args.save_frames)


if __name__ == '__main__':
    main()
