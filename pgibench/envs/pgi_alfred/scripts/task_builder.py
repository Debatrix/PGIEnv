#!/usr/bin/env python3
"""Two-step task builder for the compose-alfred-task skill.

Agent-driven flow (each stage is a separate CLI subcommand):

    # 1. preview: spawn instruction objects + distractors on AI2-THOR, snapshot
    #    the resulting world state (object_poses/toggles/init_action), and write
    #    a layout JSON for the agent to design subtasks + macro/micro instructions
    #    against. Does NOT write task.yaml.
    python -m pgibench.envs.pgi_alfred.scripts.task_builder preview \
        --intent intent.yaml --out running/preview_<name>.json

    # 2. write: assemble the agent's full cfg (preview JSON + designed subtasks +
    #    dual instructions) into the final task.yaml.
    python -m pgibench.envs.pgi_alfred.scripts.task_builder write \
        --cfg full.yaml --out-dir data/tasks/test/<name>

The legacy one-shot `build` subcommand has been removed: it wrote random_spawn
without object_poses/init_action, so the two-stage restore (SetObjectPoses +
TeleportAgent) never engaged for its tasks. preview+write always emit the full
restore block; trajectory generation + feasibility check is a separate step
(id_trajectory_generator).

Requires X server. Recommended (Xvfb):
    python -m pgibench.envs.pgi_alfred.scripts.start_xvfb 1
Alternative (Xorg):
    python -m pgibench.envs.pgi_alfred.scripts.startx 1
"""

import os
import json
import argparse

import yaml

from pgibench.envs.pgi_alfred.env.validate_composite import (
    validate_config, TASK_TYPE_PARAMS)
from pgibench.envs.pgi_alfred.env.scene_snapshot import (
    random_initialize_scene, sample_distractors)
from pgibench.envs.pgi_alfred.env.task_format import write_task


def _subtask_to_pp(st):
    """Map an intent subtask (object/parent/toggle/movable_recep/sliced) to pddl_params."""
    tt = st['task_type']
    pp = {
        'object_target': st.get('object', ''),
        'parent_target': st.get('parent', ''),
        'mrecep_target': st.get('movable_recep', ''),
        'toggle_target': st.get('toggle', ''),
        'object_sliced': bool(st.get('sliced', False)),
    }
    params = TASK_TYPE_PARAMS.get(tt, [])
    if 'movable_recep' not in params:
        pp['mrecep_target'] = ''
    if 'toggle' not in params:
        pp['toggle_target'] = ''
    return pp


def _counts_from_subtasks(subtasks):
    """Pickupable instruction objects to spawn. Also spawns pickupable parent
    receptacles (Plate/Bowl/Mug/... in MOVABLE_RECEPTACLES) so they exist; fixed
    furniture (CounterTop/Shelf/...) and facilities (microwave/fridge/lamp) are
    scene-fixed and not spawned.

    Subtasks with 'reuse: True' are skipped — they reference an object already
    spawned by an earlier subtask (search_use_later pattern: pick A from hiding,
    do other things, then move A again). Without this skip, counts[A]=2 spawns
    two instances and the second resolve finds the wrong one."""
    from pgibench.envs.pgi_alfred.env.validate_composite import MOVABLE_RECEPTACLES
    counts = {}
    for st in subtasks:
        if st.get('reuse'):
            continue
        tt = st['task_type']
        n = 2 if tt == 'pick_two_obj_and_place' else 1
        if st.get('object'):
            counts[st['object']] = counts.get(st['object'], 0) + n
        if st.get('movable_recep'):
            counts[st['movable_recep']] = counts.get(st['movable_recep'], 0) + 1
        parent = st.get('parent')
        if parent and parent in MOVABLE_RECEPTACLES:
            counts[parent] = counts.get(parent, 0) + 1
    return counts


def build_preview(intent_path, x_display='1', resolution=300):
    """Spawn instruction objects + distractors, validate, return a layout JSON
    for the agent to design final subtasks + instructions against. Does NOT
    write task.yaml. Raises if instruction objects not inherent / missing."""
    with open(intent_path) as f:
        intent = yaml.safe_load(f)

    cfg_check = {
        'name': intent['name'],
        'scene': {'room_type': intent['room_type'],
                  'scene_num': intent.get('scene_num')},
        'instruction': intent.get('macro', '') or intent.get('instruction', ''),
        'subtasks': intent['subtasks'],
        'completion': {'mode': 'all'},
    }
    ok, errors, _ = validate_config(cfg_check)
    if not ok:
        raise ValueError('intent failed validation: ' + '; '.join(errors))

    room = intent['room_type']
    scene_num = intent['scene_num']
    seed = int(intent.get('seed', 0))
    counts = _counts_from_subtasks(intent['subtasks'])
    distractors = sample_distractors(room, intent.get('n_distractors', 6),
                                     seed, exclude=set(counts))
    # Reserve free slots on every put-target receptacle type so InitialRandomSpawn
    # doesn't clutter the surface with distractors — without this, put fails with
    # "No valid positions to place object found" on small surfaces (Dresser /
    # SideTable / CounterTop). One slot per type (not cumulative): AI2-THOR's
    # minFreePerReceptacleType is a per-type floor, and asking for N slots on a
    # small surface (SideTable typically holds 1-2) makes spawn silently ignore
    # the constraint or fail. The instantiate layer handles diversity by
    # preferring unused receptacles (used_receps in choose_recep). SinkBasin
    # gets its own slot for clean subtasks.
    min_free = {}
    subtask_types = {st['task_type'] for st in intent['subtasks']}
    if 'pick_clean_then_place_in_recep' in subtask_types:
        min_free['SinkBasin'] = 1
    for st in intent['subtasks']:
        parent = st.get('parent')
        # Reserve ONE free slot per put occurrence (cumulative), not per type, so
        # a receptacle type targeted by several subtasks keeps enough room for
        # every landing object (otherwise later puts fail with
        # "No valid Receptacle found").
        if parent:
            min_free[parent] = min_free.get(parent, 0) + 1
    min_free = min_free or None

    from pgibench.envs.pgi_alfred.thor_connector import ThorConnector
    from pgibench.envs.pgi_alfred.env.scene_snapshot import (
        random_initialize_scene, inherent_pickupable_types, layout_summary)
    env = ThorConnector(x_display=x_display,
                        player_screen_height=resolution,
                        player_screen_width=resolution)
    try:
        env.reset(f'FloorPlan{scene_num}')
        inherent = inherent_pickupable_types(env)
        non_inherent = [ot for ot in counts if ot not in inherent]
        if non_inherent:
            raise RuntimeError(
                f"objects not inherent to FloorPlan{scene_num}: {non_inherent}. "
                f"Pickupable types available: {sorted(inherent)}")
        scene_block = random_initialize_scene(env, scene_num, seed, counts, distractors,
                                                   min_free=min_free)
        # Apply container constraints: force designated instruction objects into
        # closed containers (Drawer/Cabinet/Safe/Box) so the agent must OPEN the
        # receptacle to find them. Powers search_and_stow / search_use_later.
        container_constraints = intent.get('container_constraints')
        if container_constraints:
            from pgibench.envs.pgi_alfred.env.scene_snapshot import (
                spawn_into_container, snapshot_scene)
            applied = []
            for cc in container_constraints:
                result = spawn_into_container(env, cc['object'], cc['container'])
                if result:
                    applied.append({'object': cc['object'],
                                    'container': cc['container'], **result})
            if applied:
                # poses changed (object moved into container) — re-snapshot
                scene_block = snapshot_scene(env)
        # Apply scene_positions scatter: re-pose designated objects to verified
        # curated poses (incl. floor-level y<0.3 for "pick up from floor"
        # semantics). Objects must already exist (spawned/inherent); non-existent
        # types are skipped. Powers the ground-scatter archetypes.
        scatter_specs = intent.get('scene_positions_scatter')
        scatter_result = None
        if scatter_specs:
            from pgibench.envs.pgi_alfred.env.scene_snapshot import (
                scatter_from_positions, snapshot_scene)
            placed, skipped = scatter_from_positions(env, scatter_specs, seed)
            if placed:
                scene_block = snapshot_scene(env)
            scatter_result = {'placed': placed, 'skipped': skipped}
        present = {o['objectType'] for o in env.last_event.metadata['objects']}
        summary = layout_summary(env)
    finally:
        try: env.stop()
        except Exception: pass

    missing = [ot for ot in counts if ot not in present]
    if missing:
        raise RuntimeError(
            f"instruction objects missing after spawn: {missing}. "
            f"Try another seed/scene or different object types.")

    random_spawn = {'seed': seed, 'object_counts': counts,
                    'distractors': distractors, 'min_free': min_free}
    if scatter_specs:
        random_spawn['scene_positions_scatter'] = scatter_specs
    return {
        'name': intent['name'],
        'scene': {'floor_plan': f'FloorPlan{scene_num}',
                  'scene_num': scene_num, 'room_type': room},
        'random_spawn': random_spawn,
        'object_poses': scene_block['object_poses'],
        'object_toggles': scene_block['object_toggles'],
        'init_action': scene_block['init_action'],
        'layout_summary': summary,
        'scatter_result': scatter_result,
    }


def write_from_cfg(cfg_path, out_dir):
    """Write task.yaml from a complete cfg yaml assembled by the agent
    (preview JSON + agent-designed subtasks + dual instructions)."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    write_task(out_dir, cfg)
    print(json.dumps({'task_id': cfg.get('task_id'), 'out_dir': out_dir},
                     indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description='Build/preview a compose-alfred-task task')
    sub = p.add_subparsers(dest='cmd', required=True)

    pre = sub.add_parser('preview')
    pre.add_argument('--intent', required=True)
    pre.add_argument('--out', default=None,
                     help='output JSON path (default running/preview_<name>.json)')
    pre.add_argument('--x-display', default='1')
    pre.add_argument('--resolution', type=int, default=300)

    wr = sub.add_parser('write')
    wr.add_argument('--cfg', required=True)
    wr.add_argument('--out-dir', required=True)

    args = p.parse_args()
    if args.cmd == 'preview':
        # Write JSON to a file via json.dump so the AI2-THOR Unity startup banner
        # (also on stdout) cannot corrupt it — shell `>` redirect would mix the
        # banner into the file and break json.load. The output path is printed last;
        # callers read it as the final stdout line, never the whole stdout.
        data = build_preview(args.intent, args.x_display, args.resolution)
        out = args.out or f"running/preview_{data['name']}.json"
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(out)
    elif args.cmd == 'write':
        write_from_cfg(args.cfg, args.out_dir)


if __name__ == '__main__':
    main()
