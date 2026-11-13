#!/usr/bin/env python3
"""One-time floorplan probe for compose-alfred-task batch planning.

Probes each AI2-THOR FloorPlan once and caches the facts the batch planner needs
to pick realizable objects/receptacles/facilities WITHOUT spawning:

    FloorPlan<N> -> {
        "scene_num": N,
        "room": "Kitchen" | "LivingRoom" | "Bedroom" | "Bathroom",
        "pickupable": [<inherent pickupable objectTypes>, sorted],
        "receptacles": {<recep_type>: <lowest instance Y>},
        "facilities": {Microwave/Fridge/SinkBasin/Faucet/DeskLamp/FloorLamp: bool}
    }

`InitialRandomSpawn` can only add instances of object types INHERENT to a
FloorPlan, so the planner reads `pickupable` to pick instruction objects that
will actually spawn. `receptacles` (Y) filters put-targets by reachability
(agent ~0.9 tall; surfaces far above are unreliable). `facilities` gates
heat/cool/clean/look_at/toggle subtasks.

Re-runnable: reconciles with an existing cache (keeps existing entries, updates
the probed ones). Needs a running X server (Xvfb on :1 is the usual setup):

    python -m pgibench.envs.pgi_alfred.scripts.start_xvfb 1
    python <skill_dir>/scripts/probe_floorplans.py

Output: outputs/temp/floorplan_cache.json (~25 min for all 120 FloorPlans).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Auto-detect repo root by walking up to the .git dir. The skill ships under
# .claude/skills/compose-alfred-task/, so the repo root is a few levels up.
_ROOT = Path(__file__).resolve()
while _ROOT.name != '' and not (_ROOT / '.git').is_dir():
    if _ROOT.parent == _ROOT:
        break
    _ROOT = _ROOT.parent
ROOT = _ROOT
if not (ROOT / 'pgibench').is_dir():
    print(f'ERROR: could not locate repo root from {__file__}', file=sys.stderr)
    sys.exit(2)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from pgibench.envs.pgi_alfred.env.validate_composite import ROOM_SCENES
from pgibench.envs.pgi_alfred.env import scene_snapshot as _ss
from pgibench.envs.pgi_alfred.thor_connector import ThorConnector

# Reverse map scene_num -> room_type, built once from ROOM_SCENES.
SCENE_TO_ROOM = {}
for _room, _scenes in ROOM_SCENES.items():
    for _sn in _scenes:
        SCENE_TO_ROOM[_sn] = _room
ALL_SCENES = sorted(SCENE_TO_ROOM.keys())


def probe_one(scene_num, x_display, resolution):
    """Probe a single FloorPlan: reset a fresh controller, snapshot layout, stop.

    Returns the cache entry dict. Raises on controller error (no masking)."""
    env = ThorConnector(x_display=x_display,
                        player_screen_height=resolution,
                        player_screen_width=resolution)
    try:
        env.reset(f'FloorPlan{scene_num}')
        inherent = sorted(_ss.inherent_pickupable_types(env))
        summary = _ss.layout_summary(env)
    finally:
        try:
            env.stop()
        except Exception:
            pass
    return {
        'scene_num': scene_num,
        'room': SCENE_TO_ROOM[scene_num],
        'pickupable': inherent,
        'receptacles': summary.get('receptacles', {}),
        'facilities': summary.get('facilities', {}),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='outputs/temp/floorplan_cache.json',
                    help='Output cache JSON (default: outputs/temp/floorplan_cache.json).')
    ap.add_argument('--x-display', default='1', help='X display (default: 1).')
    ap.add_argument('--resolution', type=int, default=300,
                    help='THOR render resolution (default: 300).')
    ap.add_argument('--scenes', default=None,
                    help='JSON array of scene numbers to probe (default: all 120).')
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reconcile with an existing cache so re-runs only update/add probed scenes.
    cache = {}
    if out_path.is_file():
        cache = json.loads(out_path.read_text())
        print(f'loaded existing cache: {len(cache)} entries from {out_path}',
              flush=True)

    if args.scenes:
        targets = json.loads(args.scenes)
    else:
        targets = ALL_SCENES

    t0 = time.time()
    for i, sn in enumerate(targets):
        room = SCENE_TO_ROOM.get(sn)
        if room is None:
            print(f'[{i + 1}/{len(targets)}] scene {sn} not in any room; skipped',
                  flush=True)
            continue
        fp_key = f'FloorPlan{sn}'
        print(f'[{i + 1}/{len(targets)}] probing {fp_key} ({room}) ...',
              flush=True)
        t1 = time.time()
        entry = probe_one(sn, args.x_display, args.resolution)
        entry['elapsed_sec'] = round(time.time() - t1, 1)
        cache[fp_key] = entry
        n_pick = len(entry['pickupable'])
        n_rec = len(entry['receptacles'])
        fac_on = [k for k, v in entry['facilities'].items() if v]
        print(f'    pickupable={n_pick} receptacles={n_rec} '
              f'facilities={fac_on} ({entry["elapsed_sec"]}s)', flush=True)
        # Stream-write after each scene so a crash mid-run keeps progress.
        out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

    print(f'\n=== DONE ===\n{len(cache)} entries cached -> {out_path} '
          f'(elapsed {round(time.time() - t0, 1)}s)', flush=True)


if __name__ == '__main__':
    main()
