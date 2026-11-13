#!/usr/bin/env python3
"""Floorplan-aware batch generator for compose-alfred-task.

Ships inside the skill at .claude/skills/compose-alfred-task/scripts/. Two modes:

  --plan-only (recommended for batch generation): sample N diverse intent specs
    (subtask skeletons + object choices) from the floorplan cache and write one
    intent YAML per task to --intent-dir. No simulator is launched and NO
    instructions are authored — the intent carries only a goal_idea for the
    authoring subagent. The orchestrator then dispatches one subagent per intent
    (see references/batch_tools.md); each subagent runs preview, reads the layout,
    authors the real macro+micro, and writes the final task.yaml.

  default (legacy verify mode): for each (room, scene, archetype), run preview +
    verify feasibility against a TEMPORARY task.yaml that carries stub
    instructions. The id-driven checker ignores instructions (it is driven by
    pddl_params.subtasks), so those stubs never reach a delivered task. On
    feasibility, dump the full preview context to outputs/temp/phase_preview/<task_id>.json
    and discard the temp task; on infeasibility the attempt is discarded entirely.
    No task.yaml is written to the output dir by this script.

Instructions are authored ONLY by an authoring agent (you or a subagent) from
the real preview layout — never stamped as a placeholder, never a generic
template. This script only plans intents and (in verify mode) screens them for
feasibility.

Generates tasks per room (Kitchen / Bathroom / LivingRoom / Bedroom) on scenes
[Kitchen 1-30, Bathroom 401-430, LivingRoom 201-230, Bedroom 301-330].
Each task has 4-6 subtasks; distractors = random in [3x, 5x] instruction objects
(capped at 45).

Outputs (plan-only): <intent-dir>/<task_id>.yaml (one intent spec per planned
task), outputs/temp/plan_only_report.json (summary).
Outputs (verify mode): outputs/temp/phase_preview/<task_id>.json (one per feasible
task — agent input), <out-dir>/<report-name> (summary; success = feasible count).

All temp files this skill produces live under outputs/temp/ (per AGENTS.md
convention). Repo-root is auto-detected by walking up to .git, so this script is
runnable from anywhere (including the skill's own scripts/ dir).
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import yaml

# Auto-detect repo root by walking up to the .git dir. The skill ships under
# .claude/skills/compose-alfred-task/scripts/, so the repo root is a few levels
# above this file. Using .git (not pgibench/) because outputs/pgibench/ also
# exists as a legacy copy and would fool a shallower check.
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

from pgibench.envs.pgi_alfred.scripts.task_builder import (
    build_preview, write_from_cfg, _counts_from_subtasks, _subtask_to_pp)
from pgibench.envs.pgi_alfred.scripts.id_trajectory_generator import run as verify_run
from pgibench.envs.pgi_alfred.env.scene_snapshot import sample_distractors
from pgibench.envs.pgi_alfred.env.validate_composite import (
    ACTION_OBJECT_SETS, TOGGLEABLE, MOVABLE_RECEPTACLES, ROOM_SCENES)
from pgibench.envs.pgi_alfred.env import scene_snapshot as _ss

# Default output task dir + eval_set; main() overrides via --out-task-dir /
# --eval-set. Kept as module-global so attempt() / write_full_cfg reach them
# without threading a new param through every call.
COMPOSE_DIR = ROOT / 'pgibench/envs/pgi_alfred/data/tasks/compose'
EVAL_SET = 'compose'
# All skill-produced temp files live under outputs/temp/ (AGENTS.md convention).
TEMP_DIR = ROOT / 'outputs/temp'
# Default location for task_verify/ and report; main() overrides via --out-dir.
OUT_DIR = ROOT / 'outputs/temp'
INTENT_TMP = TEMP_DIR / '_v2_intent_tmp.yaml'
CFG_TMP = TEMP_DIR / '_v2_cfg_tmp.yaml'

# Receptacle blacklist as put-target (soft/unstable/closed/otherwise fail-prone).
# Closed openable receptacles (Cabinet/Drawer/Fridge/Microwave/Safe/Box) are
# excluded because put-into requires opening first; we use open flat surfaces.
RECEP_BLACKLIST = {
    'GarbageCan', 'Safe', 'Cart', 'LaundryHamper', 'Trashcan',
    'ToiletPaperHanger', 'HandTowelHolder', 'TowelHolder',
    'Bathtub', 'BathtubBasin', 'Sink', 'SinkBasin', 'Toilet',
    'Sofa', 'ArmChair', 'Bed', 'Ottoman', 'Footstool', 'Window',
    'StoveBurner', 'Toaster', 'CoffeeMachine', 'Microwave', 'Fridge',
    'Cabinet', 'Drawer', 'Box',
    'TVStand', 'Desk', 'Shelf',  # put fails on Shelf/Desk in many scenes
                                  # (cluttered or unreachable surface)
}

# Whitelisted put-targets per room — large flat surfaces, put-robust.
RECEP_PREFERRED = {
    'Kitchen': ['CounterTop', 'DiningTable'],
    'Bathroom': ['SideTable', 'CounterTop'],
    'LivingRoom': ['CoffeeTable', 'SideTable', 'Dresser', 'DiningTable'],
    'Bedroom': ['SideTable', 'Dresser', 'DiningTable'],
}

# Rooms where toggle_object_on/off and look_at_obj_in_light tend to fail in
# id_trajectory_generator (lamp exists but Turn on action fails). Drop those
# archetypes here — only pick_and_place family + slice will be tried.
NO_TOGGLE_ROOMS = {'Bedroom'}

# Subtask archetypes per room. Each lists subtask slots; the generator picks
# concrete object/parent per slot from the floorplan cache. 'n_range' specifies
# the (min, max) for repeatable slots so total subtasks land in 4-6.
# The 'macro' here is only the goal_idea the authoring subagent realizes — the
# subagent authors the real macro+micro from the preview layout. It is never a
# delivered instruction.
ARCHETYPES = {
    'Kitchen': [
        # Priority-A-led: state-change composites (heat/cool/clean/slice_place)
        # lead; simple 'place' only completes a coherent goal (plate + utensils).
        {'id': 'hot_meal', 'macro': "I'd love a hot meal — something sliced, heated, and plated.",
         'slots': [('heat', 1), ('slice_place', 1), ('place', 1, 2)]},
        {'id': 'cold_snack', 'macro': "Put together a cold snack — slice and chill it.",
         'slots': [('cool', 1), ('slice_place', 1), ('place', 1, 2)]},
        {'id': 'wash_dishes', 'macro': "Get things washed, sliced, and put away in the kitchen.",
         'slots': [('clean', 1), ('slice_place', 1), ('clean', 1)]},
        {'id': 'cook_combo', 'macro': "I want a full meal — a hot main, cold sides, with some slicing.",
         'slots': [('heat', 1), ('cool', 1), ('slice_place', 1)]},
        {'id': 'clean_tidy', 'macro': "Clean up the kitchen — wash, slice, and put things away.",
         'slots': [('clean', 1), ('slice_place', 1), ('clean', 1)]},
        {'id': 'breakfast', 'macro': "Cook breakfast — heat, chill, and slice; make it quick.",
         'slots': [('heat', 1), ('slice_place', 1), ('cool', 1)]},
        {'id': 'all_four', 'macro': "Cook me something with everything — hot, cold, washed, and cut.",
         'slots': [('heat', 1), ('cool', 1), ('clean', 1), ('slice_place', 1)]},
        {'id': 'cook_meal_movable', 'macro': "Slice the ingredients, heat one, prep another in a pan, and plate it all.",
         'slots': [('slice_place', 2), ('heat', 1), ('movable_recep', 1), ('place', 1)]},
        # Search archetypes: hidden-object exploration + memory.
        {'id': 'search_and_stow',
         'macro': "Some stuff got hidden away — find it and put it on the counter.",
         'slots': [('place', 3)], 'search': True, 'search_count': 2},
        # Floor-scatter: instruction objects dropped on the floor via
        # scene_positions_scatter; agent must pick them up from the floor.
        {'id': 'floor_tidy', 'macro': "A few things fell on the kitchen floor — pick them up and put them away.",
         'slots': [('place', 3, 4)], 'scatter': True, 'scatter_floor_only': True},
    ],
    'Bathroom': [
        # clean is the only Priority-A composite available in Bathroom.
        {'id': 'wash_stow', 'macro': "Rinse the bathroom items and stow them on the counter.",
         'slots': [('clean', 2), ('place', 1, 2)]},
        {'id': 'clean_organize', 'macro': "Straighten up the bathroom items and stow them.",
         'slots': [('clean', 1), ('place', 2, 3)]},
        {'id': 'tidy', 'macro': "Tidy up the bathroom — put things on the counter.",
         'slots': [('place', 4, 5)]},
        {'id': 'floor_tidy', 'macro': "Pick up the things on the bathroom floor and put them on the counter.",
         'slots': [('place', 3, 4)], 'scatter': True, 'scatter_floor_only': True},
    ],
    'LivingRoom': [
        # look_at is the Priority-A composite here; toggle/look_at chain first.
        {'id': 'reading_setup', 'macro': "Get the room ready so I can read tonight.",
         'slots': [('look_at', 1), ('toggle_on', 1), ('place', 1, 2)]},
        {'id': 'lights_off_tidy', 'macro': "Turn off the floor lamp and tidy up before bed.",
         'slots': [('toggle_off', 1), ('place', 2, 3)]},
        {'id': 'tidy', 'macro': "Straighten up the living room.",
         'slots': [('place', 4, 5)]},
        {'id': 'search_and_stow',
         'macro': "Some stuff got hidden away — find it and put it on the coffee table.",
         'slots': [('place', 3)], 'search': True, 'search_count': 2},
        {'id': 'floor_tidy', 'macro': "Pick up the stuff on the living room floor and put it away.",
         'slots': [('place', 3, 4)], 'scatter': True, 'scatter_floor_only': True},
    ],
    'Bedroom': [
        # look_at is the Priority-A composite here.
        {'id': 'night_reading', 'macro': "Set the bedroom up for a night of reading.",
         'slots': [('toggle_on', 1), ('look_at', 1), ('place', 1, 2)]},
        {'id': 'lights_tidy', 'macro': "Get the room sorted before bed.",
         'slots': [('look_at', 1), ('place', 2, 3)]},
        {'id': 'tidy', 'macro': "Tidy up the bedroom.",
         'slots': [('place', 4, 5)]},
        # search_and_stow: pure-tidy Bedroom is saturated; search adds exploration.
        {'id': 'search_and_stow',
         'macro': "Some stuff got hidden away — find it and put it on the side table.",
         'slots': [('place', 3)], 'search': True, 'search_count': 2},
        # search_use_later: tests long-horizon memory (B-type).
        {'id': 'search_use_later',
         'macro': "Find the hidden item, do a few things, then put it on the side table.",
         'slots': [('place', 1), ('place', 2, 3), ('place', 1)],
         'search': True, 'search_reuse': True},
        {'id': 'floor_tidy', 'macro': "Pick up the stuff on the bedroom floor and put it away.",
         'slots': [('place', 3, 4)], 'scatter': True, 'scatter_floor_only': True},
    ],
}

# Per-action object candidates (must be in floorplan's inherent set to be used).
CLEANABLE_BATH = ['Cloth', 'DishSponge', 'SoapBar']
LOOK_AT_OBJS = ['Book', 'Newspaper']

# Toggle targets (must be present in facilities).
TOGGLE_FLOOR_LAMP = 'FloorLamp'
TOGGLE_DESK_LAMP = 'DeskLamp'

# Closed-container types where hiding an object creates a search subtask.
# Mirrors scene_snapshot.HIDDEN_CONTAINER_TYPES. Cabinet/Drawer are room
# furniture present in most FloorPlans; Safe/Box are rarer.
HIDDEN_CONTAINER_TYPES = ('Cabinet', 'Drawer', 'Safe', 'Box')


def _available_hidden_containers(cache_entry):
    """Hidden container types present in this floorplan (from probe cache)."""
    return [r for r in cache_entry.get('receptacles', {})
            if r in HIDDEN_CONTAINER_TYPES]


def reachable_receps(cache_entry, room, y_max=1.05):
    """Return list of (recep_type, y) reachable in this floorplan, filtered by
    whitelist preference + blacklist + Y reachability. Sorted by Y ascending
    (lower = more reachable)."""
    receps = cache_entry.get('receptacles', {})
    preferred = RECEP_PREFERRED.get(room, [])
    out = []
    for rtype, y in receps.items():
        if rtype in RECEP_BLACKLIST:
            continue
        if y > y_max:
            continue
        out.append((rtype, y))
    # Preferred first, then by Y ascending.
    out.sort(key=lambda ry: (0 if ry[0] in preferred else 1, ry[1]))
    return out


def choose_recep(receps_list, rng, used_receps=None):
    """Pick a receptacle preferring diversity."""
    if not receps_list:
        return None
    unused = [r for r in receps_list if r[0] not in (used_receps or set())]
    pool = unused if unused else receps_list
    return rng.choice(pool)[0]


def instantiate_search_subtasks(archetype, cache_entry, room, seed):
    """Materialize search-archetype slots into subtasks + container_constraints.

    Two flavors:
      - search_and_stow: N objects hidden in closed containers, all placed on
        one open receptacle. Tests exploration + open-container coordination.
      - search_use_later: object A is hidden, must be brought to a staging
        spot first, then 2 unrelated subtasks intervene, then A is moved to
        its final spot. Tests long-horizon memory of where A was staged.

    Returns (subtasks, chosen_objects, container_constraints)."""
    rng = random.Random(seed * 1000 + cache_entry['scene_num'] + hash(archetype['id']) % 1000)
    inherent = set(cache_entry['pickupable'])
    receps = reachable_receps(cache_entry, room)
    containers = _available_hidden_containers(cache_entry)
    if not containers:
        raise ValueError('no hidden container in floorplan')

    subtasks = []
    chosen_objects = []
    container_constraints = []

    if archetype['id'] == 'search_and_stow':
        n_hide = archetype.get('search_count', 2)
        total = 4
        for i in range(total):
            pool = [o for o in inherent if o not in chosen_objects]
            if not pool:
                raise ValueError('not enough inherent objects')
            obj = rng.choice(pool)
            parent = choose_recep(receps, rng)
            if not parent:
                raise ValueError('no reachable recep')
            subtasks.append({'task_type': 'pick_and_place_simple',
                             'object': obj, 'parent': parent})
            chosen_objects.append(obj)
            if i < n_hide:
                # Prefer Cabinet (spawn_into_container reliable); Drawer open
                # fails in build_preview (agent height/orientation mismatch).
                cpool = [c for c in containers if c == 'Cabinet'] or containers
                cont = rng.choice(cpool)
                container_constraints.append({'object': obj, 'container': cont})

    elif archetype['id'] == 'search_use_later':
        # subtask 0: searched obj (hidden) -> staging recep
        # subtasks 1-2: unrelated interval objs -> other receps
        # subtask 3: reuse searched obj -> final recep
        pool = [o for o in inherent]
        if not pool:
            raise ValueError('no inherent object')
        searched_obj = rng.choice(pool)
        chosen_objects.append(searched_obj)
        staging = choose_recep(receps, rng)
        if not staging:
            raise ValueError('no staging recep')
        subtasks.append({'task_type': 'pick_and_place_simple',
                         'object': searched_obj, 'parent': staging})
        cpool = [c for c in containers if c == 'Cabinet'] or containers
        cont = rng.choice(cpool)
        container_constraints.append({'object': searched_obj, 'container': cont})
        # interval subtasks (2)
        for _ in range(2):
            pool = [o for o in inherent if o not in chosen_objects]
            if not pool:
                break
            obj = rng.choice(pool)
            parent = choose_recep(receps, rng)
            if not parent:
                break
            subtasks.append({'task_type': 'pick_and_place_simple',
                             'object': obj, 'parent': parent})
            chosen_objects.append(obj)
        # final reuse
        final_parent = choose_recep(receps, rng)
        if final_parent:
            subtasks.append({'task_type': 'pick_and_place_simple',
                             'object': searched_obj, 'parent': final_parent,
                             'reuse': True})

    else:
        raise ValueError(f"unknown search archetype: {archetype['id']}")

    if len(subtasks) < 4:
        raise ValueError(f'only {len(subtasks)} subtasks instantiated')
    return subtasks, chosen_objects, container_constraints


def instantiate_subtasks(archetype, cache_entry, room, seed):
    """Materialize archetype slots into concrete subtask dicts using floorplan
    cache. Returns (subtasks, chosen_objects) or raises if scene can't support.
    """
    rng = random.Random(seed * 1000 + cache_entry['scene_num'] + hash(archetype['id']) % 1000)
    inherent = set(cache_entry['pickupable'])
    facilities = cache_entry['facilities']
    receps = reachable_receps(cache_entry, room)

    subtasks = []
    chosen_objects = []
    used_receps = set()

    for slot in archetype['slots']:
        action = slot[0]
        n_min = slot[1]
        n_max = slot[2] if len(slot) > 2 else n_min
        n = rng.randint(n_min, n_max)

        for _ in range(n):
            if action == 'heat':
                pool = [o for o in ACTION_OBJECT_SETS['heat'] if o in inherent
                        and o not in chosen_objects]
                if not pool:
                    raise ValueError('no heatable object inherent')
                obj = rng.choice(pool)
                parent = choose_recep(receps, rng, used_receps)
                if not parent:
                    raise ValueError('no reachable recep')
                subtasks.append({'task_type': 'pick_heat_then_place_in_recep',
                                 'object': obj, 'parent': parent})
                chosen_objects.append(obj)
                used_receps.add(parent)

            elif action == 'cool':
                pool = [o for o in ACTION_OBJECT_SETS['cool'] if o in inherent
                        and o not in chosen_objects]
                if not pool:
                    raise ValueError('no coolable object inherent')
                obj = rng.choice(pool)
                parent = choose_recep(receps, rng, used_receps)
                if not parent:
                    raise ValueError('no reachable recep')
                subtasks.append({'task_type': 'pick_cool_then_place_in_recep',
                                 'object': obj, 'parent': parent})
                chosen_objects.append(obj)
                used_receps.add(parent)

            elif action == 'clean':
                if room == 'Kitchen':
                    pool = [o for o in ACTION_OBJECT_SETS['clean'] if o in inherent
                            and o not in chosen_objects]
                else:  # Bathroom
                    pool = [o for o in CLEANABLE_BATH if o in inherent
                            and o not in chosen_objects]
                if not pool:
                    raise ValueError('no cleanable object inherent')
                obj = rng.choice(pool)
                parent = choose_recep(receps, rng, used_receps)
                if not parent:
                    raise ValueError('no reachable recep')
                subtasks.append({'task_type': 'pick_clean_then_place_in_recep',
                                 'object': obj, 'parent': parent})
                chosen_objects.append(obj)
                used_receps.add(parent)

            elif action == 'slice':
                pool = [o for o in ACTION_OBJECT_SETS['slice'] if o in inherent
                        and o not in chosen_objects]
                if not pool:
                    raise ValueError('no sliceable object inherent')
                obj = rng.choice(pool)
                subtasks.append({'task_type': 'slice_in_place', 'object': obj})
                chosen_objects.append(obj)

            elif action == 'slice_place':
                # Priority-A composite: slice the object then place it in a recep.
                # sliced=True is propagated to pddl_params.object_sliced by
                # _subtask_to_pp, so BaseTask.get_targets() returns the Sliced-
                # tagged object the goal check relies on.
                pool = [o for o in ACTION_OBJECT_SETS['slice'] if o in inherent
                        and o not in chosen_objects]
                if not pool:
                    raise ValueError('no sliceable object inherent')
                obj = rng.choice(pool)
                parent = choose_recep(receps, rng, used_receps)
                if not parent:
                    raise ValueError('no reachable recep')
                subtasks.append({'task_type': 'pick_slice_then_place_in_recep',
                                 'object': obj, 'parent': parent, 'sliced': True})
                chosen_objects.append(obj)
                used_receps.add(parent)

            elif action == 'place':
                # any pickupable that's not already used; avoid heatables
                # already targeted by other actions (they'd conflict in goal state).
                pool = [o for o in inherent if o not in chosen_objects]
                if not pool:
                    raise ValueError('no placeable object inherent')
                obj = rng.choice(pool)
                parent = choose_recep(receps, rng, used_receps)
                if not parent:
                    raise ValueError('no reachable recep')
                subtasks.append({'task_type': 'pick_and_place_simple',
                                 'object': obj, 'parent': parent})
                chosen_objects.append(obj)
                used_receps.add(parent)

            elif action == 'movable_recep':
                # pick_and_place_with_movable_recep: object into a Pan/Pot/Bowl
                # (movable recep), then the recep onto a CounterTop/DiningTable.
                mov_pool = [o for o in MOVABLE_RECEPTACLES if o in inherent
                            and o not in chosen_objects]
                # prefer Pan/Pot (cookware) over Plate/Bowl/Mug for a cook task
                cookware = [o for o in mov_pool if o in ('Pan', 'Pot')]
                pool = cookware or mov_pool
                if not pool:
                    raise ValueError('no movable recep object inherent')
                obj_pool = [o for o in inherent if o not in chosen_objects
                            and o not in pool]
                if not obj_pool:
                    raise ValueError('no object for movable recep')
                obj = rng.choice(obj_pool)
                mrecep = rng.choice(pool)
                parent = choose_recep(receps, rng, used_receps)
                if not parent:
                    raise ValueError('no reachable recep')
                subtasks.append({'task_type': 'pick_and_place_with_movable_recep',
                                 'object': obj, 'parent': parent, 'movable_recep': mrecep})
                chosen_objects.append(obj)
                chosen_objects.append(mrecep)
                used_receps.add(parent)

            elif action == 'look_at':
                pool = [o for o in LOOK_AT_OBJS if o in inherent
                        and o not in chosen_objects]
                if not pool:
                    raise ValueError('no look_at object inherent')
                obj = rng.choice(pool)
                # look_at uses DeskLamp as toggle
                if not facilities.get('DeskLamp'):
                    raise ValueError('no DeskLamp')
                subtasks.append({'task_type': 'look_at_obj_in_light',
                                 'object': obj, 'toggle': TOGGLE_DESK_LAMP})
                chosen_objects.append(obj)

            elif action == 'toggle_on':
                # prefer FloorLamp (more common), fall back to DeskLamp
                if facilities.get('FloorLamp'):
                    subtasks.append({'task_type': 'toggle_object_on',
                                     'toggle': TOGGLE_FLOOR_LAMP})
                elif facilities.get('DeskLamp'):
                    subtasks.append({'task_type': 'toggle_object_on',
                                     'toggle': TOGGLE_DESK_LAMP})
                else:
                    raise ValueError('no lamp')

            elif action == 'toggle_off':
                if facilities.get('FloorLamp'):
                    subtasks.append({'task_type': 'toggle_object_off',
                                     'toggle': TOGGLE_FLOOR_LAMP})
                elif facilities.get('DeskLamp'):
                    subtasks.append({'task_type': 'toggle_object_off',
                                     'toggle': TOGGLE_DESK_LAMP})
                else:
                    raise ValueError('no lamp')

    if not (4 <= len(subtasks) <= 6):
        # Trim or accept depending on count. If >6, drop trailing place slots.
        if len(subtasks) > 6:
            subtasks = subtasks[:6]
        elif len(subtasks) < 4:
            raise ValueError(f'only {len(subtasks)} subtasks instantiated')

    return subtasks, chosen_objects


def write_intent_yaml(path, name, room, scene_num, seed, subtasks, n_distractors,
                      macro, container_constraints=None, scene_positions_scatter=None):
    intent = {
        'name': name,
        'room_type': room,
        'scene_num': scene_num,
        'seed': seed,
        'macro': macro,
        'n_distractors': n_distractors,
        'subtasks': subtasks,
    }
    if container_constraints:
        intent['container_constraints'] = container_constraints
    if scene_positions_scatter:
        intent['scene_positions_scatter'] = scene_positions_scatter
    with open(path, 'w') as f:
        yaml.safe_dump(intent, f, sort_keys=False, allow_unicode=True)


def write_full_cfg(path, preview_data, subtasks, macro, micro, task_id, room,
                   scene_num, n_distractors_planned, n_distractors_actual,
                   template_id):
    scene_block = {
        'floor_plan': preview_data['scene']['floor_plan'],
        'scene_num': scene_num,
        'room_type': room,
        'random_spawn': {
            'seed': preview_data['random_spawn']['seed'],
            'object_counts': preview_data['random_spawn']['object_counts'],
            'distractors': preview_data['random_spawn']['distractors'],
            'min_free': preview_data['random_spawn'].get('min_free'),
            # carry scene_positions_scatter through so restore reproduces it
            'scene_positions_scatter': preview_data['random_spawn'].get('scene_positions_scatter'),
        },
        'object_poses': preview_data['object_poses'],
        'object_toggles': preview_data['object_toggles'],
        'init_action': preview_data['init_action'],
        'dirty_and_empty': False,
    }
    pp_subtasks = [{'task_type': st['task_type'],
                    'pddl_params': _subtask_to_pp(st)} for st in subtasks]
    cfg = {
        'schema_version': 2,
        'task_id': task_id,
        'task_type': 'composite',
        'eval_set': EVAL_SET,
        'instructions': {'macro': macro, 'micro': micro},
        'scene': scene_block,
        'pddl_params': {
            'object_target': '', 'parent_target': '', 'mrecep_target': '',
            'toggle_target': '', 'object_sliced': False,
            'subtasks': pp_subtasks,
        },
        'subtasks': pp_subtasks,
        'completion': {'mode': 'all'},
        '_meta': {
            'template_id': template_id,
            'n_distractors_planned': n_distractors_planned,
            'n_distractors_actual': n_distractors_actual,
        },
    }
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def n_distractors_for(n_instruction_objects, room, seed):
    """Random in [3x, 5x], capped at 45. Cap is no longer bound to the
    distractor-pool size because sample_distractors now samples WITH replacement
    (allow_repeat=True), so repeat types fill the count beyond unique pool size."""
    cap = 45
    lo = max(3 * n_instruction_objects, 1)
    hi = max(5 * n_instruction_objects, lo)
    rng = random.Random(seed + 99999)
    n_target = rng.randint(lo, hi)
    return min(n_target, cap)


def attempt(room, scene_num, archetype, seed, x_display, resolution):
    """Single preview -> verify -> context-dump attempt (verify mode).

    Writes a TEMPORARY task.yaml (with stub instructions) ONLY to run the
    id-driven feasibility check, which ignores instructions entirely (driven
    by pddl_params.subtasks). If feasible, dumps the full preview context to
    outputs/temp/phase_preview/<task_id>.json and removes the temp task dir — the
    agent reads that context, authors the real macro+micro, and writes the
    FINAL task.yaml in a separate step. No [PLACEHOLDER] is ever left in any
    delivered task.
    Returns (status, result, ctx_path_or_None)."""
    cache_path = TEMP_DIR / 'floorplan_cache.json'
    cache = json.loads(cache_path.read_text())
    fp_key = f'FloorPlan{scene_num}'
    cache_entry = cache.get(fp_key)
    if not cache_entry:
        return 'no_cache', {'status': 'no_cache', 'reason': f'{fp_key} not in cache',
                            'room': room, 'scene_num': scene_num}, None

    try:
        if archetype.get('search'):
            subtasks, chosen_objects, container_constraints = \
                instantiate_search_subtasks(archetype, cache_entry, room, seed)
        else:
            subtasks, chosen_objects = instantiate_subtasks(
                archetype, cache_entry, room, seed)
            container_constraints = None
    except ValueError as e:
        return 'design_failed', {'status': 'design_failed',
                                  'reason': str(e), 'room': room,
                                  'scene_num': scene_num, 'seed': seed,
                                  'template_id': archetype['id']}, None

    # Search archetypes (esp. search_use_later) may list the same object in
    # two subtasks (reuse). spawn only ONE instance — the agent will pick and
    # re-pick the same instance. Without this dedup, counts[BasketBall]=2
    # spawns two instances and sub3's resolve finds the wrong one.
    n_instr = len(set(chosen_objects))
    n_dist_target = n_distractors_for(n_instr, room, seed)
    task_id = f"{room}{scene_num}-{seed:02d}-{archetype['id'].replace('_', '-')}"
    # Stub instructions are used ONLY in the temporary verify-only task.yaml.
    # The feasibility checker (id_trajectory_generator) ignores instructions —
    # it is driven entirely by pddl_params.subtasks — so these stubs never
    # reach a delivered task. The agent overwrites them with real macro+micro
    # when writing the FINAL task.yaml from the dumped context.
    stub_macro = archetype['macro']
    stub_micro = archetype['macro']

    # Floor-scatter archetypes: build scene_positions_scatter specs from the
    # instruction objects of place-subtasks, so they get dropped on the floor
    # (verified curated pose) and the agent must pick them up from there.
    scatter_specs = None
    if archetype.get('scatter'):
        floor_only = archetype.get('scatter_floor_only', True)
        seen = set()
        scatter_specs = []
        for st in subtasks:
            obj = st.get('object')
            if obj and obj not in seen:
                scatter_specs.append({'objectType': obj, 'count': 1,
                                      'floor_only': floor_only})
                seen.add(obj)

    write_intent_yaml(INTENT_TMP, task_id, room, scene_num, seed, subtasks,
                      n_dist_target, stub_macro,
                      container_constraints=container_constraints,
                      scene_positions_scatter=scatter_specs)
    try:
        preview_data = build_preview(str(INTENT_TMP), x_display=x_display,
                                     resolution=resolution)
    except Exception as e:
        return 'preview_failed', {'item': stub_macro,
                                   'task_id_candidate': task_id,
                                   'status': 'preview_failed',
                                   'reason': f"{type(e).__name__}: {str(e)[:300]}",
                                   'room': room, 'scene_num': scene_num,
                                   'seed': seed, 'template_id': archetype['id']}, None

    actual_distractors = len(preview_data['random_spawn']['distractors'])
    # TEMPORARY verify-only task dir, isolated under outputs/temp/ so it can never
    # be mistaken for a delivered task. Removed after the feasibility check.
    verify_dir = TEMP_DIR / '_verify_tmp' / task_id
    verify_dir.mkdir(parents=True, exist_ok=True)
    write_full_cfg(CFG_TMP, preview_data, subtasks, stub_macro, stub_micro,
                   task_id, room, scene_num, n_dist_target, actual_distractors,
                   archetype['id'])
    try:
        write_from_cfg(str(CFG_TMP), str(verify_dir))
    except Exception as e:
        shutil.rmtree(verify_dir, ignore_errors=True)
        return 'write_failed', {'item': archetype['macro'], 'task_id': task_id,
                                'status': 'write_failed', 'room': room,
                                'scene_num': scene_num, 'seed': seed,
                                'template_id': archetype['id'],
                                'reason': f"{type(e).__name__}: {str(e)[:300]}"}, None

    try:
        report = verify_run(str(verify_dir),
                            out_dir=str(OUT_DIR / 'task_verify'),
                            resolution=resolution, x_display=x_display,
                            write_trajectory=True)
    except Exception as e:
        shutil.rmtree(verify_dir, ignore_errors=True)
        return 'verify_crashed', {'item': stub_macro, 'task_id': task_id,
                                  'status': 'verify_crashed', 'room': room,
                                  'scene_num': scene_num, 'seed': seed,
                                  'template_id': archetype['id'],
                                  'reason': f"{type(e).__name__}: {str(e)[:300]}",
                                  'traceback': traceback.format_exc()[-400:]}, None

    feasible = report.get('feasible', False)
    gc = report.get('goal_conditions_met', '?')
    layout = preview_data['layout_summary']

    # Dump the full preview context ONLY when feasible. The agent reads this to
    # author real macro+micro and write the final task.yaml. No context is left
    # for infeasible attempts — they are discarded entirely.
    ctx_path = None
    if feasible:
        ctx_dir = TEMP_DIR / 'phase_preview'
        ctx_dir.mkdir(parents=True, exist_ok=True)
        ctx = {
            'task_id': task_id,
            'out_task_dir': str(COMPOSE_DIR),
            'eval_set': EVAL_SET,
            'room': room, 'scene_num': scene_num, 'seed': seed,
            'template_id': archetype['id'], 'archetype_macro': archetype['macro'],
            'subtasks': subtasks,
            'preview': {
                'scene': preview_data['scene'],
                'random_spawn': preview_data['random_spawn'],
                'object_poses': preview_data['object_poses'],
                'object_toggles': preview_data['object_toggles'],
                'init_action': preview_data['init_action'],
            },
            'layout_summary': layout,
            'distractors': preview_data['random_spawn'].get('distractors'),
            'scatter_result': preview_data.get('scatter_result'),
            'container_constraints': container_constraints,
            'n_distractors_planned': n_dist_target,
            'n_distractors_actual': actual_distractors,
        }
        ctx_path = ctx_dir / f'{task_id}.json'
        ctx_path.write_text(json.dumps(ctx, ensure_ascii=False, indent=2))
    # Always remove the verify-only temp dir — it carried stub instructions and
    # must never be delivered. The agent writes the final task.yaml fresh.
    shutil.rmtree(verify_dir, ignore_errors=True)

    result = {
        'item': archetype['macro'], 'task_id': task_id,
        'template_id': archetype['id'], 'room': room, 'scene_num': scene_num,
        'seed': seed, 'num_subtasks': len(subtasks),
        'n_instruction_objects': n_instr,
        'n_distractors': actual_distractors,
        'distractor_ratio': round(actual_distractors / max(n_instr, 1), 2),
        'facilities': layout['facilities'],
        'receptacles': layout.get('receptacles', {}),
        'feasible': feasible, 'goal_conditions_met': gc,
    }
    if feasible:
        result['status'] = 'ok'
        return 'ok', result, str(ctx_path)
    first_fail = next((e for e in report.get('exec_log', [])
                       if not e.get('ok', True)), None)
    result['status'] = 'verify_infeasible'
    result['reason'] = (first_fail.get('err') if first_fail else
                        report.get('failure_reason', 'unknown'))
    if first_fail:
        result['failed_op'] = first_fail.get('op')
    return 'verify_infeasible', result, None


def try_with_ladder(room, scene_num, cache_entry, x_display, resolution,
                    archetype_idx, seed_attempts):
    """Cycle through archetypes (start at archetype_idx), and for each try
    seeds in seed_attempts. Returns first 'ok' result (with ctx_path) or last
    failure."""
    archetypes = ARCHETYPES[room]
    n_arch = len(archetypes)
    fac = cache_entry['facilities']
    workable = []
    for off in range(n_arch):
        arch = archetypes[(archetype_idx + off) % n_arch]
        if archetype_supports_floorplan(arch, cache_entry, room):
            workable.append(arch)
    if not workable:
        return {'item': f'{room}/{scene_num}', 'status': 'no_archetype',
                'reason': 'no archetype supports this floorplan',
                'room': room, 'scene_num': scene_num,
                'facilities': fac, 'pickupable_count': len(cache_entry['pickupable'])}

    last = None
    for arch in workable:
        for seed in seed_attempts:
            status, result, ctx_path = attempt(room, scene_num, arch, seed,
                                               x_display, resolution)
            last = result
            if status == 'ok':
                result['ctx_path'] = ctx_path
                return result
    return last


def archetype_supports_floorplan(archetype, cache_entry, room):
    """Quick pre-check: archetype's slot actions must be satisfiable by
    floorplan capabilities. Prevents guaranteed-fail attempts."""
    if archetype.get('search'):
        if not _available_hidden_containers(cache_entry):
            return False
    if room in NO_TOGGLE_ROOMS:
        for slot in archetype['slots']:
            if slot[0] in ('toggle_on', 'toggle_off', 'look_at'):
                return False
    fac = cache_entry['facilities']
    inherent = set(cache_entry['pickupable'])
    slots = archetype['slots']
    needs_heat = needs_cool = needs_clean = needs_slice = False
    needs_look_at = needs_toggle = needs_movable = False
    for slot in slots:
        a = slot[0]
        if a == 'heat': needs_heat = True
        elif a == 'cool': needs_cool = True
        elif a == 'clean': needs_clean = True
        elif a in ('slice', 'slice_place'): needs_slice = True
        elif a == 'look_at': needs_look_at = True
        elif a in ('toggle_on', 'toggle_off'): needs_toggle = True
        elif a == 'movable_recep': needs_movable = True
    if needs_heat and not fac.get('Microwave'): return False
    if needs_heat and not (set(ACTION_OBJECT_SETS['heat']) & inherent): return False
    if needs_cool and not fac.get('Fridge'): return False
    if needs_cool and not (set(ACTION_OBJECT_SETS['cool']) & inherent): return False
    if needs_clean:
        if not (fac.get('SinkBasin') and fac.get('Faucet')): return False
        if room == 'Bathroom':
            if not (set(CLEANABLE_BATH) & inherent): return False
        else:
            if not (set(ACTION_OBJECT_SETS['clean']) & inherent): return False
    if needs_slice and not (set(ACTION_OBJECT_SETS['slice']) & inherent): return False
    if needs_look_at:
        if not fac.get('DeskLamp'): return False
        if not (set(LOOK_AT_OBJS) & inherent): return False
    if needs_toggle:
        if not (fac.get('FloorLamp') or fac.get('DeskLamp')): return False
    has_place = any(s[0] in ('place', 'heat', 'cool', 'clean', 'slice_place',
                             'movable_recep')
                    for s in slots)
    if has_place and not reachable_receps(cache_entry, room): return False
    if needs_movable and not (set(MOVABLE_RECEPTACLES) & inherent): return False
    return True


def plan_one(room, scene_num, cache_entry, archetype_idx, seed):
    """Plan ONE intent's subtask skeleton from the floorplan cache — no simulator,
    no instruction authoring. Returns an intent dict (name/room/scene/seed/
    subtasks/n_distractors + optional container_constraints/scene_positions_scatter)
    or None if no archetype supports this floorplan.

    The intent's `goal_idea` is guidance for the authoring subagent; the subagent
    authors the real macro+micro from the preview layout. This function never
    writes instructions or runs AI2-THOR."""
    archetypes = ARCHETYPES[room]
    n_arch = len(archetypes)
    for off in range(n_arch):
        arch = archetypes[(archetype_idx + off) % n_arch]
        if not archetype_supports_floorplan(arch, cache_entry, room):
            continue
        try:
            if arch.get('search'):
                subtasks, chosen, container_constraints = \
                    instantiate_search_subtasks(arch, cache_entry, room, seed)
            else:
                subtasks, chosen = instantiate_subtasks(
                    arch, cache_entry, room, seed)
                container_constraints = None
        except ValueError:
            continue
        n_instr = len(set(chosen))
        n_dist_target = n_distractors_for(n_instr, room, seed)
        task_id = f"{room}{scene_num}-{seed:02d}-{arch['id'].replace('_', '-')}"
        # scene_positions_scatter specs for floor-scatter archetypes, built from
        # the place-subtask instruction objects.
        scatter_specs = None
        if arch.get('scatter'):
            floor_only = arch.get('scatter_floor_only', True)
            seen = set()
            scatter_specs = []
            for st in subtasks:
                obj = st.get('object')
                if obj and obj not in seen:
                    scatter_specs.append({'objectType': obj, 'count': 1,
                                          'floor_only': floor_only})
                    seen.add(obj)
        intent = {
            'name': task_id,
            'room_type': room,
            'scene_num': scene_num,
            'seed': seed,
            # goal idea only — the authoring subagent writes real macro+micro.
            # Stored under `macro` so task_builder preview's validate_config
            # (which requires a non-empty instruction) passes; the subagent
            # overwrites it with a layout-aware macro when writing task.yaml.
            'macro': arch['macro'],
            'goal_idea': arch['macro'],
            'n_distractors': n_dist_target,
            'subtasks': subtasks,
            'eval_set': EVAL_SET,
            'out_task_dir': str(COMPOSE_DIR),
        }
        if container_constraints:
            intent['container_constraints'] = container_constraints
        if scatter_specs:
            intent['scene_positions_scatter'] = scatter_specs
        return intent
    return None


def run_plan_only(args, cache, seed_attempts):
    """Plan-only mode: sample N diverse intent specs from the floorplan cache and
    write one intent YAML per task to --intent-dir. No simulator, no instructions.

    Each intent carries a `goal_idea` (the archetype's macro template) as guidance
    for the authoring subagent — it is NOT a final instruction. The orchestrator
    dispatches one subagent per intent to author the real macro+micro from the
    preview layout and write task.yaml (see references/batch_tools.md)."""
    intent_dir = ROOT / args.intent_dir
    intent_dir.mkdir(parents=True, exist_ok=True)

    SCENE_TO_ROOM = {}
    for room, scenes in ROOM_SCENES.items():
        for sn in scenes:
            SCENE_TO_ROOM[sn] = room

    if args.target_scenes:
        target = json.loads(args.target_scenes)
        plan = []
        for sn in target:
            room = SCENE_TO_ROOM.get(sn)
            if room is None:
                print(f'WARN: scene {sn} not in any room; skipped', flush=True)
                continue
            plan.append((room, sn, 0))
    else:
        plan = []
        for room in args.rooms:
            scenes = ROOM_SCENES[room][:args.limit_per_room]
            n_arch = len(ARCHETYPES[room])
            for i, sn in enumerate(scenes):
                plan.append((room, sn, i % n_arch))

    # --n caps the total count.
    if args.n and args.n > 0:
        plan = plan[:args.n]

    written, skipped = [], []
    for idx, (room, scene_num, tidx) in enumerate(plan):
        fp_key = f'FloorPlan{scene_num}'
        cache_entry = cache.get(fp_key, {})
        intent = None
        for seed in seed_attempts:
            intent = plan_one(room, scene_num, cache_entry, tidx, seed)
            if intent:
                break
        if not intent:
            skipped.append({'room': room, 'scene_num': scene_num,
                            'reason': 'no archetype/seed supports floorplan'})
            print(f'[{idx + 1}/{len(plan)}] {room}/{scene_num} -> skipped '
                  f'(no supporting archetype)', flush=True)
            continue
        out_path = intent_dir / f"{intent['name']}.yaml"
        with open(out_path, 'w') as f:
            yaml.safe_dump(intent, f, sort_keys=False, allow_unicode=True)
        written.append(intent['name'])
        n_sub = len(intent['subtasks'])
        print(f'[{idx + 1}/{len(plan)}] {room}/{scene_num} -> {out_path.name} '
              f'({n_sub} subtasks, idea={intent["goal_idea"]!r})',
              flush=True)

    report = {
        'mode': 'plan_only',
        'planned': len(plan),
        'written': len(written),
        'skipped': len(skipped),
        'intent_dir': str(intent_dir),
        'written_ids': written,
        'skipped': skipped,
    }
    report_path = TEMP_DIR / 'plan_only_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n=== plan-only DONE ===\n"
          f"planned={len(plan)} written={len(written)} skipped={len(skipped)}\n"
          f"intents -> {intent_dir}\nreport -> {report_path}", flush=True)


def build_summary(results, planned, t_start):
    success = sum(1 for r in results if r.get('status') == 'ok')
    skipped = sum(1 for r in results if r.get('status') != 'ok')
    by_status = {}
    by_room = {}
    for r in results:
        s = r.get('status', '?')
        by_status[s] = by_status.get(s, 0) + 1
        rm = r.get('room', '?')
        by_room.setdefault(rm, {'ok': 0, 'fail': 0})
        by_room[rm]['ok' if s == 'ok' else 'fail'] += 1
    ratios = [r.get('distractor_ratio') for r in results
              if r.get('distractor_ratio') is not None]
    return {
        'planned': planned,
        'total': len(results),
        'success': success,
        'skipped': skipped,
        'by_status': by_status,
        'by_room': by_room,
        'distractor_ratio_avg': round(sum(ratios) / len(ratios), 2) if ratios else None,
        'distractor_ratio_min': min(ratios) if ratios else None,
        'elapsed_sec': round(time.time() - t_start, 1),
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache', default='outputs/temp/floorplan_cache.json')
    parser.add_argument('--rooms', nargs='+',
                        default=['Kitchen', 'Bathroom', 'LivingRoom', 'Bedroom'],
                        choices=['Kitchen', 'Bathroom', 'LivingRoom', 'Bedroom'])
    parser.add_argument('--limit-per-room', type=int, default=30)
    parser.add_argument('--seeds', default='0,7,13,23,42',
                        help='Comma-separated seeds to try per archetype.')
    parser.add_argument('--x-display', default='1')
    parser.add_argument('--resolution', type=int, default=300)
    parser.add_argument('--report-name', default='compose_batch_120_v2_report.json')
    parser.add_argument('--out-dir', default='outputs/temp',
                        help='[verify mode] Where to write task_verify/ and report.')
    parser.add_argument('--out-task-dir', default='pgibench/envs/pgi_alfred/data/tasks/compose',
                        help='[verify mode] Where the agent later writes task.yaml.')
    parser.add_argument('--eval-set', default='compose',
                        help='eval_set tag stamped into each task.yaml.')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--target-scenes', default=None,
                        help='JSON array of scene numbers, e.g. "[301,302]". '
                             'Overrides --rooms / --limit-per-room when set.')
    parser.add_argument('--archetype', default=None,
                        help='Force this archetype id (e.g. "search_and_stow") '
                             'for every scene. Skips the round-robin rotation.')
    parser.add_argument('--plan-only', action='store_true',
                        help='Plan-only mode: sample intent specs from the '
                             'floorplan cache and write one intent YAML per task '
                             'to --intent-dir. No simulator, no instructions.')
    parser.add_argument('--n', type=int, default=0,
                        help='[plan-only] Target number of intents to plan.')
    parser.add_argument('--intent-dir', default='outputs/temp/batch_intents',
                        help='[plan-only] Where to write intent YAMLs.')
    args = parser.parse_args()

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    global OUT_DIR, COMPOSE_DIR, EVAL_SET
    OUT_DIR = ROOT / args.out_dir
    COMPOSE_DIR = ROOT / args.out_task_dir
    EVAL_SET = args.eval_set
    (OUT_DIR / 'task_verify').mkdir(parents=True, exist_ok=True)
    COMPOSE_DIR.mkdir(parents=True, exist_ok=True)

    cache = json.loads(Path(args.cache).read_text())
    seed_attempts = [int(s) for s in args.seeds.split(',')]

    if args.plan_only:
        return run_plan_only(args, cache, seed_attempts)

    # ---- verify mode (legacy) ----
    plan = []
    SCENE_TO_ROOM = {}
    for room, scenes in ROOM_SCENES.items():
        for sn in scenes:
            SCENE_TO_ROOM[sn] = room

    if args.target_scenes:
        target = json.loads(args.target_scenes)
        for sn in target:
            room = SCENE_TO_ROOM.get(sn)
            if room is None:
                print(f'WARN: scene {sn} not in any room; skipped', flush=True)
                continue
            plan.append((room, sn, 0))
    else:
        for room in args.rooms:
            scenes = ROOM_SCENES[room][:args.limit_per_room]
            n_arch = len(ARCHETYPES[room])
            for i, sn in enumerate(scenes):
                plan.append((room, sn, i % n_arch))

    arch_idx_per_room = None
    if args.archetype:
        arch_idx_per_room = {}
        for room in ARCHETYPES:
            for idx, a in enumerate(ARCHETYPES[room]):
                if a['id'] == args.archetype:
                    arch_idx_per_room[room] = idx
                    break
        if not arch_idx_per_room:
            print(f'ERROR: archetype {args.archetype!r} not found in any room',
                  file=sys.stderr)
            sys.exit(2)

    if args.skip_existing:
        existing = {p.name.split('-')[0] for p in COMPOSE_DIR.glob('*') if p.is_dir()}
        kept = []
        for room, sn, tidx in plan:
            tag = f'{room}{sn}'
            if tag in existing:
                continue
            kept.append((room, sn, tidx))
        skip_count = len(plan) - len(kept)
        plan = kept
        print(f'--skip-existing: {skip_count} scenes already have tasks; '
              f'{len(plan)} remain.', flush=True)

    if arch_idx_per_room is not None:
        new_plan = []
        for r, s, _ in plan:
            if r in arch_idx_per_room:
                new_plan.append((r, s, arch_idx_per_room[r]))
            else:
                print(f'WARN: archetype not in {r}; skipped scene {s}', flush=True)
        plan = new_plan

    results = []
    t_start = time.time()
    for idx, (room, scene_num, tidx) in enumerate(plan):
        t0 = time.time()
        fp_key = f'FloorPlan{scene_num}'
        cache_entry = cache.get(fp_key, {})
        print(f"\n=== [{idx + 1}/{len(plan)}] {room} scene={scene_num} "
              f"template={tidx} (pickupable={len(cache_entry.get('pickupable', []))}) ===",
              flush=True)
        try:
            r = try_with_ladder(room, scene_num, cache_entry, args.x_display,
                                args.resolution, tidx, seed_attempts)
        except Exception as e:
            r = {'item': f"{room}/{scene_num}", 'status': 'crashed',
                 'reason': f"{type(e).__name__}: {str(e)[:200]}",
                 'traceback': traceback.format_exc()[-400:],
                 'room': room, 'scene_num': scene_num}
        r['elapsed_sec'] = round(time.time() - t0, 1)
        results.append(r)
        ctx_note = f" ctx={r['ctx_path']}" if r.get('ctx_path') else ''
        print(f"  -> {r['status']} ({r['elapsed_sec']}s) "
              f"{r.get('task_id', r.get('task_id_candidate', ''))}{ctx_note}", flush=True)

        summary = build_summary(results, len(plan), t_start)
        with open(OUT_DIR / args.report_name, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    summary = build_summary(results, len(plan), t_start)
    with open(OUT_DIR / args.report_name, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n=== DONE ===\n"
          f"total={summary['total']} success={summary['success']} "
          f"skipped={summary['skipped']} elapsed={summary['elapsed_sec']}s\n"
          f"report -> {OUT_DIR / args.report_name}", flush=True)


if __name__ == '__main__':
    main()
