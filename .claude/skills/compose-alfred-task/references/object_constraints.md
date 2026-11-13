# Object / Receptacle / Action Legality

Source of truth: `pgibench/envs/pgi_alfred/gen/constants.py`. The tables here are the
values `validate_composite.py` checks. When in doubt, read that file.

## Task-type priority — prefer composites, simple tasks as fallback

Not all legal task types are equally informative. Design each subtask chain to
**lead with a Priority-A composite** and use Priority-B (simple/atomic) tasks only
to complete a coherent goal, or when Priority-A conditions can't be met.

**Priority A — state-change composites (use first; the backbone of a meaningful task):**
`pick_heat_then_place_in_recep`, `pick_cool_then_place_in_recep`,
`pick_clean_then_place_in_recep`, `pick_slice_then_place_in_recep`,
`pick_and_place_with_movable_recep`, `look_at_obj_in_light`.

**Priority B — simple/atomic (fallback, or composing a meaningful macro):**
`pick_and_place_simple` (tidy / floor-scatter / search chains, and the slice+place
carrier via `sliced: true`), `slice_in_place`, `toggle_object_on/off`,
`open/close_receptacle`.

Use Priority-B tasks standalone **only when Priority-A conditions can't be
satisfied** (e.g. no Microwave → can't heat; no Sliceable object inherent → can't
slice_place). Otherwise fold them into a Priority-A-led chain: e.g. a hot-meal
task leads with `heat`/`slice_place`, then uses `pick_and_place_simple` only to
plate it and lay out utensils. A tidy-room or search task is an acceptable
Priority-B-led design (its purpose is the rearrangement/exploration itself).

## Task type → required params & valid rooms

| task_type | required | valid rooms |
|---|---|---|
| pick_and_place_simple | object, parent | Kitchen, LivingRoom, Bathroom, Bedroom |
| pick_two_obj_and_place | object, parent | Kitchen, LivingRoom, Bathroom, Bedroom |
| look_at_obj_in_light | object, toggle | LivingRoom, Bedroom |
| pick_clean_then_place_in_recep | object, parent | Kitchen, Bathroom |
| pick_heat_then_place_in_recep | object, parent | Kitchen |
| pick_cool_then_place_in_recep | object, parent | Kitchen |
| pick_slice_then_place_in_recep | object, parent | Kitchen, LivingRoom, Bathroom, Bedroom |
| pick_and_place_with_movable_recep | object, parent, movable_recep | Kitchen, LivingRoom, Bedroom |

Atomic task types (single primitive; mix freely with the ALFRED types above in `subtasks`):

| task_type | required | valid rooms |
|---|---|---|
| toggle_object_on | toggle | Kitchen, LivingRoom, Bathroom, Bedroom |
| toggle_object_off | toggle | Kitchen, LivingRoom, Bathroom, Bedroom |
| open_receptacle | parent | Kitchen, LivingRoom, Bathroom, Bedroom |
| close_receptacle | parent | Kitchen, LivingRoom, Bathroom, Bedroom |
| slice_in_place | object | Kitchen, LivingRoom, Bathroom, Bedroom |

- `toggle` must be Toggleable {DeskLamp, FloorLamp}.
- `parent` for open/close_receptacle must be openable {Box, Cabinet, Drawer, Fridge, Microwave, Safe}.
- `object` for slice_in_place / pick_slice_then_place_in_recep must be Sliceable {Apple, Bread, Egg, Lettuce, Potato, Tomato}. For pick_slice_then_place_in_recep also set `sliced: true` (see below).

Optional param: `sliced` (bool, default false).

Scene number ranges: Kitchen 1–30, LivingRoom 201–230, Bedroom 301–330, Bathroom 401–430.

## pick_slice_then_place_in_recep (first-class slice+place)

This is a real composite task type (goal class `PickSliceThenPlaceInRecepTask`),
now registered for validation and id-driven generation. It slices the object then
places it in the receptacle — use it instead of the old `slice_in_place` +
`pick_and_place_simple` (sliced:true) two-step wherever you want slice+place as a
single subtask. **Always set `sliced: true`** on the subtask: it propagates to
`pddl_params.object_sliced`, and `BaseTask.get_targets()` then tags the target as
e.g. `AppleSliced`, which the goal check (`in-recep AND sliced AND both`) relies on.

## Scene-appropriateness (room ↔ object matching)

Two mechanisms keep objects sensible for their room:

- **`REPRESENTATIVE_OBJECTS`** (scene_snapshot.py) — the per-room distractor pool
  (Kitchen: Apple/Bread/Mug/...; Bedroom: Book/Watch/Pillow/...; no vegetables in
  bedroom). `sample_distractors` draws from this and now samples WITH replacement,
  so repeat types (e.g. {Mug: 3}) are allowed for added discrimination difficulty.
- **`scene_positions/FloorPlan<N>.json`** — curated verified poses per FloorPlan.
  Scene-appropriateness is inherent: a Bathroom dataset holds no vegetable poses
  because no bathroom task ever placed them there. `scene_positions_scatter` re-poses
  objects ONLY to poses that exist in that FloorPlan's dataset.

## Action → object set (object must be in the set)

- heat  → Heatable  (gen/constants.py `VAL_ACTION_OBJECTS['Heatable']`, 12 items)
- cool  → Coolable  (18 items)
- clean → Cleanable (25 items)
- slice → Sliceable {Apple, Bread, Egg, Lettuce, Potato, Tomato}
- look_at toggle → Toggleable {DeskLamp, FloorLamp}
- movable_recep → MOVABLE_RECEPTACLES {Bowl, Box, Cup, Mug, Plate, Pan, Pot}

Valid object types: any key of `OBJ_PARENTS` (~113 types). Valid receptacles: any key of
`VAL_RECEPTACLE_OBJECTS` (36 types). An object is placeable into a receptacle only if it
appears in `VAL_RECEPTACLE_OBJECTS[receptacle]` (a soft check — warns, since the sim may
still allow it).

## How sliced objects appear in the data

Trajectory directories use the suffixed name, e.g. `TomatoSliced`. In YAML write the
base object (`Tomato`) and set `sliced: true`; the generator maps it to `TomatoSliced`
when matching directories and sets `object_sliced: true` in `pddl_params`.

## Discovering generatable combos yourself

The bundled subset is small. To find what actually exists in a room:

```python
import os, collections
JSON='pgibench/envs/pgi_alfred/data/json_2.1.0'
combos=collections.defaultdict(list)
for e in os.listdir(JSON):
    p=e.split('-')
    if len(p)>=5:
        try: sn=int(p[4])
        except: continue
        combos[sn].append((p[0],p[1],p[2],p[3]))  # task_type,object,mrecep,parent
# scenes with >=2 task types can host a composite
for sn,c in sorted(combos.items()):
    if len({x[0] for x in c})>=2: print(sn, c)
```

A composite subtask matches a directory `task_type-object-mrecep-parent-scene` where:
- non-movable, non-look_at tasks: mrecep="None", parent=<your parent>
- pick_and_place_with_movable_recep: mrecep=<your movable_recep>, parent=<your parent>
- look_at_obj_in_light: mrecep="None", parent=<your toggle>
