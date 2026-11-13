# Search tasks — hidden-object exploration + memory

Standard subtasks assume the instruction object is visible from the spawn
point. **Search archetypes** hide objects inside closed containers, so the
agent must explore (open cabinets/drawers) to find them. This adds two
capability dimensions otherwise untested:

- **Spatial exploration**: which container to open first, in what order
- **Memory**: which containers are already open, what was found where

## Two flavors

### A. `search_and_stow` (exploration)

N objects hidden in closed containers, all must end up on one open receptacle.

```yaml
slots: [('place', 3)]
search: true
search_count: 2     # 2 of the 3 objects are hidden; 1 stays visible
```

Tests: open-container coordination, persistence (try multiple containers).

### B. `search_use_later` (long-horizon memory)

Object A is hidden, must first be brought to a staging spot. Then 2 unrelated
subtasks intervene. Finally A is moved to its real destination.

```yaml
slots: [('place', 1), ('place', 2, 3), ('place', 1)]
search: true
search_reuse: true
```

Tests: where did I put A? Can the agent recover that memory after intervening
work?

## How hiding works (mechanism)

`scene_snapshot.spawn_into_container(env, obj_type, container_type)` runs
**after** `random_initialize_scene` (which spawns randomly). It picks one
existing instance, opens the target container, picks the object up, puts it
inside, and closes the container. The resulting pose is captured by
`snapshot_scene` and frozen into `object_poses`.

`task_builder.build_preview` reads `intent.container_constraints` and applies
each constraint via `spawn_into_container` before re-snapshotting.

At verification time, `id_trajectory_generator._exec_subtask` already calls
`_open_container_if_closed` before pick — so the hidden-object path "just
works" without generator changes.

## Where search archetypes fit

| Room | Container options | Notes |
|---|---|---|
| Kitchen | Cabinet, Drawer | Most FloorPlans have ≥1; safe default |
| LivingRoom | Drawer (TVStand/Drawer) | Some scenes lack Cabinet — probe cache filters |
| Bedroom | Drawer, Safe, Cabinet | Strong fit; Bedroom currently saturated with pure-tidy |
| Bathroom | Cabinet (under-sink) | Works but Bathroom is facility-poor; lower priority |

When planning a search intent, check the floorplan cache
(`outputs/temp/floorplan_cache.json`) for closed containers in that FloorPlan — if
none exist, skip that scene for search (the hiding step has nowhere to put the
object).

## Macro style for search tasks

Macro should mention "find" / "hidden" but NOT name the container type
(that gives away the exploration puzzle), and **must list each target
object + its destination receptacle** (otherwise the agent cannot tell
instruction objects apart from distractors). Same-name objects (e.g. two
Mug instances in `search_use_later`) are not differentiated. Examples:

- ✅ "Some stuff got hidden in the cabinets — find them for me. Put the plate and pencil on the coffee table, and put the newspaper and key chain on the side table."
- ✅ "I hid something — find it, do a few things, then bring it back. Put the mug, credit card, cell phone, and mug on the dresser."
- ❌ "Open the top drawer and get the keychain." (names container — no search)
- ❌ "Find the items hidden in the cabinets and put them on the counter." (no objects named — agent cannot tell targets from distractors)

## Verifying a search task

Run the standard authoring pipeline (SKILL.md Stage 1) with a `container_constraints`
intent, then Stage 2 trajectory generation. Inspect
`outputs/temp/task_verify/<name>/sim_log.json`: confirm at least one `op: open`
precedes the corresponding `op: pick` for the hidden object. If pick succeeds
without a prior open, the `container_constraints` didn't take effect (object
spawned outside the container) — rerun preview.

Note: `spawn_into_container` prefers **Cabinet** over Drawer — Drawer open
fails in preview (agent height/orientation mismatch at low Y). LivingRoom /
Bedroom scenes without Cabinet fall back to Drawer but may fail verification.
