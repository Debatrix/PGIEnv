# Task Contract — what PGIAlfEnv requires to load a task

## Load chain

`_load_dataset` scans `data/tasks/<eval_set>/*/task.yaml` and builds one entry per dir
→ `_reset_controller` calls `utils.load_task(task)`
→ `task_format.load_task_dir` reads `data/<task>/task.yaml` (+ optional
  `trajectory.json`) and assembles them into the traj dict that `CompositeTask` consumes
→ `restore_scene` + `set_task` → `get_task('composite', ...)` → `CompositeTask`.

A task lives in a directory `data/<eval_set>/<name>/` containing:
1. `task.yaml` — task definition + scene init (**required**).
2. `trajectory.json` — expert/exec trajectory (**optional**; absent ⇒ evaluate-only).

## task.yaml schema (v2)

```yaml
schema_version: 2
task_id: <Room><scene_num>-<seed>-<slug>   # e.g. Kitchen8-07-heat-potato
task_type: composite            # always composite; single-object = 1-subtask composite
eval_set: <set>
instructions:                   # dual NL (schema v2). loader falls back to legacy
  macro: "<natural spoken request with key objects/destinations folded in>"
  micro: "<step-by-step procedure; same goal, '; '-joined, every object named>"
scene:
  floor_plan: FloorPlan16
  scene_num: 16
  room_type: Kitchen
  random_spawn:                 # deterministic spawn rule (see "Two-stage restore")
    seed: 7
    object_counts: {Apple: 1}
    distractors: {Mug: 1}       # values may be >1 (repeat sampling enabled)
    scene_positions_scatter:    # optional: re-pose objects to verified curated poses
      - {objectType: Tomato, count: 1, floor_only: true}   # floor-level (y<0.3)
  object_poses: [{objectName: <Type_hexhash instance-level name>, position, rotation}, ...]
  object_toggles: [{objectType, isOn}, ...]
  init_action: {action: TeleportFull, x, y, z, rotation, horizon, standing: true}
  dirty_and_empty: false
pddl_params:
  object_target/parent_target/mrecep_target/toggle_target/object_sliced   # first subtask's, for compat
  subtasks:                       # ★ CompositeTask reads this; without it → fake success
    - task_type: <type>
      pddl_params: {object_target, parent_target, mrecep_target, toggle_target, object_sliced}
completion: {mode: all}
```

`object_poses[].objectName` is an **instance-level** `Type_hexhash` (e.g.
`Apple_b66a32b5`); `snapshot_scene` outputs by instance name and `SetObjectPoses`
pins each object exactly by this name. Legacy data migrated from elsewhere may
carry a bare objectType, which the loader also accepts (type matching for
back-compat). The `subtasks[].pddl_params` field names are fixed as
`object_target / parent_target / mrecep_target / toggle_target / object_sliced`.

## Two-stage restore (spawn to create + SetObjectPoses to pin)

Reloading a task cannot restore the scene with `SetObjectPoses` alone — the
spawned objects (Mug/Apple/…) created by `InitialRandomSpawn` do not exist in a
fresh controller, and `SetObjectPoses` can only move scene-fixed objects. So
restore happens in two steps (already implemented in `PGIAlfEnv` and
`id_trajectory_generator`):

1. **spawn to create**: rerun `InitialRandomSpawn` per `scene.random_spawn`
   (seed + object_counts + distractors) to re-add the spawned objects to the
   scene.
2. **pin instances**: then run `SetObjectPoses` with `scene.object_poses`
   (instance-level names) to restore every object (spawned and scene-fixed) to
   exactly the pose at snapshot time.

`random_spawn` stores the RULE, not poses; `object_poses` stores the POSES. Both
are required.

`random_spawn.scene_positions_scatter` (optional) records the scatter rule, but
**the restore chain needs no special handling for it** — the post-scatter final
poses are already captured in `object_poses` by `snapshot_scene`, and the
two-step restore's `SetObjectPoses` pins them directly (e.g. a floor-scattered
object returns to its floor-level Y after restore).

## scene_positions scatter (pick-up-from-floor semantics)

`data/scene_positions/FloorPlan<N>.json` stores every object pose ever seen in an
existing ALFRED task for that FloorPlan (each comes from an executed trajectory,
so it is verified feasible). `scatter_from_positions` (scene_snapshot.py) picks
poses from this dataset per spec and re-poses existing instances with
`SetObjectPoses`:

- `floor_only: true` filters to floor-level poses (y<0.3), producing
  "pick up from the floor" semantics.
- Scene-appropriateness is inherent: the dataset is collected per FloorPlan, so a
  bedroom scene has no vegetable poses.
- Only **existing instances** can be re-posed (object_counts-spawned or
  scene-inherent); non-inherent, un-spawned types are skipped.
- Distractors (`distractors`) allow multiple instances of the same type
  (with-replacement sampling), raising discrimination difficulty.

## trajectory.json schema (optional)

```json
{
  "high_pddl": [{high_idx, discrete_action, planner_action}, ...],
  "low_actions": [{high_idx, api_action, discrete_action}, ...],
  "images": [...]
}
```
When absent, `load_task` synthesizes a minimal fallback plan
(`high_pddl=[End]`), so `num_subgoals=0`. This does NOT affect SR/GC (judging
reads only scene state) nor episode termination (PGIAlfEnv computes its own
`done`, overriding transition_reward's).

For id-driven generated tasks, `trajectory.json` stores an `exec_log` plus the
fallback plan — enough to record how feasibility was achieved.

## Why no trajectory splicing

The old `composite_generator` spliced existing expert trajectories and required a
single scene containing every subtask's trajectory — most NL combos were not
generatable. The new pipeline (`task_builder` + `id_trajectory_generator`) spawns
objects randomly and drives each subtask with id primitives, so any legal combo
works.

## Common failure modes

- **Fake success**: `pddl_params.subtasks` missing → `CompositeTask` has zero
  subtasks → `goal_satisfied` returns `all([])=True`. `task_builder` always
  writes subtasks.
- **Goal conflict**: the same object placed into two different receptacles across
  subtasks → unsatisfiable in the final state. `validate` warns; use distinct
  objects.
- **feasible=false after generation**: a required facility is absent, or an
  object is unreachable. Change `scene_num` / `seed` / objects and rerun.
