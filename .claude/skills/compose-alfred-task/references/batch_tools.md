# Batch tools — bulk task generation & maintenance

Bulk generation (e.g. "generate 30 kitchen tasks") is run as a **subagent-per-task
flow**: the orchestrator plans diverse intents, then dispatches one subagent per
intent that runs the full authoring pipeline (preview → read layout → author
macro+micro → write `task.yaml`). Each subagent owns one task's complete context;
the orchestrator never loads dozens of layout JSONs into its own context, and the
per-task work runs in parallel.

Instructions are authored **only** by the authoring subagent, from the real
preview layout. There is no stub, no placeholder, and no backfill stage — a task
is not delivered until its macro and micro are authored against a real scene.

> **Bundled scripts**: the helper scripts ship inside this skill under
> `scripts/` — `probe_floorplans.py` (one-time floorplan cache),
> `batch_compose.py` (intent planning), and `verify_init_state.py` (Stage-2
> initial-state check). Invoke them from the repo root as
> `python .claude/skills/compose-alfred-task/scripts/<name>.py ...` (they
> auto-detect the repo root, so they run from anywhere).
>
> **Do not generate, copy, or modify scripts.** Stage 2 / batch orchestration
> must call only these bundled scripts plus the project's own commands
> (`task_builder`, `id_trajectory_generator`, `pgibench.main`). Never author a
> throwaway `.py`, never hand-roll a `python -c` one-liner to load a task, and
> never edit anything under `scripts/`. The bundled scripts are the single
> source of truth — they load tasks exactly the way the evaluator does, so
> off-script checks silently drift from real evaluation behavior.

## floorplan_cache.json + probe_floorplans.py (one-time planning probe)

`InitialRandomSpawn` only spawns types inherent to each FloorPlan. Planning an
intent against `REPRESENTATIVE_OBJECTS` directly loses ~half the attempts because
the chosen FloorPlan may lack the object. Probe eliminates this by recording, per
FloorPlan, the inherent pickupable types, receptacle heights, and facilities.

```bash
# Probe all 120 FloorPlans once, cache inherent pickupables + receptacle Y +
# facilities. Re-runnable; reconciles new entries only. Needs an X server:
python -m pgibench.envs.pgi_alfred.scripts.start_xvfb 1
python .claude/skills/compose-alfred-task/scripts/probe_floorplans.py
#   -> writes outputs/temp/floorplan_cache.json (~25 min for all 120 FloorPlans)
# Optional flags: --out <path>  --scenes '[8,305]'  --x-display 1  --resolution 300
```

Each entry looks like:
```json
{ "scene_num": 8, "room": "Kitchen",
  "pickupable": ["Apple","Bowl","Bread","Egg","Fork","Mug","Plate","Potato",...],
  "receptacles": {"CounterTop": 0.9, "DiningTable": 0.9, ...},  // recep -> Y
  "facilities": {"Microwave": true, "Fridge": true, "SinkBasin": true, "Faucet": true} }
```

When planning an intent, **always pick instruction objects from
`cache[fp].pickupable`** and check `facilities` for state-changing subtasks.

## Intent planning (orchestrator)

The orchestrator's job is to emit N **diverse, realizable** intent specs — one per
task — and hand each to a subagent. For each intent:

1. **Pick a FloorPlan** from the cache whose `pickupable` + `facilities` can host
   the chosen pattern (`references/examples.md`) and `task_type_balance.md`'s
   coverage matrix. Spread across rooms/scenes/task-types so the set is balanced.
2. **Pick 1–4 subtasks** that are legal for that room (`object_constraints.md`)
   and use objects from `cache[fp].pickupable`. Use distinct objects per
   place-subtask (avoids goal conflicts). This is the subtask *skeleton* (task_type
   + object; parent may be a preferred receptacle or left for the subagent to fix
   from the layout).
3. **Write the intent spec** (the subagent's input): a one-sentence goal **idea**
   (e.g. "tidy the kitchen — several items scattered, stow them on the counter"),
   the room/scene_num/seed, the subtask skeleton, `n_distractors`, and any
   `scene_positions_scatter` / `container_constraints` (floor-scatter / search
   tasks). The goal idea is guidance for the subagent — it is **not** the final
   macro; the subagent authors the real macro+micro from the preview layout.

Two ways to generate intents:

- **Author them yourself** from the cache + patterns (best for small N or when you
  want precise control over balance). Write each to `outputs/temp/batch_intents/<name>.yaml`
  (template: `assets/task_template.yaml`).
- **Use the planning helper** `scripts/batch_compose.py --plan-only`
  (the script's intent-sampling logic — archetypes, `instantiate_subtasks`,
  `reachable_receps`, `choose_recep` — picks concrete subtask skeletons from the
  cache and writes intent YAMLs without running any simulator or authoring any
  instruction). Inspect the emitted intents for balance before dispatching
  subagents; adjust the `ARCHETYPES` dict at the top of the script to add a flavor.

  ```bash
  python .claude/skills/compose-alfred-task/scripts/batch_compose.py --plan-only \
      --rooms Kitchen Bedroom \
      --n 30 \
      --intent-dir outputs/temp/batch_intents \
      --eval-set <eval_set>
  # (--target-scenes '[301,302,8]' instead of --rooms to plan specific scenes)
  ```

Either way, the intent YAML's `macro` field (if present) is only the goal *idea*
for the subagent; the subagent overwrites it with a real, layout-aware macro.
Never deliver an intent's `macro` field as a final instruction.

## Subagent brief (one per intent)

Dispatch the subagents in parallel (one tool block, multiple Agent calls). Each
subagent gets exactly one intent and runs the full authoring pipeline. Hand it
this brief, filled in:

```
Skill: compose-alfred-task (path: .claude/skills/compose-alfred-task/SKILL.md).
Read that SKILL.md, then execute the Stage-1 authoring pipeline for ONE task.

Intent spec (file): outputs/temp/batch_intents/<name>.yaml
Goal idea: <the one-sentence idea from planning>
Out task dir: pgibench/envs/pgi_alfred/data/tasks/<eval_set>/<task_id>
Interpreter: `conda activate stbench` then `python`

Do, in order:
1. Run `task_builder preview --intent <intent.yaml> --out outputs/temp/preview_<name>.json`
   (see SKILL.md for the stdout-banner caveat: take the LAST line of stdout as the
   preview JSON path; do not redirect stdout to a file).
2. Read the preview JSON's `layout_summary`. Finalize the subtasks (concrete
   receptacle instances from the real drop points).
3. **Semantic coherence check (SKILL.md "Core principle — the task must mean
   something")**: before any instruction, confirm the subtask chain achieves a
   real goal and every object↔destination is sensible (slice must precede
   plating/cooking; heat/cool must then deliver the food/drink somewhere; clean
   stows a dirty item at its home; tidy moves an out-of-place item to a MORE
   plausible home — a book belongs on a desk/shelf, not the floor/bed/fridge).
   Reject anti-patterns like cooling a lone cup then leaving it, or slicing a
   vegetable then leaving the slices on the counter. Redesign (change object /
   destination, add the missing follow-on, or drop a dead-end subtask) rather
   than authoring over a broken plan. If the plan can't be made meaningful,
   return status:skipped with the reason.
4. Author the `macro` AND `micro` together from `layout_summary` (core principle
   — see SKILL.md). They express the same *meaningful* goal in two registers:
   `macro` = a human-style instruction modeled on the `complex_instruction` set
   (scene/context lead-in + polite/casual request + optional short rationale,
   with the key objects/destinations folded in as natural qualifiers from the
   real layout — NOT a colloquial prefix with the subtask list appended, NOT a
   flat command); `micro` = the precise step-by-step procedure naming every
   target object + destination. Calibrate macro style by reading a few
   `pgibench/envs/pgi_alfred/data/tasks/complex_instruction/*/task.yaml`. No
   placeholders, no generic templates.
5. Assemble the full cfg (copy scene/object_poses/toggles/init_action/random_spawn
   from the preview JSON; add your subtasks + instructions) and run
   `task_builder write --cfg <full.yaml> --out-dir <out task dir>`.

Retry ladder (SKILL.md): on preview failure, try a different seed/scene/object ONCE
before giving up. Do NOT run Stage-2 trajectory verification (the orchestrator
does that separately).

Return EXACTLY this JSON on your last line:
{"status":"ok","task_id":"<id>","macro":"<your macro>","micro":"<your micro>","subtask_count":<n>}
or
{"status":"skipped","reason":"<one line: e.g. no heatable object inherent after seed change>"}
```

The subagent returns its authored macro+micro so you can sanity-check them in the
report without loading each layout yourself.

## Orchestration — Stage-1 → Stage-2 verify-and-top-up loop (max 3 rounds)

Stage 2 is mandatory and gates delivery. A task is "success" only after **two**
checks pass:
1. **Solvability** — `id_trajectory_generator` reports `feasible: true` (and writes
   `trajectory.json`): the subtask chain can actually be driven to completion.
2. **Clean initial state** — `verify_init_state.py` reports `initial_clean: true`:
   loading the task through `PGIAlfEnv` (exactly the way the evaluator loads it)
   yields `goal_conditions_met == 0/total`, i.e. the initial `task_progress` is 0.
   A non-zero start means the random spawn already satisfied some goal condition
   (an object already at its target receptacle, a toggle already in its target
   state) — the agent would get "free" progress, so the task is unfair.

The two checks answer different questions (can it be solved vs. is the start
fair), so both run on every task; **either** failing triggers a Stage-1 top-up.
Loop at most 3 times, then log unresolved failures and continue — never loop
forever.

```
accepted = []
pending = plan N diverse Priority-A-led intents (see "Intent planning")
for round in 1..3:
    # Stage 1 — author pending tasks (parallel subagents, one tool block)
    written = [subagent(intent) for intent in pending]   # each writes task.yaml
    if not written: break
    # Stage 2 — verify each newly-written task (needs X server)
    for t in written:
        run id_trajectory_generator --task-dir <t.dir> --out-dir outputs/temp/task_verify
        run verify_init_state.py    --task-dir <t.dir> --out-dir outputs/temp/task_verify
        ok = report.feasible and init.initial_clean
        if ok: accepted.append(t)
        else: failures.append({task_id, reason, round})
              # reason = first exec_log ok:false  OR  init_progress>0  OR  init restore error
    infeasible = written - accepted
    if not infeasible: break
    # top-up — plan replacements guided by the failure reasons
    pending = plan(len(infeasible)) replacement intents
        # clean-fail (object not cleaned after all faucets) -> single-sink Kitchen/Bathroom
        # put-fail (put pose) -> swap to a large fixed recep (CounterTop/DiningTable)
        # facility-missing -> swap scene_num / seed
        # init_progress>0 -> swap seed/object/scene (random spawn pre-satisfied a goal)
# after the loop
write outputs/temp/compose_batch_<NN>_report.json (incl. `unresolved` if round==3)
```

Failure-reason → fix mapping (read `report.json` `exec_log` entries with `ok:false`,
and `init_state.json` `init_progress`):
- `op: clean`, `err: object not cleaned after all faucets` → single-sink scene.
- `op: put`, err on small/spawned recep → swap to CounterTop/DiningTable/Shelf.
- `op: heat/cool` fail → usually a cascade from the preceding `put` into the
  microwave/fridge; fix that `put` first, or swap scene.
- object/recep `not found` → facility missing; swap `scene_num`/`seed`.
- `pick` fail (unreachable) → swap `seed` (different spawn point).
- `init_progress > 0` (some goal condition already met at start) → swap
  `seed` (re-roll the random spawn so the target object is no longer coincidentally
  in place), or swap the object/scene if the condition is scene-inherent.

**Report schema:**
```json
{"total": N, "success": len(accepted), "skipped": <authoring skips>,
 "unresolved": [{"task_id": "...", "reason": "...", "rounds": 3}, ...],
 "results": [{"item": "<macro>", "status": "ok", "task_id": "..."} |
             {"status":"skipped","reason":"..."} |
             {"status":"unresolved","reason":"...","rounds":3}]}
```
Print a `success/skipped/unresolved` summary. If `unresolved` is non-empty, flag
it clearly (e.g. "⚠ 2 tasks unresolved after 3 rounds: …") so the user can act.

> Subagents do **not** run Stage 2 — you (the orchestrator) do, after collecting
> each round's authoring results. Keep subagents focused on one task's authoring.


## Macro-micro + semantic audit (run before publishing a task set)

Because each subagent authored its own instructions, sanity-check the delivered
set. Read each task's `instructions` against its `pddl_params.subtasks`:

**Consistency** (name-matching):
- macro/micro mentions an object NOT in any subtask's `object_target`
- macro/micro omits a critical verb implied by subtasks (heat/cool/clean/slice/toggle)
- macro/micro names a different receptacle than the subtasks' `parent_target`
- macro and micro disagree on objects/destinations

**Macro register** (human style — `references/examples.md` "macro / micro style
red lines"; calibrate against `complex_instruction` tasks):
- macro is a colloquial prefix with the subtask list appended (rewrite it as
  scene-set + polite/casual request + optional rationale, with qualifiers folded
  in)
- macro is a flat bare command with no scene-set/rationale (reads robotic, not
  human)
- macro uses a vague category ("a vegetable") or names no discriminating object

**Semantic coherence** (meaningfulness — SKILL.md "Core principle — the task must
mean something"):
- the subtask chain achieves a recognizable real-world goal (a plated meal, a
  tidied room, washed-and-stowed items), not a list of disconnected actions
- no dead-end subtask (slice not followed by plating/cooking; heat/cool not
  followed by serving; clean of an already-clean item returned to its spot)
- every object↔destination is common-sensible for the room (a book on a desk/
  shelf, not on the bed/fridge; food on a plate/table, not in the sink)

Fix a flagged task by re-authoring its macro+micro, or redesigning its subtasks,
to be both consistent *and* meaningful. Drop tasks that cannot be made coherent.

## Sanity workflow after code edits (scene_snapshot / task_builder / generator)

1. Re-probe affected FloorPlans if `scene_snapshot` / spawn logic changed.
2. Re-run a few previously-feasible tasks as regression via a small batch
   (`--target-scenes '[306,307,310]'` in the planning helper, then a 3-subagent
   dispatch).
3. Confirm the report's success count did not drop.
