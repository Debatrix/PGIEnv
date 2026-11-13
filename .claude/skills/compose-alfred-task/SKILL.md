---
name: compose-alfred-task
description: >-
  Compose, generate, and verify executable PGI-ALFRED tasks from a natural-language
  idea through random scene initialization + id-driven trajectory generation. Use this
  skill whenever the user wants to create / design / add a task (single-step or
  composite, short or long-horizon) to the PGI-ALFRED benchmark — phrases like
  "generate a task", "composite task", "multi-step task", "create a test task",
  "make the agent clean then heat an apple", "generate 30 kitchen tasks", or any
  request to turn a natural-language task idea into a runnable ALFRED task.
---

# Compose ALFRED Tasks from Natural Language

Guide the user, in stages, from a natural-language task idea to a **verified
executable** task (`task.yaml` + `trajectory.json`) that PGIAlfEnv can load.

The pipeline is **random scene initialization + id-driven trajectory generation**
(no trajectory splicing). An agent — you in interactive mode, a subagent per task
in batch mode — drives the whole thing: write an intent, run `task_builder
preview` to snapshot a random layout, **read the layout, design the final
subtasks, and author the `macro` and `micro` instructions in the same decision**,
then `task_builder write`.

## Core principle — instructions are authored, never stubbed

A meaningful task is defined by its **`macro` (colloquial goal sentence) and
`micro` (step-by-step procedure)**. These are authored **by the agent** from the
real preview layout, and both carry the complete goal (every target object + its
destination). They are the *point* of the task, not a cosmetic finish.

This principle is non-negotiable:

- **Author macro + micro yourself, at design time, from `layout_summary`.** Read
  the layout, decide what the task *is*, and write both instructions in the same
  moment as you finalize the subtasks. Never stamp a generic/template sentence
  and plan to "fill it in later."
- **No placeholders, no stubs, no generic macros, no deferred backfill.** A task
  with `macro: "Tidy the kitchen"` and micro = macro is *not* a deliverable. A
  generic, layout-blind sentence stamped before the scene exists leads to
  meaningless, possibly-unrealizable tasks. Do it once, properly, from the layout.
- **macro and micro must be consistent with the subtasks** — same objects, same
  destinations, same verbs (heat/cool/clean/slice/toggle). Both must name every
  target object and its landing receptacle so the agent can tell targets apart
  from the random distractors. Style red lines: `references/examples.md`.
- **One task, one authoring decision.** If you change the subtasks later
  (retry on a failing seed, swap a receptacle), re-author macro + micro to match.

The skill only ships a task once its macro and micro are both authored against a
real layout. In batch mode this is delegated to one subagent per task (so each
has a focused context and they run in parallel), not deferred to a review stage.

## Core principle — the task must mean something

An executable task is not automatically a *meaningful* one. Before writing any
instruction, **reflect on why these subtasks belong together and what real-world
goal they accomplish**. A benchmark task that an agent can solve but that no
person would ever ask for tests nothing useful. Ask yourself, for every task:

1. **What concrete thing does finishing these subtasks achieve?** There must be a
   coherent end state a person would recognize as a completed goal — a meal
   plated and on the table; a room straightened; dirty items washed and put away.
   If you cannot name that end state in one plain sentence, the subtasks do not
   belong together.
2. **Why is each subtask in the chain — and why in this order?** Slicing is prep
   for cooking or plating; heating makes food ready to eat; cleaning must come
   *before* stowing a dirty item. A subtask with no role in the goal (slice a
   vegetable then leave it where it was) is filler — drop it, or chain it into
   the goal (slice → plate → table).
3. **Are the chosen objects and destinations sensible for this goal and room?**
   Objects and landing spots must match real-world logic. This is where most
   meaningless tasks fail. Judge each object↔receptacle pairing against common
   sense.

### Object↔destination common-sense rules (reject violations)

These are the failure patterns to reject at design time (full do/don't table +
guided patterns: `references/examples.md` "Semantic coherence"):

- **Cooking**: to make a meal you slice ingredients, heat what needs heating,
  plate the result, then set the table (plate on the dining table + utensils
  beside it). Cool is for drinks/dessert. A single cup cooled alone in the
  fridge and left there is pointless; cooling is meaningful when it serves a
  snack or drink that then gets placed.
- **Slicing**: slicing is always *for* something — to plate, cook, or serve. A
  standalone `slice_in_place` whose slices then sit on the counter is a dead
  end. Always follow slice with a place onto a plate/bowl/table (or fold it into
  a cook step).
- **Cleaning**: clean a *dirty* item (mug, plate, utensil) and stow it where it
  belongs — mugs to the shelf/counter, plates stacked, not dumped in the sink.
  Cleaning an already-clean object to put it back where it was is vacuous.
- **Heating/Cooling**: heat/cool food or a drink, then *deliver* it somewhere
  (plate, bowl, table). Heat-then-leave-in-microwave, or cool-then-leave-in-
  fridge, is not a task.
- **Tidying**: move out-of-place items to their logical home. A book on a desk
  or shelf is sensible; a book on the floor, bed, or in the fridge is the *mess*
  to fix, not the goal. The home must be more plausible than the start.
- **Place subtasks need distinct objects per landing spot** (also a goal-conflict
  rule), but beyond that the *grouping* must read as one purpose: "set the table"
  = plate + fork + knife → dining table, not plate→table + book→shelf mashed
  together.

If a subtask or pairing fails this check, **redesign it before authoring
instructions** — change the object, its destination, or drop the subtask. Never
author a polished macro/micro over a semantically broken plan. Both registers
(macro and micro) must describe the *meaningful* goal, not just list actions.

## Three generatability limits

1. **Inherent-object limit**: `InitialRandomSpawn` can only add instances of
   object types already INHERENT to the chosen FloorPlan. `task_builder` verifies
   this up front and, on failure, lists the pickupable types available in that
   scene — pick from that list. (`REPRESENTATIVE_OBJECTS` per room is only a hint;
   each FloorPlan's inherent set differs.)
2. **Put-robustness limit**: the id-driven `put` is unreliable for some
   receptacles — spawned small ones (Plate/Bowl), SinkBasin, Fridge, and `slice`.
   Prefer place targets that are fixed large receptacles
   (CounterTop/DiningTable/Shelf/Dresser/SideTable/Cabinet/Toilet) and heat
   (Microwave). Treat clean/cool/slice as best-effort: they may come up
   infeasible on a given scene.
3. **Floor-scatter via scene_positions**: `InitialRandomSpawn` only places objects
   ON receptacles — it cannot scatter them on the floor. To get "pick up from the
   floor" semantics, the intent carries `scene_positions_scatter`: objects are
   re-posed to **verified curated poses** read from
   `data/scene_positions/FloorPlan<N>.json` (every pose ever seen in an existing
   ALFRED task for that FloorPlan, incl. floor-level y<0.3 poses). This is stable
   (no physics simulation) and scene-appropriate by construction (a Bathroom
   dataset offers no vegetable poses). Only objects already in the scene (spawned
   via `object_counts` or inherent) can be re-posed — non-existent types are
   skipped.

Interpreter for every command below: activate the conda env first (`conda activate stbench`), then use `python`.

## The contract you must honor (read before stage 1)

PGIAlfEnv loads a task via `utils.load_task` (`task.yaml` + optional
`trajectory.json`), assembled into the traj dict that `CompositeTask` consumes.
So a valid task must:

1. Pass static validation (`validate_composite`) — `task_builder preview` runs
   this before spawning.
2. Have a realizable scene: `random_initialize_scene` must actually spawn every
   instruction object (`task_builder preview` verifies this and errors otherwise).
3. Carry `instructions.{macro,micro}` (schema v2) **and** `pddl_params.subtasks`
   populated — the agent fills both from the preview layout, in one decision.
   Without `subtasks`, `CompositeTask` reports fake success.
4. Be feasible: the id-driven generator can satisfy every subtask in AI2-THOR.
5. Start clean: loaded through `PGIAlfEnv` (the evaluator's exact load path), the
   initial `goal_conditions_met` must be `0/total` — i.e. the initial
   `task_progress` is 0. A non-zero start means the random spawn already satisfied
   some goal condition (an object already at its target receptacle, a toggle
   already in its target state), handing the agent free progress. Stage 2's
   `verify_init_state.py` checks this; a non-zero start re-enters the top-up loop.

Full details: `references/task_contract.md`. Legality tables:
`references/object_constraints.md`.

## Script discipline — bundled scripts only, never author or edit one

While running this skill you operate under two hard limits on code:

1. **Do not author scripts.** Do not create, copy, or hand-roll any `.py` file,
   and do not write a `python -c` one-liner to load a task or poke at scene
   state. Use only the tools that already exist — the bundled scripts below and
   the project's own commands (`task_builder`, `id_trajectory_generator`,
   `pgibench.main`).
2. **Do not edit bundled scripts.** Do not modify anything under this skill's
   `scripts/` directory.

The bundled scripts are the single source of truth for the mechanical parts of
this pipeline:

- `scripts/probe_floorplans.py` — one-time FloorPlan cache (inherent objects,
  facilities) for planning.
- `scripts/batch_compose.py` — intent sampling for batch generation
  (`--plan-only`).
- `scripts/verify_init_state.py` — Stage 2 initial-state check: loads a task
  through `PGIAlfEnv` and asserts the initial `task_progress` is 0.

Whenever you need to "load a task and check something" — reach for
`verify_init_state.py`, not a throwaway script. The reason is not stylistic:
off-script checks silently drift from how the evaluator actually loads and
restores a task (two-stage spawn→pin restore, `CompositeTask` goal bookkeeping),
so a hand-rolled check can report "clean" while the real evaluation sees a
different scene state. The bundled script runs the real load path, so its result
is the result the benchmark will see.

## Stage 1 — Intent → preview → agent design → task.yaml

The same authoring pipeline drives both modes. They differ only in pacing and in
how context is kept focused (see "Batch mode" below).

### Authoring pipeline (per task)

1. **Capture the intent**: room / scene_num / objects involved / one-sentence
   goal idea. Pick a task-design pattern (`references/examples.md`) that fits.
2. **Write the intent YAML** (template `assets/task_template.yaml`). Fill the
   `subtasks` field with the **skeleton needed for object counts** (task_type +
   object; parent may be a preferred receptacle or left for step 4 to fix).
3. **Run preview** (spawn + validate, outputs a layout JSON):
   ```bash
   $PY -m pgibench.envs.pgi_alfred.scripts.task_builder preview \
       --intent <intent.yaml> \
       --out outputs/temp/preview_<name>.json   # explicit path; last stdout line is the path
   ```
   Pass `--out` explicitly under `outputs/temp/` (per AGENTS.md) — the tool's
   default is `running/`, which is not the skill's temp area. preview writes the
   JSON **to a file** via `json.dump`, so the file is pure JSON and is NOT
   polluted by the AI2-THOR Unity startup banner. stdout mixes in the banner;
   the output path is on the **last line** — take the last line for the path, do
   NOT `cat` the whole stdout as JSON, and do NOT use
   `> outputs/temp/preview_<name>.json` shell redirection (it writes the banner
   into the file and breaks `json.load`).
   - Object not inherent / missing after spawn → it raises: rerun with a different
     `seed` / `scene_num` / object.
4. **Author the task yourself from `layout_summary`.** Read the preview JSON's
   `layout_summary`, finalize the subtasks (concrete receptacle instances), and
   **write `macro` and `micro` together** based on the layout's real drop points.
   "Reasonable" means the layout can support the task, **not** that the objects are
   tidy (for a "tidy the room" task, objects being scattered is exactly right).
   - **Semantic coherence check (run first — "Core principle — the task must mean
     something")**: before finalizing subtasks, confirm the chain accomplishes a
     real goal and every object↔destination is sensible (slice must be followed
     by plating/cooking; heat/cool must deliver the food/drink somewhere; clean
     stows a dirty item at its home; tidy moves an out-of-place item to a more
     plausible home). Redesign — don't just author around — any subtask or pairing
     that is pointless.
   - **Task-design priority**: do not default to pick/put. Prefer the patterns in
     `references/examples.md` — tidy a room (multiple pick_and_place to home
     positions), cook + plate (lay utensils + several slices + plate), composite
     chores (look_at/toggle/clean/heat chained), floor scatter
     (`scene_positions_scatter` drops instruction objects on the floor; the agent
     picks them up from the floor), single-step degenerate. Pick the one that best
     matches the user's intent.
   - **macro + micro co-authoring (core principle)**: `macro` and `micro` are
     produced in the **same decision** so they express the same goal, but in two
     different registers:
     - `macro` = a human-style instruction modeled on the existing
       `complex_instruction` set — **scene/context lead-in + a polite or casual
       request, often followed by a short rationale** (e.g. "After a long day…,
       could you …? That way, …"). Fold in the discriminating objects/destinations
       as natural qualifiers read from `layout_summary`. It must NOT enumerate the
       subtasks (the "colloquial prefix + appended procedure" anti-pattern is just
       the micro wearing a hat), nor be a flat bare command.
     - `micro` = the precise step-by-step procedure ("; "-joined), tied to the
       layout's real receptacle instances — this names every target object and
       landing spot in subtask order.
     - Both must carry the *meaningful* goal (see "Core principle — the task must
       mean something"); macro expresses it as a human request, micro spells out
       the steps. Style red lines + do/don't table: top of `references/examples.md`.
       Calibrate by reading a few
       `pgibench/envs/pgi_alfred/data/tasks/complex_instruction/*/task.yaml`.
   - **Floor-scatter design (`scene_positions_scatter`)**: if the task needs
     "pick up from the floor / a cluttered spot" semantics (e.g. "put the
     ingredients on the floor into the fridge", "tidy the kitchen"), add a
     `scene_positions_scatter` field to the intent listing the objects to scatter.
     `floor_only: true` restricts to floor-level poses (y<0.3). Only instruction
     objects are scattered (distractors stay on receptacles, keeping targets
     unambiguous). Scattered objects must already be in `object_counts` or
     scene-inherent (non-inherent, un-spawned types are skipped).
   - **Distractor density and repeats**: default `n_distractors: 6`, raise to 8–10
     for complex scenes. Distractors **may repeat** (`sample_distractors` samples
     with replacement by default) — multiple instances of the same type
     (e.g. 3 Mugs) raise the agent's discrimination difficulty among look-alikes.
     Pool capacity is ~12–20 (per room, see `REPRESENTATIVE_OBJECTS`).
5. **Assemble the full cfg** (reuse the preview's
   `object_poses`/`random_spawn`/`scene` + your `subtasks`/`instructions`,
   template `assets/full_cfg_template.yaml`), name
   `task_id = <Room><scene_num>-<seed>-<slug>`, and write it:
   ```bash
   $PY -m pgibench.envs.pgi_alfred.scripts.task_builder write \
       --cfg <full.yaml> --out-dir pgibench/envs/pgi_alfred/data/tasks/<eval_set>/<task_id>
   ```

### Retry ladder (agent decision)

- Object drop point makes the task meaningless / stuck → change `seed` (preview
  has a built-in max_retries; you can add one outer seed change).
- Facility missing (want heat but no microwave) → change `scene_num`.
- Object type unsuitable → change the object.
- Exhausted → **interactive**: report the layout summary to the user to decide /
  **batch**: skip and record the reason.

### Interactive mode

After step 5, **pause** for each task, show the user the `task.yaml` key fields
(`instructions`, `subtasks`, a `scene.random_spawn` summary) for review, then run
**Stage 2** (verify feasibility). Same loop as batch mode but one task at a time:
author → verify → if `feasible: false`, re-author with a fix (swap seed/scene/
object/recep guided by the failing `exec_log` entry) → verify again, up to 3
rounds. If still infeasible after 3 rounds, report the failure to the user and
ask how to proceed (do not silently ship an unverified task).

### Batch mode

When the user invokes this skill with a one-liner like "generate 30 cooking
tasks", you orchestrate a **subagent-per-task** flow with a **Stage-1 → Stage-2
verify-and-top-up loop**. Stage 2 is mandatory and gates delivery: a task is
"success" only after **both** `id_trajectory_generator` reports `feasible: true`
(and writes `trajectory.json`) **and** `verify_init_state.py` reports
`initial_clean: true`. Either failing triggers a Stage-1 top-up, looping at most
3 times.

The orchestrator (you) keeps the big picture and runs the loop; each authoring
subagent holds one task's full context and authors its macro+micro from that
task's layout (it does NOT run Stage 2 — you do).

**The loop (max 3 rounds):**
```
accepted = []
pending_intents = plan N diverse intents (see step 1)
for round in 1..3:
    # ---- Stage 1: author pending tasks (parallel subagents) ----
    newly_written = dispatch one subagent per pending_intent
        each subagent: preview → read layout → finalize subtasks
            (Priority-A composites first) → author macro+micro
            → write task.yaml → return {task_id, macro, micro} | {skipped, reason}
    if not newly_written: break       # nothing more to author

    # ---- Stage 2: verify each newly-written task (you, needs X server) ----
    for task in newly_written:
        run id_trajectory_generator --task-dir <task> --out-dir outputs/temp/task_verify
        run verify_init_state.py    --task-dir <task> --out-dir outputs/temp/task_verify
        if report.feasible and init.initial_clean: accepted.append(task)  # DONE
        else: record {task_id, reason, round}
              # reason = first exec_log ok:false  OR  init_progress>0  OR  init restore error

    infeasible = newly_written - accepted
    if not infeasible: break          # all of this round's tasks verified — done

    # ---- top-up: plan replacements for the infeasible, guided by reasons ----
    pending_intents = plan(len(infeasible)) replacement intents
        (clean-fail → single-sink scene; put-fail → swap recep;
         facility-missing → swap scene/seed; guided by the recorded reasons)

# after the loop (round hit 3 OR all accepted)
if infeasible remains after round 3:
    record them in the report under "unresolved" (task_id + reason + rounds=3)
    PROCEED to Stage 3 anyway — do not loop forever
```

**Guarantees:**
- Stage 2 **always** runs and **gates** delivery — no task ships unverified.
- The top-up keeps the target count N when possible (replacements fill the gap).
- Hard cap at 3 rounds; unresolved failures are logged and the run continues to
  Stage 3 + a clear problem report, so the task always terminates.

**Step 1 (planning, once at the start):** Plan N diverse intents — different
rooms, scenes, objects, and Priority-A-led patterns — balanced across task types
(`references/task_type_balance.md`). Use `outputs/temp/floorplan_cache.json` to
pick objects inherent to each FloorPlan (probe once if missing —
`references/batch_tools.md`). Each intent carries a goal *idea* (NOT a final
macro); the subagent authors the real macro+micro from the layout. Lead every
intent with a Priority-A composite (`object_constraints.md` "Task-type priority").

**Why subagents:** each task's macro+micro stays a genuine design decision made
with the real layout in view; each authoring context stays focused (you never
load dozens of layout JSONs); per-task authoring runs in parallel. A
generator-style helper exists for the planning side only (intent sampling from
the floorplan cache) — it never writes instructions.

Full orchestration detail, the subagent brief, and the verify/top-up script:
`references/batch_tools.md`.

## Stage 2 — Verify solvability + clean initial state (needs X server)

Stage 2 is **mandatory and gates delivery**. A task is not "success" until two
independent checks pass — they answer different questions, so both run:

1. **Solvability** — can the subtask chain actually be driven to completion?
   `id_trajectory_generator` runs each subtask via id primitives and reads the
   final goal state; `feasible: true` means the task is executable (and it writes
   `trajectory.json`).
2. **Clean initial state** — is the start fair? `verify_init_state.py` loads the
   task through `PGIAlfEnv` (the evaluator's exact load path) and asserts the
   initial `task_progress` is 0 (`goal_conditions_met == 0/total`). A non-zero
   start means the random spawn already satisfied some goal condition (an object
   already at its target receptacle, a toggle already in its target state), so
   the agent gets free progress — the task is unfair and must be re-rolled.

`task_progress` is only computed inside `step()`; `reset()` restores the scene
and sets the task but computes nothing, so `verify_init_state.py` reads
`goal_conditions_met()` right after `reset()`, with zero agent steps taken. This
is also why a hand-rolled check is risky: it must reproduce PGIAlfEnv's
two-stage spawn→pin restore exactly, or it inspects a different scene state than
the benchmark will.

In batch mode both checks run **automatically inside the Stage-1↔Stage-2 loop**
(see "Batch mode" above); in interactive mode you run them per task right after
the user confirms the `task.yaml`.

```bash
# one-time, separate tmux (an Xvfb on :1 already running in this session is fine):
$PY -m pgibench.envs.pgi_alfred.scripts.start_xvfb 1

# 1) solvability — drive every subtask, read final goal state
$PY -m pgibench.envs.pgi_alfred.scripts.id_trajectory_generator \
    --task-dir pgibench/envs/pgi_alfred/data/tasks/<eval_set>/<task_id> \
    --out-dir outputs/temp/task_verify

# 2) clean initial state — load via PGIAlfEnv, assert init task_progress == 0
$PY .claude/skills/compose-alfred-task/scripts/verify_init_state.py \
    --task-dir pgibench/envs/pgi_alfred/data/tasks/<eval_set>/<task_id> \
    --out-dir outputs/temp/task_verify
```

Read `outputs/temp/task_verify/<name>/report.json` (solvability):
- `feasible: true` → the task is executable; `trajectory.json` was written.
- `feasible: false` → some subtask could not be satisfied. Look at `exec_log`
  entries with `ok: false` first — they pin down the failing subtask:
  - `op: clean`, `err: object not cleaned after all faucets` → in a multi-sink
    scene none of the faucets marked the object's SinkBasin as cleaned. Switch
    `scene_num` to a single-sink Kitchen or Bathroom; check
    `layout_summary.facilities.SinkBasin/Faucet` at preview time.
  - `op: heat`, `err: object not heated after microwave cycle` → the microwave
    loop did not update `heated_objects` (rare; usually a cascade from a failed
    `op: put` into the microwave just before). Check whether the earlier `op: put`
    in the same subtask already errored.
  - `op: cool`, `err: object not cooled after fridge cycle` → same idea, check the
    fridge.
  - Other common cases: facility missing (the `facilities` field does not match),
    object spawned at an unreachable point, pick/put pose failure. Change
    `scene_num` / `seed` / objects and rerun.

Read `outputs/temp/task_verify/<name>/init_state.json` (clean start):
- `initial_clean: true` (`init_progress: 0.0`, `goal_conditions_met: "0/N"`) →
  fair start. (The script also exits 0.)
- `initial_clean: false` (`init_progress > 0`) → the random spawn pre-satisfied a
  goal condition. Re-roll: change `seed` (different spawn layout so the target
  object is no longer coincidentally in place), or swap the object/scene if the
  condition is scene-inherent. The script exits non-zero, so in batch mode this
  feeds the top-up loop just like a `feasible: false`.

A task ships only when **both** `feasible: true` **and** `initial_clean: true`.

If no X server / GPU is available, Stage 2 cannot run — say so plainly and stop
after Stage 1. Never claim a task is verified without a feasible, clean run.

## Stage 3 — Smoke test (switch instruction by instruction_mode)

`PGIAlfEnv._load_dataset` scans `data/tasks/<eval_set>/*/task.yaml` directly, so
the task written in Stage 1 is already discoverable — no index/splits
registration. Schema v2 carries both `instructions.macro` and
`instructions.micro`; pick which one the agent hears via `instruction_mode`
(default `micro`):
```bash
$PY -m pgibench.main env=pgi-alf eval_sets='[<eval_set>]' selected_indexes='[0]' \
    exp_name=smoke instruction_mode=micro   # or macro
```

## Task types

**ALFRED-level** (multi-step; place a target object into a receptacle, optionally
after a state change): `pick_and_place_simple`, `pick_two_obj_and_place`,
`look_at_obj_in_light`, `pick_clean_then_place_in_recep`,
`pick_heat_then_place_in_recep`, `pick_cool_then_place_in_recep`,
`pick_and_place_with_movable_recep`.

**Atomic-level** (one primitive; mix freely with ALFRED-level in `subtasks`):
`toggle_object_on`, `toggle_object_off`, `open_receptacle`, `close_receptacle`,
`slice_in_place`.

Required params, valid rooms, and action→object rules:
`references/object_constraints.md`.
Scene ranges: Kitchen 1–30, LivingRoom 201–230, Bedroom 301–330, Bathroom 401–430.

## Facility dependencies (random init spawns objects, NOT fixed furniture)

Random init spawns pickupable instruction objects + distractors. State-changing
and receptacle subtasks depend on scene-fixed facilities already present in the
FloorPlan:

| subtask | required facility | valid rooms |
|---|---|---|
| pick_heat_then_place_in_recep | Microwave | Kitchen |
| pick_cool_then_place_in_recep | Fridge | Kitchen |
| pick_clean_then_place_in_recep | SinkBasin + Faucet | Kitchen, Bathroom |
| look_at_obj_in_light / toggle_object_on(off) lamp | DeskLamp / FloorLamp | LivingRoom, Bedroom (typical) |
| open/close_receptacle | an openable receptacle of that type | any |

Kitchen FloorPlans have microwave/fridge/sink/faucet. If a facility is missing
the generator reports "object/receptacle not found" — pick another `scene_num`.

## Goal conflicts (composite)

All subtasks must hold simultaneously in the FINAL state. So never target the
same object with two place-subtasks using different receptacles — that is
unsatisfiable. Use distinct objects per place-subtask.

## Capability coverage — design for balance

Without explicit balancing, generated tasks collapse to `pick_and_place_simple`
(73% of an unbalanced set). When designing or extending a task set, audit the
distribution across the 13 task types and aim for ≥8 covered with no single type
above 50%. Full checklist + room affinity table:
**`references/task_type_balance.md`**.

## Search tasks — hidden-object exploration + memory

Standard subtasks assume the instruction object is visible. To test exploration
and spatial memory, hide objects inside closed containers (Cabinet/Drawer/Safe)
via `container_constraints` — the agent must OPEN the receptacle to find them.
Two flavors: `search_and_stow` (exploration) and `search_use_later`
(long-horizon memory). Design guide + verification flow:
**`references/search_tasks.md`**.

## References

- `references/task_contract.md` — full task schema, validation rules, restore logic, scene_positions scatter
- `references/object_constraints.md` — legal (object, recep, task_type) triples + scene-appropriateness
- `references/examples.md` — task-design patterns (incl. floor scatter) + semantic-coherence anti-patterns/patterns + macro/micro style red lines
- `references/task_type_balance.md` — capability coverage matrix + balance checklist
- `references/search_tasks.md` — hidden-object task design (A/B flavors)
- `references/batch_tools.md` — floorplan probe cache, intent planning, subagent-per-task batch flow
- `data/scene_positions/FloorPlan<N>.json` — curated verified poses per FloorPlan (powers floor scatter)
- `assets/task_template.yaml` — intent YAML skeleton
- `assets/full_cfg_template.yaml` — final task.yaml skeleton
- `scripts/probe_floorplans.py` — one-time FloorPlan cache (inherent objects, facilities)
- `scripts/batch_compose.py` — intent sampling for batch generation (`--plan-only`)
- `scripts/verify_init_state.py` — Stage 2 initial-state check (init `task_progress == 0`)
