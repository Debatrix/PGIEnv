# Task type balance — design for even capability coverage

Without explicit balancing, generated tasks collapse to `pick_and_place_simple`
(the dominant easy default). That undermines the benchmark: many capabilities go
untested. Use this checklist whenever designing or extending a task set.

## Task-type priority (governs which type a subtask should be)

**Priority A — state-change composites (use first):** `pick_heat_then_place_in_recep`,
`pick_cool_then_place_in_recep`, `pick_clean_then_place_in_recep`,
`pick_slice_then_place_in_recep`, `pick_and_place_with_movable_recep`,
`look_at_obj_in_light`. **Priority B — simple/atomic, fallback only:**
`pick_and_place_simple`, `slice_in_place`, `toggle_object_on/off`,
`open/close_receptacle`. Lead each chain with a Priority-A type; use Priority-B
standalone only when Priority-A conditions can't be met (no facility / no eligible
object). Full rules: `references/object_constraints.md` "Task-type priority".

## Coverage matrix (target)

13 task types exist; aim for **≥8** in any task set of 30+, with no single
type above 50%.

| Category | task_type | Where it tests |
|---|---|---|
| Place | `pick_and_place_simple` | nav + pick + put (basic; Priority-B fallback) |
| Place variants | `pick_two_obj_and_place` | repeated pick-put with object tracking |
| Place variants | `pick_and_place_with_movable_recep` | nested put (object → Bowl → CounterTop) |
| State change | `pick_heat_then_place_in_recep` | microwave open/put/close/toggle cycle |
| State change | `pick_cool_then_place_in_recep` | fridge cycle |
| State change | `pick_clean_then_place_in_recep` | sink + faucet multi-step |
| State change | `pick_slice_then_place_in_recep` | slice then place (single composite) |
| State change | `slice_in_place` | one-shot slice (Priority-B) |
| Light/toggle | `look_at_obj_in_light` | pick + toggle lamp |
| Light/toggle | `toggle_object_on` / `toggle_object_off` | one-shot toggle |
| Container | `open_receptacle` / `close_receptacle` | one-shot open/close |

## Room affinity (which type fits which room)

Pick task types whose required objects are inherent and facilities exist:

| Room | Strong fit (Priority-A first) | Avoid |
|---|---|---|
| Kitchen | heat/cool/clean/slice_place + place + place_with_movable_recep | toggle (no lamps in most K) |
| Bathroom | clean (Cloth/SoapBar) + place + open/close (Cabinet) | heat/cool/slice_place (no facilities) |
| LivingRoom | look_at + toggle (FloorLamp) + place + open (Drawer) | heat/cool/clean |
| Bedroom | look_at + place + open/close (Drawer/Dresser) | heat/cool/clean; toggle (DeskLamp unreliable) |

`toggle_object_on` for DeskLamp tends to fail in Bedroom (AI2-THOR view angle
strict) — prefer `FloorLamp` or skip toggle in Bedroom entirely.

## Search tasks (exploration + memory coverage)

Standard task types assume the object is **visible**. To test exploration +
spatial memory, add search tasks — see `references/search_tasks.md`.

## How to enforce balance in batch generation

When planning a batch (`references/batch_tools.md`), cycle task types
round-robin across the planned intents so no single type dominates. To rebalance:

1. Audit the existing set: count `task_type` per room across all task.yaml.
2. If a type is missing or under-represented, add intents that use it next time.
3. Each task's macro+micro must follow the **macro style red lines** in
   `references/examples.md` — they are authored per-task by the authoring
   subagent from the real layout, not templated.

## Macro-micro + semantic audit (run before publishing a task set)

P0 violations to grep for (read each task's `instructions` against its
`pddl_params.subtasks`):
- macro/micro mentions an object NOT in any subtask's `object_target`
- macro/micro omits a critical verb implied by subtasks (heat/cool/clean/slice/toggle)
- macro/micro names a different receptacle than the subtasks' `parent_target`

Also check **semantic coherence** (SKILL.md "Core principle — the task must mean
something"): each subtask chain must achieve a recognizable real-world goal with
sensible object↔destination pairings (no dead-end slices, no heat/cool-then-
abandon, no tidying an item to a less-plausible spot). Instructions are authored
only by the authoring agent; after a batch addition, re-audit each task's
`instructions` against its `pddl_params.subtasks` for both consistency and meaning.
