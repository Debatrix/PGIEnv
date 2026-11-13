# Designing tasks (random init — no trajectory dependency)

Random initialization spawns instruction objects + distractors into a scene, so
any object/action combo legal for a room is generatable. You are NOT limited to
combos that already exist as trajectories (the old splicing constraint is gone).

## Recipe

1. Pick a room + `scene_num` (ranges below). State-changing subtasks need the
   scene's fixed facilities — see SKILL.md "Facility dependencies".
2. Pick 1–4 subtasks. Prefer designs beyond pick/put — see the patterns below.
   For place subtasks use distinct target objects (avoids goal conflicts). Honor
   action→object rules (`object_constraints.md`).
3. Run `task_builder preview`, read `layout_summary`, then write **macro + micro
   together** (same decision, two registers). They carry the same goal but differ
   in *register and granularity*:
   - **`macro`** = a natural, human-style instruction that sounds like something a
     person would actually say — scene-set + request, optionally followed by a
     short rationale. Model it on the existing `complex_instruction` set (see
     "macro style" below). It must NOT enumerate the subtasks; it frames the goal
     as a real want/need and folds in the discriminating objects/destinations as
     natural qualifiers drawn from `layout_summary`.
   - **`micro`** = the explicit step-by-step procedure tied to the layout's real
     receptacle instances ("; "-joined) — the precise, unambiguous version. This
     is where every target object and landing spot is named, in subtask order.
   - Both are authored **by you (the authoring agent)** after reading
     `layout_summary` — never script-spliced from subtasks, never stamped from a
     generic template. In batch mode each task's subagent authors the pair from
     that task's layout (see `batch_tools.md`); no `[PLACEHOLDER]` is ever stamped.

## macro / micro style red lines

**`micro`** is the precise register: it must **name every target object and its
landing spot**, in subtask order; consecutive place subtasks sharing one
receptacle merge into one clause; action verbs (heat/slice/clean/put/turn on/off)
each carry a specific object. Without this, the agent cannot tell targets apart
from the random distractors.

**`macro`** is the *human* register. Mirror the style of the existing
`complex_instruction` task set — read a few
`pgibench/envs/pgi_alfred/data/tasks/complex_instruction/*/task.yaml` to calibrate.
A real macro is a **scene/context lead-in + a polite or casual request, often
followed by a short rationale ("… It's a great way to …", "… That way, …")**.
The two traps to avoid:

1. **"prefix + append"** — a colloquial opener glued to a mechanical enumeration
   of every subtask. That is just the micro wearing a hat.
2. **flat command** — a bare procedure with no scene-set or rationale
   (`"Turn on the floor lamp."`), which reads robotic, not human.

| ✅ human macro (complex_instruction style) | ❌ don't |
|---|---|
| `After a long day of preparing meals, I'm longing to kick back and enjoy dinner. Could you please set the plate with a spoon in it on the kitchen table? That way, everything will be perfectly in place.` | `Set the table for dinner. Put a plate, fork, and knife on the dining table.` (flat command + appended enumeration) |
| `When organizing reading spaces, kindly move the book to the couch. It's perfect for when the next chapter awaits.` | `Tidy up before guests arrive. Put the book and watch on the shelf, and put the pillow on the sofa.` (prefix + appended enumeration) |
| `Planning a spa day at home? Place a cleaned sponge in the bathtub. It's perfect for any relaxing me-time sessions you have in mind.` | `Clean the sponge and put it in the bathtub.` (flat command, no rationale) |
| `After purchasing some fresh produce, rinse off the lettuce thoroughly and then place it on the countertop. Preparing a salad later will be a breeze!` | `I want a salad. Clean the lettuce and put it on the counter top.` (irrelevant prefix + mechanical tail) |
| `While you're tidying up the bathroom, find a spot on the back of the toilet to put two rolls of toilet paper. It's a great place for convenient access when needed.` | `Put two toilet papers on the toilet.` (flat, no scene/rationale) |

The red lines for macro:
- **Open with a scene/context lead-in or a rationale question.** "After a long
  day…", "When organizing reading spaces…", "Planning a spa day at home?",
  "While you're tidying up the bathroom…", "For freshness…", "As you finish up
  with reading…". This grounds the request in a real situation — the hallmark of
  the complex_instruction style.
- **Phrase the request politely or casually, like speech.** "Could you…",
  "Would you mind…", "Kindly move…", "Please place…", or a relaxed declarative.
  Not a bare imperative list.
- **Add a short rationale clause when it fits** — "It's an excellent way to…",
  "That way, …", "It keeps everything in one place…". This is what makes it read
  as a human instruction rather than a command. (Optional but characteristic.)
- **Fold the discriminating objects/destinations in as natural qualifiers, read
  from `layout_summary`** — e.g. "the plate with a spoon in it", "two bars of
  soap", "a cleaned sponge", "the back of the toilet". Specific enough to separate
  targets from distractors; do NOT enumerate every subtask the way micro does.
- **Contrast/qualifier phrasing is idiomatic here** — "Instead of on the sofa,
  put the books on the desk.", "Despite the space on the shelf, put a pan with an
  apple in it in the fridge." Use it when the task has a "not there, but here"
  intent.
- **Never a vague category** (`"a vegetable"`) and never objects/verbs that
  aren't in the task.

## Task-design patterns

Below are the canonical task families. Each example shows `macro` / `micro` you
would fill in the full cfg after preview. Write macro + micro **together** so
subtask logic stays consistent across both registers.

### Pattern 0 — Floor scatter (pick up from the floor) ★ core new semantics

Drop instruction objects onto the **floor** (y<0.3 verified poses) via
`scene_positions_scatter`; the agent must pick them up from the floor before
stowing/processing them. This is semantics `InitialRandomSpawn` cannot produce
(it only places on receptacles). Add to the intent:
`scene_positions_scatter: [{objectType: X, count: N, floor_only: true}]`.

- Kitchen FP8: scatter Tomato + Apple onto the floor, pick them up onto the
  counter.
  - subtasks: `pick_and_place_simple` Tomato → CounterTop;
    `pick_and_place_simple` Apple → CounterTop
  - scene_positions_scatter: `[{Tomato, floor_only}, {Apple, floor_only}]`
  - macro: "A couple of things rolled off the kitchen counter — would you grab the
    tomato and the apple from the floor and set them back up where they belong?"
  - micro: "Pick up the tomato from the floor and put it on the counter top; pick
    up the apple from the floor and put it on the counter top."
- LivingRoom FP210: scatter RemoteControl + Newspaper onto the floor, pick them
  up to their homes.
  - macro: "The living room floor's gotten cluttered. Could you drop the remote
    control on the coffee table and tuck the newspaper onto the shelf? It keeps
    everything within easy reach."
  - micro: "Pick up the remote control from the floor and put it on the coffee
    table; pick up the newspaper from the floor and put it on the shelf."

> **Limit**: scattered objects must already be in `object_counts` (spawned by
> `InitialRandomSpawn`) or scene-inherent. Non-inherent, un-spawned types are
> skipped by `scatter_from_positions`. Scene-appropriateness is inherent to the
> scene_positions dataset (collected per FloorPlan — a bedroom scene has no
> vegetable poses). Only instruction objects are scattered (distractors stay on
> receptacles).

### Pattern 1 — Tidy a room

Stow objects scattered on counters/sofas/chairs back to sensible home positions.
**Multiple `pick_and_place_simple`**, distinct objects, target receptacles being
each object's "home" (Shelf/Drawer/Cabinet/Dresser).

- Bedroom FP305: `pick_and_place_simple` Book → Shelf;
  `pick_and_place_simple` Pillow → Bed; `pick_and_place_simple` Watch → Dresser.
  - macro: "When tidying the bedroom, could you stow the book on the shelf, settle
    the pillow on the bed, and place the watch on the dresser? It's a great way to
    keep everything in its proper spot."
  - micro: "Put the book on the shelf; put the pillow on the bed; put the watch
    on the dresser."
- LivingRoom FP210: `pick_and_place_simple` RemoteControl → CoffeeTable;
  `pick_and_place_simple` Newspaper → Shelf; `pick_and_place_simple` Pillow →
  Sofa.
  - macro: "As you straighten up the living room, please set the remote control on
    the coffee table, tuck the newspaper onto the shelf, and fluff the pillow onto
    the sofa. That way the room feels inviting again."
  - micro: "Put the remote control on the coffee table; put the newspaper on the
    shelf; put the pillow on the sofa."

> **Note**: `InitialRandomSpawn` only places objects into receptacles the scene
> has inherently (counters/tables/sofas/…); the floor is not in the receptacle
> set, so "pick up from the floor" semantics cannot be produced this way — use
> Pattern 0 (`scene_positions_scatter`) for that. The tidy pattern realizes
> "scattered across surfaces" instead.

### Pattern 2 — Cook + plate

Lay the utensils first, then prepare and plate the ingredients, optionally
reheating. Prefer the **Priority-A composite `pick_slice_then_place_in_recep`**
(slice + plate in one subtask, set `sliced: true`) over the old two-step
(`slice_in_place` then `pick_and_place_simple` sliced:true). Use
`pick_and_place_simple` only to lay out utensils/plates (Priority-B completing the
goal), and `pick_heat_then_place_in_recep` to reheat.

- Kitchen FP8: `pick_and_place_simple` Plate → DiningTable;
  `pick_and_place_simple` Fork → DiningTable;
  `pick_slice_then_place_in_recep` Tomato → Plate (sliced: true);
  `slice_in_place` Bread.
  - macro: "Before sitting down to eat, could you lay out a plate and fork on the
    dining table, slice up the tomato onto the plate, and slice the bread? It
    makes for a nicely prepped meal."
  - micro: "Put the plate on the dining table; put the fork on the dining table;
    slice the tomato and put it on the plate; slice the bread."
- Kitchen FP16: `pick_and_place_simple` Bowl → CounterTop;
  `pick_slice_then_place_in_recep` Apple → Bowl (sliced: true);
  `pick_heat_then_place_in_recep` Potato → CounterTop.
  - macro: "I'm putting together a quick bite — set a bowl on the counter top,
    slice the apple into the bowl, and heat the potato. It's all set for serving."
  - micro: "Put the bowl on the counter top; slice the apple and put it in the
    bowl; heat the potato in the microwave and put it on the counter top."

> `pick_slice_then_place_in_recep` is now a first-class composite (Priority A):
> set `sliced: true` on the subtask so `BaseTask.get_targets()` tags the target
> as e.g. `AppleSliced`, which the goal check (in-recep AND sliced AND both)
> relies on. The old two-step (`slice_in_place` + `pick_and_place_simple`
> sliced:true) still works as a fallback, but prefer the single composite. A
> standalone `slice_in_place` (slice with no plating) is Priority-B — only use
> it when slice+place can't apply (e.g. slice-for-the-sake-of-prep inside a chain).

### Pattern 3 — Composite chores (mix)

Chain state-changing subtasks (look_at / clean / heat / cool / slice_place) +
toggle + pick/put.

- Bedroom FP310: `look_at_obj_in_light` Book → DeskLamp;
  `toggle_object_on` FloorLamp; `pick_and_place_simple` Watch → Dresser.
  - macro: "While settling in for an evening read, would you bring the book over
    to the desk lamp, switch on the floor lamp, and set the watch on the dresser?
    It creates just the right atmosphere for unwinding."
  - micro: "Bring the book to the desk lamp to read; turn on the floor lamp; put
    the watch on the dresser."
- Kitchen FP22: `pick_clean_then_place_in_recep` Mug → CounterTop;
  `pick_heat_then_place_in_recep` Potato → Plate; `slice_in_place` Bread.
  - macro: "After a busy morning, I'd love a warm drink and a light bite — kindly
    rinse the mug and leave it on the counter top, heat the potato and plate it,
    and slice the bread. It gets everything ready at once."
  - micro: "Rinse the mug at the sink and put it on the counter top; heat the
    potato in the microwave and put it on the plate; slice the bread."

### Pattern 4 — Single step (degenerate composite)

Degenerate to a 1-subtask composite; useful for simple probes / smoke tests.

- Kitchen FP1: `pick_and_place_simple` Mug → CounterTop.
  - macro: "For my morning coffee, could you move the mug over to the counter top?
    It's the most convenient spot to brew a cup."
  - micro: "Put the mug on the counter top."
- LivingRoom FP210: `toggle_object_on` FloorLamp.
  - macro: "It's getting a little dim in here — would you turn on the floor lamp?
    It'll brighten the room up nicely."
  - micro: "Turn on the floor lamp."

## Semantic coherence — the task must mean something

An executable task is not automatically a *meaningful* one. Before authoring any
instruction, reflect: **what real-world goal do these subtasks accomplish, and is
each object↔destination sensible?** A benchmark task no person would ever ask for
tests nothing. Reject these anti-patterns (redesign, do not polish):

### Anti-patterns (never ship)

| ❌ meaningless plan | why it fails | ✅ meaningful fix |
|---|---|---|
| Cool a single Mug in the fridge, leave it there | no goal — a lone cup chilled then abandoned | Cool a Tomato/Apple, then plate it as a cold snack |
| `slice_in_place` Tomato, slices left on the counter | slice leads nowhere | Slice Tomato → put slices on the Plate → Plate on DiningTable |
| Heat an Apple, leave it in the microwave | heat-then-abandon is not a task | Heat Apple → place on CounterTop/Plate to serve |
| Clean an already-clean Mug, put it back where it was | no state change, no relocation | Clean a dirty Mug → stow on the shelf/counter |
| Tidy: put Book on the Bed, Pillow on the Shelf | destinations are *less* plausible than starts | Book → Desk/Shelf; Pillow → Bed/Sofa (logical homes) |
| Plate → DiningTable + Book → Shelf in one task | two unrelated goals mashed together | One purpose per task (set the table: plate+fork+knife → DiningTable) |

### Meaningful composite patterns (the goal justifies the chain)

Design subtask chains whose end state a person recognizes as a finished goal:

- **Cook a meal** — the canonical kitchen chain. Slice the ingredients (prep),
  heat the one that needs cooking, plate the result, then set the table (plate +
  utensils → DiningTable). Each subtask plays a role; the end state is "food
  ready on the table."
  - subtasks: `slice_in_place` Tomato; `pick_heat_then_place_in_recep` Potato →
    Plate; `pick_and_place_simple` Plate → DiningTable;
    `pick_and_place_simple` Fork → DiningTable; `pick_and_place_simple` Knife →
    DiningTable
  - macro: "I'm putting together a proper lunch — slice the tomato, heat the
    potato and get it onto a plate, then lay out a fork and knife at the dining
    table. It's a lovely way to have everything ready to sit down and eat."
  - micro: "Slice the tomato; heat the potato in the microwave and put it on the
    plate; put the plate on the dining table; put the fork on the dining table;
    put the knife on the dining table."
- **Cold snack** — slice, chill, plate (cool serves the snack, then it is served).
  - subtasks: `pick_slice_then_place_in_recep` Apple → Plate (sliced: true);
    `pick_cool_then_place_in_recep` Apple → Plate; `pick_and_place_simple` Plate → DiningTable
  - macro: "For a light afternoon treat, could you slice the apple, chill it in
    the fridge, and bring it out on a plate to the dining table? It's perfect for
    a refreshing bite."
- **Wash and stow** — clean a genuinely dirty item, then put it at its home.
  - subtasks: `pick_clean_then_place_in_recep` Mug → CounterTop (clean → stow);
    `pick_clean_then_place_in_recep` Plate → Shelf
  - macro: "When doing the kitchen clean-up, would you rinse the mug and set it on
    the counter top, and wash the plate and tuck it onto the shelf? It keeps
    everything in its place and ready for next time."
- **Tidy a room** — move each out-of-place item to its *logical* home (book →
  desk/shelf, pillow → bed/sofa, watch → dresser). The home must be more
  plausible than the start; a book belongs on a desk, not the floor/bed/fridge.
- **Floor tidy** (uses `scene_positions_scatter`) — items fell on the floor; pick
  them up to sensible homes. The floor is the *mess*, never the *destination*.

### Coherence checklist (run before authoring macro/micro)

1. Name the finished end state in one plain sentence. Can't? The subtasks don't
   belong together — split or drop.
2. Each subtask has a role in that goal, in a sensible order (slice before plate,
   clean before stow, heat/cool before serve). Dead-end subtasks get dropped or
   chained in.
3. Every object↔destination passes common sense for the room (see anti-patterns).
4. One coherent purpose per task — don't mash unrelated goals (cooking + tidying
   the bedroom) into one task.

Only once the plan passes this checklist do you author the macro and micro — and
both must describe the *meaningful* goal, not just recite actions.

## Composite design principles

- **Distinct objects per place-subtask**: placing the same object into two
  different receptacles is an unsatisfiable goal conflict (`validate_composite`
  warns).
- **Prefer large fixed receptacles**: CounterTop/DiningTable/Shelf/Dresser/
  SideTable/Cabinet are reliable; spawned small ones (Plate/Bowl) and
  SinkBasin/Fridge are unstable put targets, use with caution as final landing
  spots. See `object_constraints.md`.
- **Treat heat/cool/clean as best-effort**: in multi-sink / multi-facility scenes
  the generator already does faucet-priority fallback (see SKILL.md "Stage 2
  failure diagnosis"), but it can still fail on a missing facility; check
  `layout_summary.facilities` at preview time to confirm.
- **Subtask count**: 1–4 is ideal. Above 4, agent execution time and failure rate
  rise noticeably.

## Picking a scene

- Kitchen (1–30): Microwave, Fridge, SinkBasin, Faucet, CounterTop, DiningTable —
  supports heat / cool / clean + place.
- LivingRoom (201–230): desks / sofas / shelves + lamps — place, toggle, look_at.
- Bedroom (301–330): bed / dresser / nightstand + lamps — place, toggle, look_at.
- Bathroom (401–430): SinkBasin, BathtubBasin, Toilet — clean, place toiletries.

If the generator reports a missing facility ("object/receptacle not found"),
switch `scene_num` within the room.
