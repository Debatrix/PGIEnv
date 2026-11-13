---
name: compose-eqa-generator
description: >-
  Compose, generate, and verify executable EQA question generators for the
  PGI-ALFRED benchmark from a natural-language question idea through interactive
  dialogue + simulator verification. Use this skill whenever the user wants to
  add / design / create a new EQA question type (single-choice or multi-choice,
  any temporal variant) into pgibench/envs/pgi_alfred/eqa/generators/ — phrases
  like "generate an EQA generator", "add a question type", "create a question category",
  "make a 'how many apples are visible' question", "design an EQA question about
  X", or any request to turn a natural-language question idea into a runnable
  generator that plugs into the auto-discovery + benchmark_design.yaml pipeline.
  Also trigger when the user mentions ThorTracker, ObjectState, EQA evidence
  dict, or asks why a question generator returns None / fails to register.
---

# Compose EQA Generators from Natural Language

Guide the user, in stages, from a natural-language question idea to a **verified
executable** EQA generator (`generators/<name>.py` + `benchmark_design.yaml`
entry) that `EQAGenerator` auto-discovers and `scripts/verify_eqa_generator.py` can exercise.
The pipeline is interactive: you elicit intent, draft the generator + yaml
entry, write to disk, then verify by running a real AI2-THOR trajectory.

Two contracts constrain what is generatable:

1. **Tracker-data limit**: a generator can only query what `ThorTracker` has
   recorded. If the question needs a field not in `ObjectState` (e.g., agent
   gaze target, audio events), the generator must return `None` for episodes
   without it. Read `references/tracker_api.md` BEFORE designing.
2. **Option-pool limit**: distractors come from the scene's actual objects /
   containers / locDescs — you cannot fabricate. If the pool is too small to
   meet the per-type minimum (3 for single_choice, 2 for multi_choice), the
   generator returns `None`. See `references/generator_contract.md`.

Interpreter for every command below: activate the conda env first (`conda activate stbench`), then use `python`.

## The contract you must honor (read before stage 1)

`EQAGenerator._discover_generators()` scans every `*.py` in `generators/`, finds
functions decorated with `@eqa_generator`, and registers them by `question_id`.
So a valid generator must:

1. Define the `eqa_generator` decorator in-file (each file copy-pastes it —
   there is no shared decorator module; copy from `assets/generator_template.py`).
2. Be decorated with `question_id` (snake_case, unique across all generators),
   `question_type` (`"single_choice"` or `"multi_choice"`), `templates`
   (list of `{placeholder}` format strings), `description`.
3. Have signature `def gen(tracker: ThorTracker, constants: ModuleType,
   timestep: int, rng: random.Random) -> Optional[Dict[str, Any]]`. The caller
   invokes with keyword args — do not rename parameters.
4. Return a `create_question_dict(...)` whose `options` is non-empty and whose
   `correct_answer` labels all exist in `options`. `validate_question_dict`
   rejects otherwise (silently — the question is dropped).
5. Be registered in `benchmark_design.yaml` (matching `id` / `type` /
   `templates` / `option_generation` / `generation_notes`), otherwise it is
   discoverable via `generate_specific_question` but won't be sampled by the
   batch random pool.

Full schema + naming conventions: `references/generator_contract.md`.
Tracker/object helper API: `references/tracker_api.md`.

## Stage 1 — Natural-language intent → interactive dialogue → write to disk

### Shared pipeline

1. **Capture intent**: what is asked (object / agent / count / spatial relation / property /
   visibility) + temporal variant (initial / current / N-steps-ago / random-T /
   history-cumulative) + single-choice or multi-choice.
2. **Cross-check the tracker API**: against `references/tracker_api.md`, confirm the required
   fields/helpers exist; if missing, tell the user to rephrase the question, or
   `return None` when the data is insufficient (do not bypass).
3. **Pick the question_id**: `<concept>_<variant>` (e.g. `obj_loc_initial`).
   First run `grep -r "question_id=" pgibench/envs/pgi_alfred/eqa/generators/`
   **and** `grep -n "id: \"" pgibench/envs/pgi_alfred/eqa/benchmark_design.yaml`
   once to avoid colliding with the existing 30+ ids — an id collision has two
   surfaces (.py decorator + yaml `id:`), check both.
4. **Draft the two artifacts**:
   - `generators/<concept>.py`: copy `assets/generator_template.py` and fill in
     the blanks. Multiple variants of the same topic can share one file (see the
     5-in-1 `object_count.py`); a new topic gets a new file.
   - `benchmark_design.yaml`: add an entry under the corresponding category;
     `generation_notes` must state the naming convention (objectType vs locDesc)
     + the sequential-number filtering rule.
5. **Self-review the 4 red lines** (details at the top of `references/examples.md`):
   - objectType fields (`<obj>`, `<obj_t_list>`) only use `state.object_type`;
     locDesc fields (`<RE_obj>`, `<RE_obj_t>`) only use
     `get_object_loc_desc(state)`, and must pass `has_sequential_number`.
   - For multi_choice, `correct_options` is a subset and `distractors` do not
     overlap with the correct options.
   - Insufficient data (empty pool / too few candidates) always means
     `return None`; **never** degrade the question to force an output.
   - `evidence` contains at least `question_timestep`; every option must be
     traceable back to a tracker field.

### Interactive mode (per generator)

After step 5 **pause** and have the user review the following key fields; only
proceed to Stage 2 after confirmation:

- `question_id` + `description` (one-line semantics)
- Pick 1–2 templates, fill in concrete objects / containers, and **simulate the
  full question_text**
- `option_generation`: where the correct options come from, where the
  distractors come from, the minimum count
- `evidence` structure: list the tracker field paths that prove the correct answer

If the user wants to "see the rough shape first" during the intent stage, first
fill in a minimal runnable version from `assets/generator_template.py`
(pick the first candidate, write a single template sentence), get through
Stage 2, then come back to polish the templates / evidence / filter conditions.

## Stage 2 — Generate questions + verify (needs X server)

```bash
# one-time, separate tmux (an Xvfb on :1 already running in this session is fine):
$PY -m pgibench.envs.pgi_alfred.scripts.start_xvfb 1
```

`scripts/verify_eqa_generator.py` is the verification entry point. It runs a
short live episode (so the tracker accumulates timesteps), then exercises the
selected generators:

```bash
$PY scripts/verify_eqa_generator.py \
    --eval-set smoke_test --selected-index 0 \
    --generators "<your_new_question_id>" \
    --num-questions 3 --x-display 1 \
    --out running/verify_eqa/questions.json

# list every registered generator id:
$PY scripts/verify_eqa_generator.py --list --eval-set smoke_test
```

Read stdout + `running/verify_eqa/questions.json`:

- **Every question was generated** → good; spot-check whether `question_text` /
  `options` / `correct_answer` / `evidence` are sensible (especially whether the
  distractors are truly incorrect, whether locDescs contain sequential numbers,
  and whether the multi-choice correct subset is labeled correctly).
- **`✗ Generator returned None (insufficient data?)`** → the filter is too
  strict or there is a logic bug.
  - Filter too strict (e.g. requires ≥3 candidates but the scene has only 2) →
    relax the threshold or switch `trajectory_file` to a longer / richer one
    (`spatial_*` / `long_horizon_*`).
  - Logic bug (e.g. mistaking `parent_receptacles` for
    `receptacle_object_ids`) → fix the code and re-run.
- **Exceptions being swallowed** → `_discover_generators` does
  `except Exception: pass`, so the generator is not registered. Temporarily
  change the `except Exception` at lines 84/163/273 of `eqa_generator.py` to
  `except Exception as e: print(e); raise` to locate the problem, or add
  `try/except + print(traceback.format_exc())` at the top of your own
  generator. **Restore the original state** after fixing.

If no X server / GPU is available, Stage 2 cannot run — say so plainly and stop
after Stage 1. Never claim a generator is verified without a real run.

## Stage 3 — (optional) Distribution check

Getting one trajectory to run only confirms the generator can produce output;
to see the distribution and difficulty of questions across multiple scenes /
trajectories, re-run Stage 2 with several different `trajectory_file`s
(recommended: at least one each of `base_*` / `spatial_*` / `long_horizon_*`,
covering different step counts and rooms). This step is **not** must-pass,
just a quality guardrail.

## Question types & option conventions

| type | correct_answer format | minimum distractors | when to use |
|---|---|---|---|
| `single_choice` | single label (e.g. `"B"`) | 3 | asking position / direction / count / state with a unique correct answer |
| `multi_choice` | comma-separated (e.g. `"A,C,D"`) | 2 (incorrect) | "Select all that apply" — subset questions |

**objectType fields** (`<obj>`, `<obj_t_list>`, `<obj_f_list>`): use
`state.object_type` directly, no sequential-number filtering (type names
contain no digits).

**locDesc fields** (`<RE_obj>`, `<RE_obj_t>`, `<RE_obj_f>`): must use
`get_object_loc_desc(state)` and **pass `has_sequential_number`** — a locDesc
like `"Drawer 1 2"` contains sequential numbers, which would make multiple
options ambiguous under the same name.

## Tracker & helper API (cheat-sheet)

- `tracker.all_object_ids`, `tracker.historically_seen_objects`,
  `tracker.observations[t]`, `tracker.current_timestep`
- `tracker.get_object_state_at_step(obj_id, t)` → `ObjectState` or None
- `tracker.is_unique_object_type(obj_id)`, `tracker.was_object_ever_held(obj_id)`,
  `tracker.get_hold_ordinal(obj_id)` — used when referencing an object that was picked up
- `utils.get_object_loc_desc(state)`, `has_sequential_number(text)`,
  `get_object_history(tracker, id, t)`, `get_objects_in_container(tracker, id, t)`,
  `get_visible_objects_at_step(tracker, t)`, `calculate_relative_direction(a, b)`,
  `get_object_property_state(state)`, `generate_count_distractors(n, k, rng)` …
  full list (17 helpers + 21 ObjectState fields) in `references/tracker_api.md`.

## Naming the file

Topic grouping (multiple variants of the same topic share one file):

| file | topic |
|---|---|
| `object_location.py` | single-object position + temporal variants |
| `object_history.py` | cumulative history (multi) |
| `object_count.py` | counting inside containers / in the scene |
| `object_attributes.py` | property state + existence |
| `object_relative_position.py` | relative directions between objects |
| `object_types_in_container.py` | object-type sets inside a container |
| `agent_position.py` | agent's own position / directions relative to the agent |

New topics (e.g. room transition, agent gaze) get new files; temporal variants
of existing topics are appended to the end of the corresponding file
(requirement: keep reusing that file's `_get_*` private helpers).

## Coordinate system (do not get this wrong)

EQA direction questions use the AI2-THOR coordinate system: the vertical Y axis
is **ignored**; in the XZ plane +Z=North, +X=East, -Z=South, -X=West. To
compute the relative direction between two objects use
`utils.calculate_relative_direction`, which already converts
`math.atan2(dz, dx)` into a North=0° clockwise system. **Do not** use
`math.atan2(dy, dx)` — Y is vertical and you will get completely wrong
directions. See `pgibench/envs/pgi_alfred/CLAUDE.md` for details.
