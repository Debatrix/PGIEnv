# Generator Contract — what EQAGenerator requires to load a generator

## Discovery chain

`EQAGenerator.__init__` → `_discover_generators()` scans `generators/*.py` →
`importlib.import_module` per file → `inspect.getmembers(module, isfunction)`
→ keeps functions with `is_eqa_generator=True` attribute → registers under
`func.question_id`. The generator is then callable via:

- `generate_specific_question(question_id, timestep, seed, shuffle_opts)` —
  deterministic, one question of that id.
- `generate_random_question(...)` — picks a random registered id.
- `generate_batch_questions(num, specific=True, ...)` — one per registered id.

Any `Exception` inside a generator call is silently swallowed
(`except Exception: continue`), so a buggy generator returns `None` from the
outside. To see the real traceback, temporarily unswallow in
`eqa_generator.py` lines 161–163 / 273.

## In-file decorator (no shared module)

Each generator file defines its own `eqa_generator` decorator. This is by
design — keeps each file standalone and avoids import cycles. Copy it verbatim
from `assets/generator_template.py` or any existing generator file. The
decorator must attach these attributes (consumed by `_discover_generators`):

```python
func.is_eqa_generator = True   # the flag getmembers filters on
func.question_id      = "..."  # unique snake_case id
func.question_type    = "single_choice" | "multi_choice"
func.templates        = [...]  # list of format strings
func.held_templates   = [...]  # optional: for hold-ordinal references
func.description      = "..."  # one-line, mirrored in yaml
```

Forgetting any of these → silent skip (function won't register).

## Function signature (fixed)

```python
def generate(tracker: ThorTracker,
             constants: ModuleType,
             timestep: int,
             rng: random.Random) -> Optional[Dict[str, Any]]:
```

Caller uses keyword args (`generator(tracker=..., constants=..., timestep=...,
rng=...)`). Do **not** rename params. `timestep` is the **current** timestep
(passed separately from `tracker.current_timestep` for reproducibility — they
are usually equal). `rng` is per-question; use it for all randomness so seeds
reproduce.

## Return dict (via `create_question_dict`)

```python
{
    "id": "",                          # filled by EQAGenerator post-hoc
    "question_text": "Where was the apple at the start?",
    "options": {"A": "CounterTop", "B": "Shelf", "C": "Drawer", "D": "SinkBasin"},
    "correct_answer": "B",             # "B" for single, "A,C,D" for multi
    "metadata": {
        "question_id": "obj_loc_initial",
        "question_type": "single_choice",
        "question_timestep": 0,        # the timestep ASKED about
        "current_timestep": 12,        # when generated
        "template_used": "Initially, which place contained the {obj}?",
    },
    "evidence": {                      # optional but expected
        "question_timestep": 0,
        "current_timestep": 12,
        # ... generator-specific proof fields ...
    },
}
```

Use `utils.create_question_dict(...)` to assemble — it handles the `evidence`
key only being set when not None. `id` is overwritten by `EQAGenerator` with
`f"{question_id}_{counter:03d}"` after validation, so pass `""`.

## Validation (`utils.validate_question_dict`)

A question is dropped silently if **any** fails:

1. Required top-level keys: `id`, `question_text`, `options`,
   `correct_answer`, `metadata`.
2. `options` non-empty dict.
3. For `single_choice`: `correct_answer` is a single label in `options`.
   For `multi_choice`: `correct_answer` is `"A,C,D"` and every split label
   exists in `options`.
4. `metadata` has: `question_timestep`, `current_timestep`, `question_id`,
   `question_type`, `template_used`.
5. If `evidence` present: must contain `question_timestep`. (Other
   `evidence` fields — `current_timestep`, `options_info`, etc. — are
   conventional but not enforced.)

## benchmark_design.yaml entry (must match the generator)

For each generator, add a question block under the matching category:

```yaml
- id: "obj_loc_initial"                    # == func.question_id
  type: "single_choice"                    # == func.question_type
  description: "<one-liner>"               # == func.description
  variables:                               # documents the placeholders
    - "<obj>": "objectType"
    - "<RE_obj_t>": "locDesc (at t=0)"
    - "<RE_obj_f>": "list of locDesc (distractors)"
    - "<X>": "int"
  templates:                               # == func.templates (mirror verbatim)
    - "Where was the <obj> at the start of the task?"
    - "Initially, which place contained the <obj>?"
  option_generation:
    correct_options: "<natural-language rule>"
    distractors: "<natural-language rule>"
    type: "Single Choice (One Correct)"
  generation_notes:                        # these are the agent's design rules
    - "<RE_obj_f> should exclude parent containers of <obj>, including <RE_obj_t>."
    - "<obj> uses objectType; <RE_obj_t>/<RE_obj_f> use locDesc and must not contain sequential numbers."
    - "<X> must not exceed pool size; generally at least 3."
```

Note: yaml uses `<obj>` (angle brackets) while Python templates use `{obj}`
(braces). They describe the same placeholder in two notations.

## Naming conventions (the #1 bug source)

| Field kind | yaml placeholder | Python template | Source | Filter |
|---|---|---|---|---|
| Object type | `<obj>`, `<obj_t>`, `<obj_f>` | `{obj}`, `{obj_t}` | `state.object_type` | none (types have no numbers) |
| Object location | `<RE_obj>`, `<RE_obj_t>`, `<RE_obj_f>` | `{RE_obj}` | `get_object_loc_desc(state)` | drop `has_sequential_number` |
| Container | `<RE_obj>` (context: count/types-in-container) | `{RE_obj}` | `get_object_loc_desc(state)` | drop `has_sequential_number` |
| Direction | `<D>` | n/a (selected from {E,S,W,N}) | `calculate_relative_direction(a, b)` | none |
| Count | `<X>`, `<X1>`, `<X2>` | n/a (computed int) | `len(...)` | n/a |
| Timestep | `<t>`, `<n>` | `{t}`, `{n}` | selected int | n/a |

**Why locDesc must be filtered for sequential numbers**: AI2-THOR emits locDescs
like `"Drawer 1 2"` when there are multiple instances of the same receptacle
type. Including them creates ambiguous distractors (which drawer?). The
benchmark_design.yaml `generation_notes` always says so; honor it.

## Option-count minimums

- `single_choice`: 1 correct + **≥3** distractors (4 options total). If pool
  < 3, `return None`.
- `multi_choice`: `X1` correct (≥1) + **≥2** distractors. The yaml
  `option_generation` notes the minimum per question type.

These floors are not in `validate_question_dict` — they are policy you must
enforce inside the generator. Sample size below the floor → `return None`.

## Common failure modes

- **Generator not registered**: decorator forgotten / attribute typo. Symptom:
  `generate_specific_question` returns None even though file is in place.
  Fix: `grep "is_eqa_generator" <your_file>` should show the assignment.
- **`validate_question_dict` rejects silently**: `correct_answer` label not in
  `options`, or `metadata` missing a key. Symptom: questions.count stays 0
  but generator clearly ran. Add `print(question)` before return to inspect.
- **All-None for one trajectory**: filter too strict for that scene's object
  set. Either relax or pick a different `trajectory_file` in Stage 2.
- **Wrong coordinate axis**: using `dy` instead of `dz` for direction. Use
  `calculate_relative_direction` — never hand-roll `atan2`.
