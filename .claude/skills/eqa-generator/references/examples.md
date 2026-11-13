# Designing EQA generators — patterns & red lines

A generator is a small Python function, but the contract is dense. These
patterns cover the canonical shapes; copy the closest one and adapt.

## Red lines (self-review before Stage 2)

1. **objectType vs locDesc**: never mix sources.
   - `<obj>` / `<obj_t_list>` → `state.object_type`, no number filter.
   - `<RE_obj>` / `<RE_obj_t>` / `<RE_obj_f>` → `get_object_loc_desc(state)`,
     **gated by `has_sequential_number`**.
2. **Distractor disjointness**: distractors must not include any correct option.
   Use `select_distractors(correct_list, pool, n, rng)` — it filters.
3. **Return None on thin data**: pool smaller than the type minimum
   (single_choice ≥3 distractors, multi_choice ≥2 distractors) → `return None`.
   Do NOT fabricate or pad with duplicates.
4. **Evidence traceability**: every option (correct + distractor) should be
   checkable against a tracker field listed in `evidence`. For single_choice,
   at least list the query object's id + `parent_receptacles`. For multi_choice,
   enumerate per-option proof in `evidence["options_info"]`.

## Pattern 1 — Single-choice with distractor pool (obj_loc_initial style)

The most common shape. Ask "where is X?" → 1 correct locDesc + 3 distractors
from scene's other locDescs.

```python
@eqa_generator(question_id="obj_loc_initial",
               question_type="single_choice",
               templates=["Initially, which place contained the {obj}?"],
               description="Object location at t=0.")
def generate(tracker, constants, timestep, rng):
    target_timestep = 0
    current_timestep = tracker.current_timestep

    # 1. Pick a unique-seen object as the query.
    candidates = [oid for oid in tracker.historically_seen_objects
                  if tracker.is_unique_object_type(oid)]
    if not candidates:
        return None
    obj_id = rng.choice(candidates)
    state = tracker.get_object_state_at_step(obj_id, target_timestep)
    if not state:
        return None

    # 2. Correct = its locDesc at t=0.
    correct_loc = get_object_loc_desc(state)
    if not correct_loc or has_sequential_number(correct_loc):
        return None

    # 3. Distractor pool = other objects' locDescs.
    pool = set()
    for oid in tracker.all_object_ids:
        if oid == obj_id:
            continue
        s = tracker.get_object_state_at_step(oid, target_timestep)
        loc = get_object_loc_desc(s) if s else None
        if loc and loc != correct_loc and not has_sequential_number(loc):
            pool.add(loc)
    pool = list(pool)

    if len(pool) < 3:
        return None
    distractors = select_distractors([correct_loc], pool, 3, rng)

    # 4. Assemble.
    all_opts = [correct_loc] + distractors
    rng.shuffle(all_opts)
    options = get_options_list(all_opts)
    correct_label = next(l for l, t in options.items() if t == correct_loc)

    template = rng.choice(generate.templates)
    return create_question_dict(
        question_id="",
        question_text=template.format(obj=state.object_type.lower()),
        options=options,
        correct_answer=correct_label,
        metadata=format_question_metadata(
            question_id="obj_loc_initial", question_type="single_choice",
            question_timestep=target_timestep,
            current_timestep=current_timestep, template_used=template),
        evidence={"question_timestep": target_timestep,
                  "current_timestep": current_timestep,
                  "query_object": {"objectId": obj_id,
                                   "parentReceptacles": state.parent_receptacles}})
```

Temporal variants: copy the function, change `target_timestep`:

- `obj_loc_current` → `target_timestep = current_timestep`
- `obj_loc_n_steps_ago` → `target_timestep = max(0, current_timestep - n)`
  (sample `n` from `[1, current_timestep]`)
- `obj_loc_random_t` → `target_timestep = rng.randint(0, current_timestep)`

## Pattern 2 — Multi-choice subset (obj_types_in_container_current style)

"Select all that apply". Correct options are a subset of the true items;
distractors are items not in the container.

```python
@eqa_generator(question_id="obj_types_in_container_current",
               question_type="multi_choice",
               templates=["Which types of objects are currently inside the {RE_obj}?"],
               description="Object types inside a container now.")
def generate(tracker, constants, timestep, rng):
    target_timestep = tracker.current_timestep

    # 1. Find containers where ALL contents have been seen.
    containers = []
    for cid in tracker.all_object_ids:
        cs = tracker.get_object_state_at_step(cid, target_timestep)
        if not (cs and cs.receptacle_object_ids):
            continue
        if all(o in tracker.historically_seen_objects
               for o in cs.receptacle_object_ids):
            loc = get_object_loc_desc(cs)
            if loc and not has_sequential_number(loc):
                containers.append((cid, loc, cs.receptacle_object_ids))
    if not containers:
        return None

    cid, c_loc, inside_ids = rng.choice(containers)
    correct_types = list({tracker.get_object_state_at_step(o, target_timestep).object_type
                          for o in inside_ids})

    # 2. Distractor pool = types in scene but NOT in this container.
    in_set = set(correct_types)
    pool = set()
    for oid in tracker.all_object_ids:
        s = tracker.get_object_state_at_step(oid, target_timestep)
        if s and s.object_type not in in_set:
            pool.add(s.object_type)
    pool = list(pool)

    if len(correct_types) < 1 or len(pool) < 2:
        return None

    # 3. Subset of corrects + distractors.
    x1 = rng.randint(1, min(3, len(correct_types)))
    x2 = rng.randint(2, min(3, len(pool)))
    correct_opts = rng.sample(correct_types, x1)
    distractors = select_distractors(correct_opts, pool, x2, rng)

    all_opts = correct_opts + distractors
    rng.shuffle(all_opts)
    options = get_options_list(all_opts)
    correct_labels = [l for l, t in options.items() if t in correct_opts]
    correct_answer = ",".join(correct_labels)

    template = rng.choice(generate.templates)
    return create_question_dict(
        question_id="",
        question_text=template.format(RE_obj=c_loc),
        options=options,
        correct_answer=correct_answer,
        metadata=format_question_metadata(
            question_id="obj_types_in_container_current",
            question_type="multi_choice",
            question_timestep=target_timestep,
            current_timestep=tracker.current_timestep,
            template_used=template),
        evidence={"question_timestep": target_timestep,
                  "current_timestep": tracker.current_timestep,
                  "container": {"objectId": cid, "locDesc": c_loc,
                                "receptacleObjectIds": inside_ids},
                  "options_info": {l: {"option_text": t,
                                       "is_correct": t in correct_opts}
                                   for l, t in options.items()}})
```

## Pattern 5 — Options-as-candidates, no placeholder (closest/visible/etc.)

All 4 patterns above put the queried object in the stem (`{obj}`). When the
question enumerates objects AS the options ("which of these is closest?",
"which are visible?", "which was picked up first?"), putting the answer in
the stem would leak it. Use a **placeholder-free template** and list all
candidate objects as options.

Signs you're in this pattern:

- The user prompt says "which of these..." / "select all..." with the answer
  being one of the listed options.
- The correct answer is selected FROM the option set, not referenced BY the stem.
- Templates have no `{obj}` / `{RE_obj}` placeholder (or only a non-answer
  placeholder like `{RE_obj}` referring to a container, not the answer).

```python
@eqa_generator(question_id="obj_nearest_agent_initial",
               question_type="single_choice",
               templates=[
                   "Initially, which of these objects was closest to the agent?",
                   "At the start of the episode, which object was nearest to the agent?",
               ],
               description="Object nearest the agent at t=0.")
def generate_obj_nearest_agent_initial(tracker, constants, timestep, rng):
    target_timestep = 0
    current_timestep = tracker.current_timestep

    # Candidates: top-level, historically-seen objects with clean locDescs.
    candidates = []
    for obj_id in tracker.historically_seen_objects:
        s = tracker.get_object_state_at_step(obj_id, target_timestep)
        if not s or not is_top_level_object(s):
            continue
        loc = get_object_loc_desc(s)
        if loc and not has_sequential_number(loc):
            candidates.append((obj_id, loc, s.distance))  # distance already precomputed
    if len(candidates) < 4:    # 1 correct + 3 distractors
        return None

    # Correct = min by distance (ObjectState.distance is to agent — no math needed).
    candidates.sort(key=lambda c: c[2])
    correct_id, correct_loc, correct_dist = candidates[0]

    distractors = select_distractors(
        [correct_loc], [c[1] for c in candidates[1:]], 3, rng)
    if len(distractors) < 3:
        return None

    all_opts = [correct_loc] + distractors
    rng.shuffle(all_opts)
    options = get_options_list(all_opts)
    correct_label = next(l for l, t in options.items() if t == correct_loc)

    template = rng.choice(generate_obj_nearest_agent_initial.templates)
    # NOTE: no template.format() — template has no placeholders.

    # Per-option proof so a grader can verify "nearest" via distance comparison.
    id_by_loc = {c[1]: c[0] for c in candidates}
    dist_by_loc = {c[1]: c[2] for c in candidates}
    options_info = {
        label: {"option_text": text,
                "objectId": id_by_loc[text],
                "distance_to_agent_at_t0": dist_by_loc[text],
                "is_correct": text == correct_loc}
        for label, text in options.items()
    }

    return create_question_dict(
        question_id="",
        question_text=template,             # no .format() call
        options=options,
        correct_answer=correct_label,
        metadata=format_question_metadata(
            question_id="obj_nearest_agent_initial", question_type="single_choice",
            question_timestep=target_timestep,
            current_timestep=current_timestep, template_used=template),
        evidence={"question_timestep": target_timestep,
                  "current_timestep": current_timestep,
                  "query_object": {"objectId": correct_id, "locDesc": correct_loc,
                                   "distance_to_agent_at_t0": correct_dist},
                  "options_info": options_info})
```

Key differences from Patterns 1–4:

- **Templates have no placeholder** — listing the answer object in the stem
  would reveal it.
- **`question_text = template`** — no `.format()` call.
- **Per-option proof is more important** — since the answer is computed (min
  distance / visibility flag / etc.), `evidence["options_info"]` should
  include the discriminator value (distance, is_visible, …) for every option
  so a grader can re-derive the correct answer.

## Pattern 3 — History / cumulative (obj_loc_history style)

Asks about everywhere an object has been. Needs ≥2 unique parents to be
interesting.

```python
@eqa_generator(question_id="obj_loc_history",
               question_type="multi_choice",
               templates=["Select all locations where the {obj} has been placed so far."],
               description="Locations the object has been (cumulative).")
def generate(tracker, constants, timestep, rng):
    current_timestep = tracker.current_timestep
    candidates = []
    for oid in tracker.historically_seen_objects:
        if not tracker.is_unique_object_type(oid):
            continue
        history = get_object_history(tracker, oid, current_timestep)
        if len(history) < 2:    # need ≥2 unique parents
            continue
        candidates.append((oid, history))
    if not candidates:
        return None

    oid, history = rng.choice(candidates)
    correct_locs = []
    for pid, _ in history:
        ps = tracker.get_object_state_at_step(pid, current_timestep)
        loc = get_object_loc_desc(ps)
        if loc and not has_sequential_number(loc):
            correct_locs.append(loc)
    correct_locs = list(set(correct_locs))
    if not correct_locs:
        return None

    # Distractor pool: all other parent locDescs in scene.
    pool = set()
    for oid2 in tracker.all_object_ids:
        s = tracker.get_object_state_at_step(oid2, current_timestep)
        loc = get_object_loc_desc(s) if s else None
        if loc and loc not in correct_locs and not has_sequential_number(loc):
            pool.add(loc)
    pool = list(pool)

    x1 = rng.randint(1, min(3, len(correct_locs)))
    x2 = min(rng.randint(2, 3), len(pool))
    if x2 < 2:
        return None
    correct_opts = rng.sample(correct_locs, x1)
    distractors = select_distractors(correct_opts, pool, x2, rng)

    all_opts = correct_opts + distractors
    rng.shuffle(all_opts)
    options = get_options_list(all_opts)
    correct_labels = [l for l, t in options.items() if t in correct_opts]

    template = rng.choice(generate.templates)
    return create_question_dict(
        question_id="",
        question_text=template.format(obj=tracker.get_object_state_at_step(
            oid, 0).object_type.lower()),
        options=options,
        correct_answer=",".join(correct_labels),
        metadata=format_question_metadata(
            question_id="obj_loc_history", question_type="multi_choice",
            question_timestep=current_timestep,
            current_timestep=current_timestep, template_used=template),
        evidence={"question_timestep": current_timestep,
                  "current_timestep": current_timestep,
                  "query_object": {"objectId": oid,
                                   "history": [(pid, t) for pid, t in history]}})
```

## Pattern 4 — Hold-ordinal reference (for non-unique objects)

When an object type has multiple instances ("three apples"), reference by
pickup ordinal ("the second apple you picked up"). Set `held_templates`:

```python
@eqa_generator(question_id="obj_loc_history",
               question_type="multi_choice",
               templates=["Select all locations where the {obj} has been placed so far."],
               held_templates=[
                   "Select all locations where the {ordinal} {obj} you picked up has been placed."
               ],
               description="...")
def generate(tracker, constants, timestep, rng):
    # ... pick from non-unique held objects ...
    if is_unique:
        template = rng.choice(generate.templates)
        question_text = template.format(obj=obj_type.lower())
    else:
        template = rng.choice(generate.held_templates)
        ordinal_str = _format_ordinal(tracker.get_hold_ordinal(oid))  # "2nd"
        question_text = template.format(obj=obj_type.lower(), ordinal=ordinal_str)
```

`_format_ordinal` is a 5-line helper; copy from `object_history.py`.

## What makes a good question_id?

- Pattern: `<concept>_<temporal_variant>` (`obj_loc_initial`,
  `agent_vis_n_steps_ago`).
- All snake_case. Suffix matches yaml variant naming (`initial` / `current` /
  `n_steps_ago` / `random_t` / `history`).
- Unique across the whole `generators/` directory. `grep` first.

## What makes a good template?

- 3–5 alternative phrasings per question type — gives the benchmark linguistic
  diversity.
- Placeholder names match yaml `<...>` placeholders: `{obj}`, `{RE_obj}`,
  `{t}`, `{n}`, `{D}`.
- Avoid words that leak the answer ("the apple on the **counter**" → leaks
  "counter"). Use spatial / temporal language only.

## When a question idea is NOT implementable

Tell the user plainly if any of these:

- Needs data not in `ObjectState` (see `references/tracker_api.md` "What's
  NOT in the tracker").
- Requires comparing across episodes (tracker is per-episode).
- Needs a coordinate axis the tracker doesn't expose (Y for "above/below").
- Requires fabricating distractors (e.g., "object types not in any scene" —
  pool comes from scene reality).

Propose the closest implementable alternative instead.
