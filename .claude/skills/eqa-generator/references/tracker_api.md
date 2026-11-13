# ThorTracker & utils API reference

Quick lookup for what you can query inside a generator. Read this BEFORE
deciding whether a question idea is implementable.

## ThorTracker (the `tracker` argument)

Imported from `pgibench.envs.pgi_alfred.eqa.thor_tracker`.

### Attributes

| Attribute | Type | What it is |
|---|---|---|
| `tracker.all_object_ids` | `List[str]` | Every objectId in the scene (static + dynamic). |
| `tracker.historically_seen_objects` | `Set[str]` | ObjectIds the agent has **ever** seen up to `current_timestep`. Use this for candidate iteration (asking about unseen objects is meaningless). |
| `tracker.observations[t]` | `ObservationSnapshot` | Per-timestep visibility set. Has `visible_objects`, `nearby_objects`, `historically_seen_objects`. |
| `tracker.current_timestep` | `int` | Highest recorded timestep. |

### Methods

| Method | Returns | Notes |
|---|---|---|
| `tracker.get_object_state_at_step(obj_id, t)` | `ObjectState \| None` | The atomic object lookup. Returns None if `t` out of range or obj_id unknown at `t`. |
| `tracker.get_agent_state_at_step(t)` | `AgentState \| None` | The atomic agent lookup. **Required** for any "where is the agent / what's the agent facing / distance from agent" question. Returns None if `t` out of range. |
| `tracker.is_unique_object_type(obj_id)` | `bool` | True if this is the only instance of its `object_type` in scene. Gate for unambiguous "the X" reference. |
| `tracker.was_object_ever_held(obj_id)` | `bool` | True if the agent picked it up at some point. Pairs with `get_hold_ordinal`. |
| `tracker.get_hold_ordinal(obj_id)` | `Optional[int]` | 1-based pickup ordinal (the 2nd apple picked up → 2). Used in `held_templates` ("the second {obj} you picked up"). |

## AgentState (returned by `get_agent_state_at_step`)

```python
@dataclass
class AgentState:
    timestep: int
    position: Dict[str, float]        # {'x','y','z'} — feed into calculate_relative_direction
    rotation: Dict[str, float]        # agent body rotation
    camera_horizon: float             # camera pitch (looking up/down)
    facing_direction: str             # 'north' | 'northeast' | ... (8-way)
    facing_angle: float               # body yaw, North=0° clockwise
    inventory_objects: List[Dict]     # [{'objectId': str, 'objectType': str}, ...]
```

Use cases:

- **Distance-to-agent**: `ObjectState.distance` already has it precomputed; you usually
  don't need `agent_state.position` for distance-based questions. But you DO need it
  for direction-based questions (`calculate_relative_direction(agent_state.position,
  obj_state.position)`).
- **Facing-aware filters**: if you need "objects in front of the agent", use
  `calculate_relative_direction` between `agent_state.position` and each
  candidate's position, then match against `agent_state.facing_direction`
  (note: 4-way match against 8-way facing — pick the cardinal that contains
  the agent's 8-way direction, e.g., `'northeast'` matches `"North"` or
  `"East"`; define your rule explicitly).
- **Inventory check**: `inventory_objects` is the source of truth for "is the
  agent holding X right now" (alternative to `was_object_ever_held` for
  current-state questions).

## ObjectState (returned by `get_object_state_at_step`)

```python
@dataclass
class ObjectState:
    object_id: str
    object_type: str             # e.g. "Apple"
    position: Dict[str, float]   # {'x','y','z'} — Y is vertical, ignore for direction
    rotation: Dict[str, float]
    pickupable: bool
    receptacle: bool             # True if it can hold other objects
    openable: bool
    toggleable: bool
    sliceable: bool
    dirtyable: bool
    is_open: bool                # valid only if openable
    is_toggled_on: bool          # valid only if toggleable
    is_sliced: bool              # valid only if sliceable
    is_dirty: bool               # valid only if dirtyable
    is_visible: bool             # at this timestep, from agent's view
    distance: float              # to agent
    is_heated: bool              # special-tracked (microwave cycle)
    is_cooled: bool              # special-tracked (fridge cycle)
    is_cleaned: bool             # special-tracked (sink cycle)
    parent_receptacles: List[str]   # objectIds this is inside (empty → top-level)
    receptacle_object_ids: List[str]  # objectIds inside this (empty → leaf)
    loc_desc: str                # natural-language location, e.g. "CounterTop 1"
```

`is_heated/is_cooled/is_cleaned` are not native AI2-THOR; `ThorTracker` infers
them from action history (microwave / fridge / sink interactions). They are
reliable for `target_timestep ≤ current_timestep`.

## utils.py helpers (import from `pgibench.envs.pgi_alfred.eqa.utils`)

### Option building

| Helper | Signature | When |
|---|---|---|
| `get_options_list(items, labels="ABCDEFG")` | `List[str] → Dict[label, text]` | Always — produces the `options` dict. |
| `select_distractors(correct, pool, n, rng)` | filtered random sample | Single + multi choice distractors. |
| `shuffle_options(options, rng)` | shuffled dict + label mapping | When caller passes `shuffle_opts=True`. |
| `generate_count_distractors(actual, n, rng)` | ints near actual (within ±3) | Count questions. |

### Metadata / question dict

| Helper | Signature | Notes |
|---|---|---|
| `format_question_metadata(...)` | → metadata dict | Required; enforces the 5 keys. |
| `create_question_dict(...)` | → full question dict | Required; handles optional `evidence`. |
| `validate_question_dict(q)` | bool | Pre-flight before storing. Called by EQAGenerator. |

### locDesc / object filtering

| Helper | Signature | When |
|---|---|---|
| `get_object_loc_desc(state)` | `ObjectState → str \| None` | Always for locDesc fields. |
| `has_sequential_number(text)` | bool | Always gate locDescs (drops `"Drawer 1 2"` etc.). |
| `filter_sequential_numbers(list)` | filtered list | Batch version. |
| `is_top_level_object(state)` | bool | True if `parent_receptacles` is empty. |
| `is_immovable_object(state)` | bool | True if not pickupable (for "fixed" candidates). |

### History / containment

| Helper | Signature | Returns |
|---|---|---|
| `get_object_history(tracker, obj_id, t)` | `List[(parent_id, t)]` | Unique parents across `[0, t]`. |
| `get_objects_in_container(tracker, container_id, t)` | `List[obj_id]` | ObjectIds inside the container at `t`. |
| `get_objects_by_type(tracker, type, t)` | `List[obj_id]` | All instances of a type. |
| `get_seen_objects(tracker, t)` | `List[obj_id]` | Anything visible at any step ≤ t. |
| `get_visible_objects_at_step(tracker, t)` | `List[(obj_id, type)]` | Visible exactly at t. |

### Spatial

| Helper | Signature | Returns |
|---|---|---|
| `calculate_relative_direction(pos_a, pos_b)` | `(direction, angle_deg)` | `"East"\|"South"\|"West"\|"North"`, North=0° clockwise. Uses `dz, dx` (Y ignored). |
| `get_objects_in_direction(tracker, agent_pos, direction, t)` | `(in_dir, not_in_dir)` | Two lists of `(obj_id, type, pos)`. |
| `get_object_property_state(state)` | `Optional[Dict]` | `{"property_type": ..., "state": "OPEN"\|"ON"\|"SLICED"\|"DIRTY"}` or None. |

## Coordinate system (don't get this wrong)

AI2-THOR:

- Y is **vertical** — never use for horizontal direction.
- XZ plane: `+X = East`, `-X = West`, `+Z = North`, `-Z = South`.
- Rotation 0°=North, clockwise: 90°=East, 180°=South, 270°=West.

`calculate_relative_direction(pos_a, pos_b)` already maps `atan2(dz, dx)` to
the North=0° clockwise convention. **Never** hand-roll `atan2(dy, dx)` — that
gives elevation angle, not horizontal direction.

## What's NOT in the tracker

If a question needs any of these, the generator must return None (or the user
must extend the tracker first — out of scope for this skill):

- Agent gaze / camera frustum beyond `is_visible` boolean.
- Object-on-object stacking (only `parent_receptacles` is tracked, not "on top
  of" relationships).
- Audio events.
- Container open/close history (only current `is_open` per timestep).
- Agent's intended action vs. executed action (only the executed one is logged).

## Constants module (the `constants` argument)

`constants` is the `pgibench.envs.pgi_alfred.utils.constants` module (formerly
ALFRED's `constants.py`). Use it to:

- Get valid receptacle types: `constants.RECEPTACLES` (list).
- Get scene object lists per FloorPlan: `constants.OBJECTS_OF_INTEREST`.
- Get pickupable / openable / sliceable type predicates.

Most generators don't need `constants` directly — tracker fields suffice. If you
find yourself reaching for `constants` to filter candidates, double-check the
tracker doesn't already expose the predicate (it usually does, via ObjectState
boolean fields).
