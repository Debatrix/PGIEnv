# PGI-ALFRED Environment Guide

An OpenAI Gym environment (`PGIAlfEnv`, see `PGIAlfEnv.py`) built as a wrapper on top of [ALFRED](https://arxiv.org/abs/1912.01734) (an AI2-THOR household-task benchmark).

This document describes the **action space**, **observation space**, and the design of the **EQA (Embodied Question Answering)** subsystem that this environment provides under different configurations.

---

## 1. Architecture Overview

```
PGIAlfEnv(gym.Env)
    ├── ThorConnector ──► ThorEnv ──► AI2-THOR Controller
    │       ├── Visible / nearby object queries (instance segmentation + distance)
    │       ├── path_points (KD-tree waypoint navigation)
    │       └── cooled / heated / cleaned_objects (implicit environment state)
    ├── _build_static_action_dict()      # builds the static action dictionary once from scene objects
    ├── _update_language_skill_set()     # refreshes the display action list every step according to the visibility mode
    ├── ThorTracker                       # spatiotemporal state tracking (used by EQA)
    └── EQAGenerator                      # auto-discovers and registers question generators
```

Key files:

| File | Role |
|------|------|
| `PGIAlfEnv.py` | Gym environment wrapper; builds/refreshes the action space and dispatches EQA |
| `thor_connector.py` | Low-level AI2-THOR API and spatial queries (waypoints, visible objects, interactions) |
| `eqa/thor_tracker.py` | Records per-timestep spatiotemporal state of objects/agent/observations/actions |
| `eqa/eqa_generator.py` | EQA question generation entry point (auto-discovers `eqa/generators/`) |
| `eqa/generators/*.py` | Question templates and generation logic for each category (agent_position, object_location, etc.) |
| `eqa/benchmark_design.yaml` | Design specification for question categories and templates |
| `data/splits/splits.json` | Task splits for each `eval_set` |
| `gen/layouts/*-path-points.json` | Precomputed waypoints for each scene |

---

## 2. Action Space

The action space is represented by `gym.spaces.Discrete`, and its size changes dynamically with the scene and configuration. `__init__` first initializes it with 8 global actions; **on the first `reset()`**, `_build_static_action_dict()` scans all objects in the scene, builds the complete static action dictionary, and extends the `Discrete` dimension. Afterwards, if the scene changes or new objects appear, it is rebuilt/appended incrementally.

Actions fall into two broad categories: **global navigation actions** and **object interaction actions**. Interaction actions use `objectId` as an immutable key, and are filled in with natural language based on the current `locDesc` when presented to the model.

### 2.1 Global Navigation Actions (constant 8)

```
LookUp, LookDown, RotateLeft, RotateRight,
MoveAhead, MoveBack, MoveLeft, MoveRight
```

Defined by `get_global_action_space()`, always available, with fixed indices 0–7.

### 2.2 Object Interaction Actions (generated dynamically from scene objects)

Each interaction type is generated only for objects in the scene that have the corresponding attribute (and whose type is in the `alfred_pick_obj / alfred_open_obj / alfred_toggle_obj / alfred_slice_obj` whitelist):

| Action Type | Constant | Trigger Condition | Description |
|-------------|----------|-------------------|-------------|
| Pick up | `ACTION_PICKUP` | `pickupable == True` | `pick up {loc_desc}`, including Sliced variants |
| Drop (no target) | `ACTION_DROP` | Constant 1 | `drop the object in hand`, only valid while holding an object |
| Place | `ACTION_PUTDOWN` | Pickupable objects × valid receptacles (`VAL_RECEPTACLE_OBJECTS`) | `put down {obj_type} {in/on} {recep_loc_desc}`, automatically choosing `in`/`on` based on the receptacle type |
| Open / Close | `ACTION_OPEN` / `ACTION_CLOSE` | `openable == True` | Generated in pairs: `open {loc_desc}` / `close {loc_desc}` |
| Toggle On / Off | `ACTION_TOGGLE_ON` / `ACTION_TOGGLE_OFF` | `toggleable == True` | `turn on/off {loc_desc}` |
| Slice | `ACTION_SLICE` | `sliceable == True` | `slice {loc_desc}`, only valid while holding a knife |

> The preposition for placement actions is determined by `_get_preposition_for_receptacle()`: receptacles such as `Fridge/Microwave/Cabinet/Bowl/...` use `in`, while surfaces such as `DiningTable/CounterTop/Shelf/...` use `on`.

### 2.3 Navigation Actions (controlled by `nav_mode`)

| `nav_mode` | Actions Generated | Description |
|------------|-------------------|-------------|
| `path_point` (default) | `goto waypoint_#NN` | Navigate to a precomputed waypoint of the scene (from `gen/layouts/{scene}-path-points.json`; the `loc_desc` looks like `waypoint_#00`) |
| `object` | `navigate to {loc_desc}` | Navigate near each interactable object/receptacle in the scene (the position is located automatically via `nav_obj`) |
| `none` | None | No additional navigation actions are generated |

### 2.4 Display Mode (controlled by `action_display_mode`)

`language_skill_set` is the action list **actually presented to the model**. Two modes control its contents:

| `action_display_mode` | Behavior |
|-----------------------|----------|
| `full` | Shows **all** valid actions in the static dictionary |
| `visible` (default/recommended) | Shows only actions related to **currently visible / nearby** objects; placement actions additionally require that the corresponding object is currently held; slicing additionally requires holding a knife |

> In `visible` mode, `language_skill_set` changes dynamically at every step, and the integer indices of `action_space` are refreshed accordingly. `_should_include_action_visible()` implements the filtering logic. Model actions are mapped back to static keys via `display_to_static_key` before execution.

### 2.5 Per-Episode Constraints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `_max_episode_steps` | 30 (100 for `composite` tasks) | Maximum number of steps |
| `_max_invalid_actions` | 3 | Cap on consecutive invalid actions; `done` once exceeded |

---

## 3. Observation Space

Each `reset()` / `step()` returns two dictionaries, `obs` and `info`.

### 3.1 `obs` (Observation)

| Key | Type | Description |
|-----|------|-------------|
| `head_rgb` | `np.ndarray (H, W, 3)` | First-person RGB image from the agent's viewpoint; the resolution is determined by `resolution` (default 500) |

### 3.2 `info` (Environment Feedback and State, returned by `step()`)

| Key | Type | Description |
|-----|------|-------------|
| `instruction` | `str` | Natural-language task instruction for this episode |
| `env_step` | `int` | Current step count (starting from 1) |
| `env_feedback` | `str` | Natural-language feedback: `"Last action execute successfully."` on success; on failure, the AI2-THOR error message is appended |
| `last_action_success` | `float` | 0.0 / 1.0 |
| `task_success` | `float` | Whether the goal conditions are satisfied (0/1) |
| `task_progress` | `float` | Completed subgoals / total subgoals |
| `reward` | `float` | Dense reward (`reward_type='dense'`) |
| `episode_elapsed_seconds` | `float` | Cumulative elapsed time of this episode |
| `agent_position` | `dict` | Agent pose (see below) |
| `object_states` | `dict` | Implicit object states (see below) |
| `action_id` / `action_description` | `int`/`str` | The action taken at this step |
| `reasoning` | `str` | Reasoning process passed in by the caller (optional) |
| `eqa_question` | `dict` | Present only when `enable_eqa=True` and `generate_question=True` |

**`agent_position` fields** (`thor_connector.get_agent_position()`):

| Field | Description |
|-------|-------------|
| `position` | `{x, y, z}`; Y is the vertical axis, X = east-west, Z = north-south (+Z = north) |
| `horizon` | Camera pitch angle |
| `rotation_y` | Heading angle, 0° = north, increasing clockwise (90° = east, 180° = south, 270° = west) |
| `direction` | 8-way textual direction (north / northeast / ... ) |
| `closest_path_point` | Nearest waypoint `waypoint_#NN` |
| `is_on_path_point` | Whether the agent is currently on a waypoint (< 0.2m) |
| `distance_to_closest_path_point` | Distance to the nearest waypoint |
| `relative_direction_to_path_point` | Direction relative to the nearest waypoint |

**`object_states` fields** (implicit environment state, not native AI2-THOR properties):

| Field | Description |
|-------|-------------|
| `cooled_objects` | Set of objects that have been "cooled" (e.g., in the Fridge) |
| `heated_objects` | Set of objects that have been "heated" (e.g., with the Microwave) |
| `cleaned_objects` | Set of objects that have been "cleaned" (e.g., with the SinkBasin) |
| `visible_objs` | List of currently visible object IDs (determined by instance segmentation) |
| `nearby_objs` | Currently nearby objects (distance < `PROXIMITY_DETECTION_DISTANCE`) |

### 3.3 Key Visualization Configuration

| Parameter | Description |
|-----------|-------------|
| `resolution` | Render resolution (300 for debugging / 500 standard) |
| `detection_box` | Whether to overlay detection boxes of visible objects on saved images |
| `visual_prompt` | Whether to enable visual prompting (implicitly enables detection as well) |

---

## 4. EQA Embodied Question Answering System

EQA evaluates the agent's **spatiotemporal memory** capability — during/after task execution, it generates multiple-choice questions about the states of objects and the agent at **different points in time**, based on the historical trajectory, and scores the model's answers against them. The subsystem consists of three parts.

### 4.1 How to Enable

Pass `enable_eqa=True` when constructing the environment:

```python
env = PGIAlfEnv(eval_set='base',
                action_display_mode='visible',
                nav_mode='path_point',
                enable_eqa=True)
```

Or in the config file `pgibench/configs/pgi-alf.yaml`:

```yaml
enable_eqa: True
eqa_interval_steps: -1            # >0: generate a batch every N steps; -1: do not generate mid-episode
eqa_questions_per_interval: 5     # number of questions generated per interval
eqa_end_episode_questions: -15    # >0: generate N questions per category; <0: randomly generate |N| questions at the end; =0: generate none
```

When enabled:
- `self.thor_tracker = ThorTracker()` with `enable_thor_tracking=True`, recording spatiotemporal state at every step;
- `self.eqa_generator = EQAGenerator(self.thor_tracker, constants)`, which automatically scans `eqa/generators/*.py` to register generators.

### 4.2 ThorTracker (Spatiotemporal State Tracking)

At every step, `update_step()` records four kinds of snapshots (all serializable dataclasses):

| Data Class | Contents | Example Query Interfaces |
|------------|----------|--------------------------|
| `ObjectState` | Per-object, per-step position, orientation, open/closed, powered on/off, sliced, dirty/clean, heated/cooled/cleaned, visibility, parent receptacle, `locDesc` | `get_object_state_at_step()`, `get_object_state_changes()` |
| `AgentState` | Per-step agent position, orientation, camera angle, heading direction, inventory (held object) | `get_agent_state_at_step()`, `get_agent_path()` |
| `ObservationSnapshot` | Per-step visible object set, nearby object set, cumulative set of objects seen so far | `was_object_visible_at_step()`, `get_first_seen_timestep()` |
| `ActionRecord` | Per-step available action set, executed action, success flag, environment feedback, reasoning | `get_action_at_step()` |

The tracker provides rich cross-timestep queries (object position history, moments when objects were picked up/placed down, state change sequences, receptacle contents, etc.) and serves as the ground-truth source for EQA question generation. It can be exported via `save_thor_tracking()` as `episode_{i}_thor.json`.

### 4.3 EQAGenerator (Question Generation)

`EQAGenerator` auto-discovers all functions decorated with `@eqa_generator(...)` under `eqa/generators/` (identified via the `is_eqa_generator` attribute) and registers them by `question_id`.

**Core interface**:

| Method | Purpose |
|--------|---------|
| `generate_random_question(timestep, seed)` | Randomly picks a generator to produce 1 question |
| `generate_specific_question(question_id, ...)` | Produces 1 question of the specified category |
| `generate_batch_questions(num, specific=...)` | Batch generation: with `specific=True`, produces `num` questions for **each registered category**; otherwise produces `num` random questions |
| `save_to_json(filepath, episode_metadata)` | Exports to `episode_{i}_eqa.json`, containing the question list and summary statistics |

**Question data structure** (see `eqa/utils.create_question_dict`):

```jsonc
{
  "id": "obj_loc_current_003",
  "question_text": "Where is the Apple right now?",
  "options": { "A": "...", "B": "...", "C": "...", "D": "..." },
  "correct_answer": "C",                       // "A,C" for multi-choice
  "metadata": {
    "question_id": "obj_loc_current",
    "question_type": "single_choice",          // or "multi_choice"
    "question_timestep": 5,                    // the timestep the question asks about
    "current_timestep": 7,
    "template_used": "Where is the {obj} right now?"
  },
  "evidence": { ... }                          // optional; contains ground-truth tracking data
}
```

### 4.4 Question Categories

The design specification is in `eqa/benchmark_design.yaml`, with three major categories. Each category usually generates questions in five temporal variants — **initial (t=0) / current / n steps ago / random time t / history** — to probe memory over different time spans. The registered `question_id`s are as follows:

**1. Object Spatiotemporal Position**

| `question_id` | Type | Description |
|---------------|------|-------------|
| `obj_loc_initial` / `obj_loc_current` / `obj_loc_n_steps_ago` / `obj_loc_random_t` | Single choice | The receptacle (`locDesc`) the object is in at the corresponding timestep; distractors are other receptacles in the scene |
| `obj_loc_history` | Multi choice | All locations the object was placed at over the whole episode (including held-object variants) |
| `obj_rel_pos_initial` / `_current` / `_n_steps_ago` / `_random_t` | Single choice | Relative direction of two objects in the XZ plane (east/south/west/north) |
| `obj_types_in_container_initial` / `_current` / `_n_steps_ago` / `_random_t` | Multi choice | Which object types a receptacle contains at the corresponding timestep |

**2. Object Attributes and State**

| `question_id` | Type | Description |
|---------------|------|-------------|
| `obj_existence_seen` | Multi choice | Which objects actually exist and have been seen by the agent (distractors are types drawn from `constants.py` that do not exist in this scene) |
| `obj_property_initial` / `_current` / `_n_steps_ago` / `_random_t` | Single choice | Object attribute state: receptacle open/closed, powered on/off, sliced or not, heated/cooled/cleaned, whether picked up |
| `obj_count_in_container_initial` / `_current` / `_n_steps_ago` / `_random_t` | Single choice | Number of objects in a receptacle (distractors are adjacent integers ±1–3) |
| `obj_count_in_scene_current` | Single choice | Total count of a given type in the scene |

**3. Agent Spatiotemporal Position**

| `question_id` | Type | Description |
|---------------|------|-------------|
| `obj_rel_agent_initial` / `_current` / `_n_steps_ago` / `_random_t` | Single choice | Direction of an object relative to the agent (east/south/west/north) |
| `agent_rel_objs_initial` / `_current` / `_n_steps_ago` / `_random_t` | Multi choice | Which objects lie in a given direction (east/south/west/north) from the agent |
| `agent_vis_initial` / `_n_steps_ago` / `_random_t` | Multi choice | Which object types were visible to the agent at the corresponding timestep |

> Note: the design specification (`benchmark_design.yaml`) still lists `agent_pos_random_t`, `agent_vis_current`, `obj_vis_pos`, etc.; whether they are actually registered depends on the implementation of the corresponding files in `eqa/generators/`. Use `eqa_generator.get_registered_generators()` to query the list of currently registered `question_id`s.

### 4.5 Integration into the Evaluation Pipeline

`pgibench/evaluator/pgi_alfred_evaluator.py` provides two EQA evaluation timings:

1. **Mid-task (interval sampling)**: when `eqa_interval_steps > 0` and `current_step % interval == 0`, it calls `generate_batch_questions(eqa_questions_per_interval, specific=False)` to generate questions, immediately has the model answer them via `planner.ask_eqa_question()`, and grades the answers (`evaluate_eqa_questions`).
2. **Episode end (batch)**: `specific=True/False` is chosen based on the sign of `eqa_end_episode_questions`; a final batch of questions is generated and graded.

Grading (`_compare_eqa_answer`) supports single choice (exact match of the letter) and multi choice (set equality after deduplication). Episode-level EQA accuracy is written to `episode_info['eqa_results']` and included in the result summary. In addition, `env.save_eqa_questions()` and `env.save_thor_tracking()` write the questions and tracking data to disk for offline analysis.

---

## 5. Coordinate System Conventions (Important)

- **Y**: vertical axis, **must not** be used for horizontal direction computations.
- **X**: +X = east, −X = west; **Z**: +Z = north, −Z = south.
- Heading angle `rotation_y`: 0° = north, increasing clockwise (90° = east, 180° = south, 270° = west).
- Relative direction is computed with `math.atan2(dz, dx)` and then converted to the north-as-0° clockwise system (see `eqa/utils.calculate_relative_direction`).

---

## 6. Runtime Requirements

- Python interpreter: the `stbench` conda environment (`conda activate stbench`).
- X Server (required for AI2-THOR rendering): `X_DISPLAY = '1'` (hardcoded in `PGIAlfEnv.py`); it must be started first in a separate tmux session:
  ```bash
  python -m pgibench.envs.pgi_alfred.scripts.startx 1
  ```
- Data: `data/splits/splits.json`, `data/json_2.1.0/`, `gen/layouts/*-path-points.json`.

> For the original upstream ALFRED instructions, citations, and hardware requirements, see historical versions of this repository or [askforalfred.com](https://askforalfred.com).
