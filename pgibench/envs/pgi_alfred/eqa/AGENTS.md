# CLAUDE.md — EQA (Embodied Question Answering) System

## Architecture

```
EQAGenerator                              # Auto-discovers and registers all generators
    ├── ThorTracker                       # Records an ObjectState snapshot at every timestep
    ├── benchmark_design.yaml             # Question category definitions (36 question types)
    └── generators/                       # Modular per-category generators
        ├── object_location.py
        ├── object_history.py
        ├── object_relative_position.py
        ├── object_attributes.py
        ├── object_count.py
        ├── object_types_in_container.py
        └── agent_position.py
```

## Question Categories (3 categories, 36 types)

Full definitions in `benchmark_design.yaml`.

**Category 1: Object Spatiotemporal Position**
- Object Location (single): initial / current / N steps ago / random T / history (multi)
- Relative Position object-object (single): initial / current / N steps ago / random T
- Object Types in Container (multi): initial / current / N steps ago / random T

**Category 2: Object Attributes and State**
- Object Existence Seen (multi): current
- Object Property State (single): initial / current / N steps ago / random T
- Object Count in Container (single): initial / current / N steps ago / random T
- Object Count in Scene (single): current

**Category 3: Agent Spatiotemporal Position**
- Agent Location (single): random T
- Agent Relative to Objects (multi): initial / current / N steps ago / random T
- Object Relative to Agent (single): initial / current / N steps ago / random T
- Agent Visible Objects (multi): current / N steps ago / random T
- Object Visible Locations (multi): cumulative

**Temporal variants**: `initial` (t=0), `current`, `N steps ago` (t-N), `random T` (uniform [0,t]), `cumulative` (all timesteps).

## Core Classes

| Class | File | Responsibility |
|-------|------|---------------|
| `EQAGenerator` | `eqa_generator.py` | Auto-discovers generators and manages the question generation flow |
| `ThorTracker` | `thor_tracker.py` | Records an `ObjectState` snapshot at every timestep |
| `ObjectState` | `thor_tracker.py` | Dataclass for an object's state at a single timestep |

## Adding New Question Types

1. Create new file in `generators/`
2. Define a generator function decorated with `@eqa_generator` (sets `is_eqa_generator = True` on it)
3. `EQAGenerator._discover_generators()` will auto-discover it
4. Register question type in `benchmark_design.yaml`

## Configuration

EQA parameters are in `pgibench/configs/pgi-alf.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_eqa` | `True` | Enable EQA question generation |
| `eqa_interval_steps` | `-1` | Generate every N steps (-1: disabled) |
| `eqa_questions_per_interval` | `5` | Questions per interval |
| `eqa_end_episode_questions` | `-15` | Questions per category at episode end |

## Coordinate System

EQA spatial questions use AI2-THOR's coordinate system. See `../CLAUDE.md` (pgi_alfred) for full details.

Key utility: `utils.py:calculate_relative_direction()` — converts world coordinates to cardinal directions (East/South/West/North).
