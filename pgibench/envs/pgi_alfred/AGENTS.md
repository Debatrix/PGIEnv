# CLAUDE.md — PGI-ALFRED (AI2-THOR) Environment

## Architecture

```
PGIAlfEnv(gym.Env)
    ├── ThorConnector → ThorEnv → AI2-THOR Controller
    ├── _build_static_action_dict()  # dynamic action space
    └── EQA System (eqa/)
```

## X Server (Required)

Recommended (Xvfb - virtual display, works on all environments):
```bash
python -m pgibench.envs.pgi_alfred.scripts.start_xvfb 1
```

Alternative (Xorg):
```bash
python -m pgibench.envs.pgi_alfred.scripts.startx 1
```

## Coordinate System (CRITICAL)

- **Y**: Vertical, NOT for horizontal direction
- **X**: +X=East, -X=West; **Z**: +Z=North, -Z=South
- Rotation 0°=North, clockwise: 90°=East, 180°=South, 270°=West
- Direction: `math.atan2(dz, dx)` → 0°=East, 90°=North

```python
# CORRECT (eqa/utils.py)
angle_deg = (90 - math.degrees(math.atan2(dz, dx))) % 360
# WRONG: math.atan2(dy, dx) — Y is vertical!
```

## Simulator Config

| Setting | Debug | Standard | Production |
|---------|-------|----------|------------|
| resolution | 300 | 500 | 400 |
| quality | Low | MediumCloseFitShadows | MediumCloseFitShadows |
| action_display_mode | full | visible | visible |

## Key APIs

- `get_visible_objects(min_pixel_ratio=0.00005)` — instance segmentation visibility
- `get_obj_id_from_name()` — closed container penalty = 100000, invisible penalty = 1000
- KD-tree for reachable positions/path points: `gen/layouts/{scene}-path-points.json`

## Common Issues

- **Object not found**: Check priority penalties in `thor_connector.py:get_obj_id_from_name()`
- **PutDown fail**: Check `VAL_RECEPTACLE_OBJECTS` compatibility, retry 3x then `drop()`
- **Memory overflow**: ThorTracker keeps the full per-timestep `object_history` for the whole episode (no pruning)

## Key Files

| File | Function |
|------|----------|
| `PGIAlfEnv.py` | Gym env wrapper, action space |
| `thor_connector.py` | AI2-THOR API, spatial queries (2100+ lines) |
| `data/json_2.1.0/` | ALFRED task data (134 scenes) |
| `gen/layouts/*.json` | Pre-computed path points |
