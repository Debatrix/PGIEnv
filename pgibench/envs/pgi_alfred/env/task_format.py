#!/usr/bin/env python3
"""Task format: task.yaml (task definition + scene init) + optional
trajectory.json (expert trajectory).

This module is the loader/serializer boundary. It assembles a task back
into the traj dict shape that PGIAlfEnv / CompositeTask consume, so the judging
layer needs no changes. When no expert trajectory is present, a minimal placeholder
plan (high_pddl=[End]) is synthesized: num_subgoals becomes 0, which does NOT
affect SR/GC (judging reads only scene state) nor episode termination (PGIAlfEnv
computes its own `done`, overriding transition_reward's).
"""

import os
import json
from typing import Any, Dict, Optional

import yaml


def _minimal_plan() -> Dict[str, list]:
    """Placeholder plan for tasks without an expert trajectory."""
    return {
        "high_pddl": [
            {"high_idx": 0,
             "discrete_action": {"action": "End", "args": []},
             "planner_action": {"action": "End", "value": 1}}
        ],
        "low_actions": [],
    }


def _resolve_instructions(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Return {'macro':..., 'micro':...}. New schema uses cfg['instructions'];
    legacy single cfg['instruction'] is treated as micro (macro left empty)."""
    ins = cfg.get('instructions')
    if isinstance(ins, dict):
        return {'macro': ins.get('macro', ''), 'micro': ins.get('micro', '')}
    legacy = cfg.get('instruction', '')
    return {'macro': '', 'micro': legacy}


def _resolve_scene_block(scene_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the scene block: ensure floor_plan and restore_scene fields exist."""
    scene = dict(scene_cfg)
    if "scene_num" in scene and "floor_plan" not in scene:
        scene["floor_plan"] = "FloorPlan%d" % scene["scene_num"]
    scene.setdefault("object_poses", [])
    scene.setdefault("object_toggles", [])
    scene.setdefault("dirty_and_empty", False)
    return scene


def load_task_dir(task_dir: str) -> Dict[str, Any]:
    """Read <task_dir>/task.yaml (+ optional trajectory.json) -> traj dict."""
    with open(os.path.join(task_dir, "task.yaml")) as f:
        cfg = yaml.safe_load(f)

    traj_path = os.path.join(task_dir, "trajectory.json")
    if os.path.exists(traj_path):
        with open(traj_path) as f:
            traj_file = json.load(f)
        plan = {
            "high_pddl": traj_file.get("high_pddl", _minimal_plan()["high_pddl"]),
            "low_actions": traj_file.get("low_actions", []),
        }
        images = traj_file.get("images", [])
    else:
        plan = _minimal_plan()
        images = []

    return {
        "task_id": cfg.get("task_id", os.path.basename(task_dir)),
        "task_type": cfg.get("task_type", "composite"),
        "pddl_params": cfg["pddl_params"],
        "scene": _resolve_scene_block(cfg["scene"]),
        "plan": plan,
        "turk_annotations": {
            "anns": [{
                "task_desc": _resolve_instructions(cfg)['micro'],
                "high_descs": [],
                "votes": [1, 1],
            }]
        },
        "instructions": _resolve_instructions(cfg),
        "images": images,
    }


def write_task(task_dir: str, cfg: Dict[str, Any],
                         trajectory: Optional[Dict[str, Any]] = None) -> None:
    """Write <task_dir>/task.yaml (+ trajectory.json if given). Creates task_dir."""
    os.makedirs(task_dir, exist_ok=True)
    with open(os.path.join(task_dir, "task.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    if trajectory is not None:
        with open(os.path.join(task_dir, "trajectory.json"), "w") as f:
            json.dump(trajectory, f, indent=2)
