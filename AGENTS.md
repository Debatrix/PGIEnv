# AGENTS.md

This file provides guidance to all AI agents (Claude Code, OpenCode, etc.) when working with code in this repository.

## Python Environment

```bash
conda env create -f conda_envs/environment.yaml
conda activate stbench
```

**STRICT TESTING:** All code MUST be tested with the actual simulator (AI2-THOR). No mock, stub, or simplified bypass testing.
**NO UNSOLICITED DOCS:** Do not generate tutorials, HOWTOs, or API docs; explain via inline comments instead. `README.md` and `docs/` are maintainer-owned.

## Architecture

```
pgibench/main.py (Hydra entry point)
    → pgibench/evaluator/pgi_alfred_evaluator.py (evaluation loop)
    → pgibench/envs/pgi_alfred/PGIAlfEnv.py (Gym-style env wrapper)
    → pgibench/planner/vlm_planner.py (VLM interaction)
```

Sub-module docs: `pgibench/evaluator/AGENTS.md`, `pgibench/envs/pgi_alfred/AGENTS.md`, `pgibench/planner/AGENTS.md`, `pgibench/envs/pgi_alfred/eqa/AGENTS.md`

## Common Commands

```bash
conda activate stbench

# Start headless X server for AI2-THOR (required on machines without a display)
python -m pgibench.envs.pgi_alfred.scripts.start_xvfb 1  # Recommended (Xvfb)
python -m pgibench.envs.pgi_alfred.scripts.startx 1      # Alternative (Xorg)

# Run an evaluation (model must be configured, see pgibench/planner/model_configs/)
python -m pgibench.main env=pgi-alf model_name=gpt-4o-mini exp_name='baseline'

# Debug: subsample episodes or run a single episode
python -m pgibench.main env=pgi-alf down_sample_ratio=0.1
python -m pgibench.main env=pgi-alf selected_indexes='[0]'

# Human evaluation web server (start the Xvfb display(s) first)
python -m pgibench.human_eval.server --config pgibench/configs/human_eval.yaml
```

## Configuration (Hydra)

- `pgibench/configs/config.yaml` — Base (all null)
- `pgibench/configs/pgi-alf.yaml` — AI2-THOR defaults
- CLI overrides: `python -m pgibench.main env=pgi-alf model_name=X eval_sets="['short_simple']"`

## Evaluation Subsets

Eval sets live in `pgibench/envs/pgi_alfred/data/tasks/<eval_set>/`; the default
configuration uses `short_simple`, `short_hard`, `long_simple`, `long_hard`.

## Dev Constraints

1. Test with the actual simulator, never mock
2. No unsolicited documentation files, inline comments only
3. Temp files → `outputs/temp`, configs → `pgibench/configs/`, model configs → `pgibench/planner/model_configs/`
4. No try-catch to mask errors
5. Edit existing scripts, don't create duplicates

## Extending the Benchmark

- **Add a model**: drop a YAML in `pgibench/planner/model_configs/` (see the
  README there), add the API key to `.env` (see `.env.template`).
- **Add EQA question generators**: see `pgibench/envs/pgi_alfred/eqa/AGENTS.md`.
- **Author new tasks**: see `.claude/skills/compose-alfred-task/` — the task
  authoring pipeline (`task_builder` + `id_trajectory_generator`).

## Sub-module Documentation

- [pgibench/evaluator/AGENTS.md](./pgibench/evaluator/AGENTS.md) — Evaluator layer documentation
- [pgibench/envs/pgi_alfred/AGENTS.md](./pgibench/envs/pgi_alfred/AGENTS.md) — PGI-ALFRED environment documentation
- [pgibench/planner/AGENTS.md](./pgibench/planner/AGENTS.md) — Planner and model layer documentation
- [pgibench/envs/pgi_alfred/eqa/AGENTS.md](./pgibench/envs/pgi_alfred/eqa/AGENTS.md) — EQA system documentation
