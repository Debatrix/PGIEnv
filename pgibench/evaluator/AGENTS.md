# CLAUDE.md — Evaluator Layer

## Architecture

```
PGI_AlfredEvaluator                       # AI2-THOR evaluation loop
    ├── PGIAlfEnv                          # Environment instance
    ├── VLMPlanner                         # Policy model
    └── EQA integration                    # Spatiotemporal QA evaluation

evaluator_utils.py                        # Shared utilities (load_saved_data, update_config_with_args, make_json_serializable)
summarize_result.py                       # Result aggregation (average_json_values)
config/                                   # System prompts, action-mode profiles
```

## Evaluator Files

| File | Function |
|------|----------|
| `pgi_alfred_evaluator.py` | AI2-THOR household tasks evaluation with EQA integration |
| `evaluator_utils.py` | Shared utilities: `load_saved_data()`, `update_config_with_args()`, `make_json_serializable()` |
| `summarize_result.py` | Result aggregation: `average_json_values()` |
| `config/system_prompts.py` | System prompts per environment |
| `config/action_modes.py` | ActionModeProfile: action_display_mode × object_reference_mode |

## Evaluation Subsets

Eval sets are discovered at runtime: `PGIAlfEnv.available_eval_sets()` scans
`pgibench/envs/pgi_alfred/data/tasks/<eval_set>/<task_dir>/task.yaml` — there
is no hard-coded `ValidEvalSets` list. The default `eval_sets` in
`pgibench/configs/pgi-alf.yaml` is `['short_simple', 'short_hard', 'long_simple', 'long_hard']`.

## Evaluation Loop

1. Load config → merge env-specific YAML with CLI overrides
2. Initialize Environment + VLMPlanner
3. Episode loop: `reset()` → `act(img, instruction)` → `step(action)` → repeat
4. EQA: ThorTracker records state → EQAGenerator creates questions → `ask_eqa_question()`
5. Save results to `running/pgi_alfred/{exp_name}/`

## Result Summary

```bash
python -m pgibench.evaluator.summarize_result --directory running/pgi_alfred/exp_name/results
```

## Debugging

```bash
conda activate stbench

# Single episode test
python -m pgibench.main env=pgi-alf selected_indexes='[0]'

# 10% data quick test
python -m pgibench.main env=pgi-alf down_sample_ratio=0.1
```
