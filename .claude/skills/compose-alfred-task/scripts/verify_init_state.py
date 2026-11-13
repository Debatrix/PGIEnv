#!/usr/bin/env python3
"""Initial-state cleanliness check for compose-alfred-task Stage 2.

Loads a task through PGIAlfEnv -- the EXACT load path the evaluator uses
(_load_dataset -> _reset_controller -> restore scene -> set_task -> CompositeTask)
-- and asserts the initial goal state is zero: no goal condition may be already
satisfied before the agent acts. This is the "fair start" guarantee that
id_trajectory_generator cannot give: that checker drives the task to completion
to prove solvability, but says nothing about whether the random spawn happened to
pre-satisfy some condition (an object already at its target receptacle, a toggle
already in its target state). An init_progress > 0 hands the agent free progress,
so the task is unfair and must be re-rolled (swap seed/object/scene).

task_progress is only computed inside step(); reset() restores the scene and sets
the task but computes nothing. So we read goal_conditions_met() directly off the
underlying ThorEnv right after reset(), with zero agent steps taken.

task location mirrors PGIAlfEnv._load_dataset: the
eval_set's task directories are scanned in sorted order, and selected_indexes
picks by position in that list, so the index computed here matches what the
evaluator would load.

Requires the X server (PGIAlfEnv hardcodes DISPLAY :1):
    python -m pgibench.envs.pgi_alfred.scripts.start_xvfb 1

Run:
    python .claude/skills/compose-alfred-task/scripts/verify_init_state.py \
        --task-dir pgibench/envs/pgi_alfred/data/tasks/<eval_set>/<task_id>
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Auto-detect repo root by walking up to the .git dir. The skill ships under
# .claude/skills/compose-alfred-task/, so the repo root is a few levels up.
_ROOT = Path(__file__).resolve()
while _ROOT.name != '' and not (_ROOT / '.git').is_dir():
    if _ROOT.parent == _ROOT:
        break
    _ROOT = _ROOT.parent
ROOT = _ROOT
if not (ROOT / 'pgibench').is_dir():
    print(f'ERROR: could not locate repo root from {__file__}', file=sys.stderr)
    sys.exit(2)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def find_task_index(task_dir):
    """Resolve (eval_set, task_name, index) for a task directory path.

    Mirrors PGIAlfEnv._load_dataset: scan data/tasks/<eval_set>/ in sorted order
    and return the position of <task_name>'s directory, so selected_indexes=[idx]
    loads exactly this task. Raises SystemExit if the dir is not found under the
    stated eval_set."""
    task_dir = Path(task_dir).resolve()
    eval_set = task_dir.parent.name
    task_name = task_dir.name
    eval_dir = ROOT / 'pgibench' / 'envs' / 'pgi_alfred' / 'data' / 'tasks' / eval_set
    if not eval_dir.is_dir():
        raise SystemExit(
            f'eval_set directory not found: {eval_dir}. '
            f'Available: {sorted(p.name for p in (ROOT / "pgibench/envs/pgi_alfred/data/tasks").iterdir() if p.is_dir())}')
    names = sorted(n for n in os.listdir(eval_dir)
                   if (eval_dir / n / 'task.yaml').is_file())
    if task_name not in names:
        raise SystemExit(
            f'task "{task_name}" not found under {eval_dir}. '
            f'Use the directory name (task_id).')
    return eval_set, task_name, names.index(task_name)


def run(task_dir, out_dir='outputs/temp/task_verify', resolution=300):
    """Load <task_dir> via PGIAlfEnv, check the initial goal state is zero.

    Writes <out_dir>/<name>/init_state.json and returns the report dict."""
    from pgibench.envs.pgi_alfred.PGIAlfEnv import PGIAlfEnv

    eval_set, task_name, idx = find_task_index(task_dir)
    # Minimal kwargs mirroring PGIAlfEnv defaults. action_display_mode='full'
    # avoids command/visible-mode resolver setup we never use (we call no step()).
    # enable_eqa=False skips the tracker/EQA generator. resolution=300 for speed.
    env = PGIAlfEnv(eval_set=eval_set,
                    selected_indexes=[idx],
                    down_sample_ratio=1.0,
                    resolution=resolution,
                    action_display_mode='full',
                    enable_eqa=False)
    try:
        env.reset()                                  # restore scene + set_task; no step taken
        thor = env.env                              # ThorConnector (subclass of ThorEnv)
        task = thor.task                            # CompositeTask set by set_task
        # Coerce numpy scalars to native Python types: goal_conditions_met /
        # goal_satisfied flow through np.any/comparisons and return numpy.bool_ /
        # numpy.int64, which json.dump cannot serialize.
        met, total = (int(v) for v in thor.get_goal_conditions_met())
        satisfied = bool(thor.get_goal_satisfied())
    finally:
        try:
            env.close()                             # ThorConnector.stop()
        except Exception:
            pass

    init_progress = (met / total) if total else 0.0
    # initial_clean: there are goal conditions (total>0) AND none are met (met==0).
    # total==0 means CompositeTask has no subtasks -> fake-success bug (contract
    # rule 3); that is NOT a clean start, it is a malformed task.
    initial_clean = (total > 0 and met == 0)
    num_subtasks = len(task.traj['pddl_params'].get('subtasks', [])) if task else 0
    report = {
        'task_dir': str(Path(task_dir).resolve()),
        'task_name': task_name,
        'eval_set': eval_set,
        'initial_clean': initial_clean,
        'init_progress': round(init_progress, 4),
        'goal_conditions_met': f'{met}/{total}',
        'goal_met': met,
        'goal_total': total,
        'goal_satisfied_at_init': satisfied,
        'num_subtasks': num_subtasks,
    }

    task_out = Path(out_dir) / task_name
    task_out.mkdir(parents=True, exist_ok=True)
    out_path = task_out / 'init_state.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    summary = dict(report)
    summary['init_state_path'] = str(out_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'init_state -> {out_path}')
    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task-dir', required=True,
                    help='task directory (contains task.yaml), same as id_trajectory_generator.')
    ap.add_argument('--out-dir', default='outputs/temp/task_verify',
                    help='output dir (default: outputs/temp/task_verify).')
    ap.add_argument('--resolution', type=int, default=300,
                    help='THOR render resolution (default: 300, debug speed).')
    args = ap.parse_args()

    report = run(args.task_dir, args.out_dir, args.resolution)
    # Exit non-zero when the start is not clean, so a calling orchestrator /
    # shell can branch on the exit code without parsing JSON.
    sys.exit(0 if report['initial_clean'] else 1)


if __name__ == '__main__':
    main()
