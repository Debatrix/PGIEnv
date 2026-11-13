#!/usr/bin/env python3
"""Summarize human evaluation results on disk.

Walks the per-evaluator result layout produced by the human evaluation server
(``running/pgi_alfred/{exp_name}/{eval_set}/{evaluator_id}/results/``),
aggregates metrics per evaluator, per eval_set (pooled across evaluators) and
overall, and writes JSON + CSV summaries.

Usage:
    python scripts/summarize_human_eval.py \
        --root running/pgi_alfred/human_eval \
        --out running/pgi_alfred/human_eval/summary
"""

import os
import csv
import json
import argparse
import statistics


def iter_result_files(root):
    """Yield (eval_set, evaluator_id, json_path) for every episode result."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith('_final_res.json'):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            parts = rel.split(os.sep)
            if len(parts) < 4:   # <eval_set>/<evaluator_id>/results/<file>
                print(f"[skip] unexpected layout: {rel}")
                continue
            eval_set, evaluator_id = parts[0], parts[1]
            yield eval_set, evaluator_id, os.path.join(dirpath, fn)


def safe_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize_episode(rec):
    """Extract the flat per-episode metrics used for aggregation."""
    times = rec.get('action_selection_times') or []
    return {
        'task_success': rec.get('task_success', 0.0),
        'task_progress': rec.get('task_progress', 0.0),
        'reward': rec.get('reward', 0.0),
        'num_steps': rec.get('num_steps', 0),
        'num_invalid_actions': rec.get('num_invalid_actions', 0),
        'num_invalid_action_ratio': rec.get('num_invalid_action_ratio', 0.0),
        'subtask_completion': rec.get('subtask_completion', 0.0),
        'episode_elapsed_seconds': rec.get('episode_elapsed_seconds', None),
        'mean_decision_time': safe_mean(times),
        'eqa_accuracy': rec.get('eqa_accuracy', -1),
        'eqa_total': rec.get('eqa_total', -1),
        'eqa_correct': rec.get('eqa_correct', -1),
        # Per-question-type EQA counts, aggregated across episodes
        'eqa_by_type': _eqa_by_type(rec.get('eqa_results') or []),
    }


def _eqa_by_type(eqa_results):
    counts = {}
    for r in eqa_results:
        q_type = (r.get('metadata') or {}).get('question_type', 'unknown')
        c = counts.setdefault(q_type, {'total': 0, 'correct': 0})
        c['total'] += 1
        c['correct'] += 1 if r.get('is_correct') else 0
    return counts


def aggregate(records):
    """Aggregate a list of per-episode summaries into group-level metrics."""
    n = len(records)
    out = {'n_episodes': n}
    if n == 0:
        return out
    eqa_recs = [r for r in records if r['eqa_accuracy'] is not None
                and r['eqa_accuracy'] >= 0]
    out.update({
        'success_rate': safe_mean([r['task_success'] for r in records]),
        'mean_task_progress': safe_mean([r['task_progress'] for r in records]),
        'mean_reward': safe_mean([r['reward'] for r in records]),
        'mean_steps': safe_mean([r['num_steps'] for r in records]),
        'median_steps': statistics.median([r['num_steps'] for r in records]),
        'mean_invalid_action_ratio': safe_mean(
            [r['num_invalid_action_ratio'] for r in records]),
        'mean_subtask_completion': safe_mean(
            [r['subtask_completion'] for r in records]),
        'mean_decision_time': safe_mean(
            [r['mean_decision_time'] for r in records
             if r['mean_decision_time'] is not None]),
        'eqa_accuracy': safe_mean([r['eqa_accuracy'] for r in eqa_recs]),
        'eqa_episodes': len(eqa_recs),
    })
    by_type = {}
    for r in eqa_recs:
        for q_type, c in r['eqa_by_type'].items():
            t = by_type.setdefault(q_type, {'total': 0, 'correct': 0})
            t['total'] += c['total']
            t['correct'] += c['correct']
    for q_type, c in by_type.items():
        c['accuracy'] = c['correct'] / c['total'] if c['total'] else None
    out['eqa_by_type'] = by_type
    return out


CSV_FIELDS = ['group', 'eval_set', 'evaluator_id', 'n_episodes', 'success_rate',
              'mean_task_progress', 'mean_reward', 'mean_steps', 'median_steps',
              'mean_invalid_action_ratio', 'mean_subtask_completion',
              'mean_decision_time', 'eqa_accuracy', 'eqa_episodes']


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate human evaluation episode results (JSON + CSV).')
    parser.add_argument('--root', type=str,
                        default='running/pgi_alfred/human_eval',
                        help='Root directory holding '
                             '<eval_set>/<evaluator_id>/results/ layouts.')
    parser.add_argument('--out', type=str, default=None,
                        help='Output prefix; writes <out>.json and <out>.csv '
                             '(default: <root>/summary).')
    args = parser.parse_args()
    out_prefix = args.out or os.path.join(args.root, 'summary')

    # group -> list of per-episode summaries; group is (eval_set, evaluator_id)
    # for per-rater rows or (eval_set, '*') / ('*', '*') for pooled rows.
    episodes = {}
    for eval_set, evaluator_id, path in iter_result_files(args.root):
        with open(path, encoding='utf-8') as f:
            rec = json.load(f)
        s = summarize_episode(rec)
        episodes[(eval_set, evaluator_id)] = episodes.get(
            (eval_set, evaluator_id), []) + [s]
        episodes[(eval_set, '*')] = episodes.get((eval_set, '*'), []) + [s]
        episodes[('*', '*')] = episodes.get(('*', '*'), []) + [s]

    rows = []
    summary = {}
    for (eval_set, evaluator_id), recs in sorted(episodes.items()):
        metrics = aggregate(recs)
        group = ('overall' if evaluator_id == '*' and eval_set == '*'
                 else 'pooled_eval_set' if evaluator_id == '*'
                 else 'per_evaluator')
        rows.append({'group': group, 'eval_set': eval_set,
                     'evaluator_id': evaluator_id, **{
                         k: v for k, v in metrics.items()
                         if k != 'eqa_by_type'}})
        summary.setdefault(group, {})[f'{eval_set}/{evaluator_id}'] = metrics

    with open(out_prefix + '.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_prefix + '.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_prefix}.json and {out_prefix}.csv "
          f"({len(rows)} aggregate rows)")


if __name__ == '__main__':
    main()
