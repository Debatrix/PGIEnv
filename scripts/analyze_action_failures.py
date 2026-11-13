#!/usr/bin/env python3
"""
Analyze action failures across *_full experiment results.

Counts:
- Total actions, failed actions, failure ratio
- Failures caused by "not visible" (object/receptacle not visible)
- Other failure categories
"""

import json
import os
from collections import defaultdict

RESULT_ROOT = os.path.join(os.path.dirname(__file__), "..", "running", "result")


def analyze_experiment(exp_dir):
    """Analyze one experiment directory (may contain sub-dirs like base/)."""
    results_dir = os.path.join(exp_dir, "results")
    if not os.path.isdir(results_dir):
        # try sub-directories (e.g. base/)
        subdirs = [d for d in os.listdir(exp_dir)
                   if os.path.isdir(os.path.join(exp_dir, d))]
        if subdirs:
            merged = {"total_actions": 0, "total_failures": 0,
                      "not_visible_failures": 0, "other_failures": 0,
                      "other_reasons": defaultdict(int), "episodes": 0,
                      "failed_episodes": 0}
            for sub in sorted(subdirs):
                sub_res = analyze_experiment(os.path.join(exp_dir, sub))
                for k in ("total_actions", "total_failures", "not_visible_failures",
                          "other_failures", "episodes", "failed_episodes"):
                    merged[k] += sub_res[k]
                for reason, cnt in sub_res["other_reasons"].items():
                    merged["other_reasons"][reason] += cnt
            return merged
        return None

    stats = {
        "total_actions": 0,
        "total_failures": 0,
        "not_visible_failures": 0,
        "other_failures": 0,
        "other_reasons": defaultdict(int),
        "episodes": 0,
        "failed_episodes": 0,
    }

    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith("_final_res.json"):
            continue
        fp = os.path.join(results_dir, fn)
        with open(fp) as f:
            data = json.load(f)

        log = data.get("episode_log", [])
        stats["episodes"] += 1
        ep_failed = False

        for entry in log:
            stats["total_actions"] += 1
            if not entry.get("last_action_success", True):
                stats["total_failures"] += 1
                ep_failed = True
                fb = entry.get("env_feedback", "")
                if "not visible" in fb.lower():
                    stats["not_visible_failures"] += 1
                else:
                    stats["other_failures"] += 1
                    if fb:
                        reason = fb.replace("Last action is invalid. ", "").strip()
                        if len(reason) > 80:
                            reason = reason[:77] + "..."
                        stats["other_reasons"][reason] += 1
                    else:
                        stats["other_reasons"]["(empty feedback)"] += 1

        if ep_failed:
            stats["failed_episodes"] += 1

    return stats


def main():
    result_root = os.path.abspath(RESULT_ROOT)
    if not os.path.isdir(result_root):
        print(f"Result directory not found: {result_root}")
        return

    # Find all *_full directories
    experiments = sorted([
        d for d in os.listdir(result_root)
        if d.endswith("_full") and os.path.isdir(os.path.join(result_root, d))
    ])

    if not experiments:
        print("No *_full experiments found.")
        return

    print(f"{'Experiment':<40} {'Episodes':>8} {'Actions':>8} {'Failed':>8} "
          f"{'Fail%':>7} {'NotVisible':>11} {'NV%':>6} {'Other':>6}")
    print("-" * 100)

    all_stats = {
        "total_actions": 0, "total_failures": 0,
        "not_visible_failures": 0, "other_failures": 0,
        "episodes": 0, "failed_episodes": 0,
        "other_reasons": defaultdict(int),
    }

    for exp_name in experiments:
        exp_dir = os.path.join(result_root, exp_name)
        stats = analyze_experiment(exp_dir)
        if stats is None:
            print(f"{exp_name:<40} (no results found)")
            continue

        fail_pct = (stats["total_failures"] / stats["total_actions"] * 100
                    if stats["total_actions"] > 0 else 0)
        nv_pct = (stats["not_visible_failures"] / stats["total_failures"] * 100
                  if stats["total_failures"] > 0 else 0)

        print(f"{exp_name:<40} {stats['episodes']:>8} {stats['total_actions']:>8} "
              f"{stats['total_failures']:>8} {fail_pct:>6.1f}% "
              f"{stats['not_visible_failures']:>11} {nv_pct:>5.1f}% "
              f"{stats['other_failures']:>6}")

        # Accumulate
        for k in ("total_actions", "total_failures", "not_visible_failures",
                   "other_failures", "episodes", "failed_episodes"):
            all_stats[k] += stats[k]
        for reason, cnt in stats["other_reasons"].items():
            all_stats["other_reasons"][reason] += cnt

    # Summary
    print("-" * 100)
    ta = all_stats["total_actions"]
    tf = all_stats["total_failures"]
    nv = all_stats["not_visible_failures"]
    fail_pct = tf / ta * 100 if ta > 0 else 0
    nv_pct = nv / tf * 100 if tf > 0 else 0
    print(f"{'TOTAL':<40} {all_stats['episodes']:>8} {ta:>8} {tf:>8} "
          f"{fail_pct:>6.1f}% {nv:>11} {nv_pct:>5.1f}% "
          f"{all_stats['other_failures']:>6}")

    if all_stats["other_reasons"]:
        print(f"\nOther failure reasons breakdown:")
        for reason, cnt in sorted(all_stats["other_reasons"].items(),
                                   key=lambda x: -x[1]):
            print(f"  {cnt:>4}x  {reason}")


if __name__ == "__main__":
    main()
