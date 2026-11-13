#!/usr/bin/env python3
"""Verify EQA question generators against a live AI2-THOR episode.

Runs a short real episode so ``ThorTracker`` accumulates per-timestep state,
then invokes the selected question generators and prints/saves the generated
questions. Use this to test a new generator (see
``.claude/skills/eqa-generator/``) or to inspect the questions a scene yields.

Usage:
    # verify generators on the shipped smoke_test task
    python scripts/verify_eqa_generator.py \
        --eval-set smoke_test \
        --generators obj_loc_current,obj_loc_initial \
        --num-questions 3 --x-display 2

    # list all registered generator ids
    python scripts/verify_eqa_generator.py --list
"""

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pgibench.envs.pgi_alfred.PGIAlfEnv import PGIAlfEnv

# Safe filler actions used to accumulate tracker timesteps before generation.
FILLER_ACTIONS = ['rotate right', 'look up', 'move ahead', 'rotate left',
                  'look down']


def main():
    parser = argparse.ArgumentParser(
        description='Run EQA generators on a live AI2-THOR episode.')
    parser.add_argument('--eval-set', type=str, default='smoke_test',
                        help='Eval set under pgibench/envs/pgi_alfred/data/tasks/.')
    parser.add_argument('--selected-index', type=int, default=0,
                        help='Task index within the eval set (0-based).')
    parser.add_argument('--generators', type=str, default='',
                        help='Comma-separated generator ids to exercise.')
    parser.add_argument('--num-questions', type=int, default=3,
                        help='Questions to sample per generator.')
    parser.add_argument('--steps', type=int, default=5,
                        help='Filler actions to execute before generating.')
    parser.add_argument('--x-display', type=str, default='1',
                        help='X display number running Xvfb/Xorg.')
    parser.add_argument('--out', type=str,
                        default='running/verify_eqa/questions.json',
                        help='Where to save the generated questions.')
    parser.add_argument('--list', action='store_true',
                        help='List registered generator ids and exit.')
    args = parser.parse_args()

    env = PGIAlfEnv(
        eval_set=args.eval_set,
        down_sample_ratio=1.0,
        exp_name='verify_eqa',
        selected_indexes=[args.selected_index],
        detection_box=False,
        resolution=500,
        action_display_mode='command',
        enable_eqa=True,
        x_display=args.x_display,
    )
    env.reset()

    if args.list:
        for gen_id in sorted(env.eqa_generator.get_registered_generators()):
            print(gen_id)
        env.close()
        return

    selected = [g.strip() for g in args.generators.split(',') if g.strip()]
    if not selected:
        parser.error('--generators is required unless --list is given')
    registered = env.eqa_generator.get_registered_generators()
    unknown = [g for g in selected if g not in registered]
    if unknown:
        print(f'Unknown generators: {unknown}')
        print('Registered generators:')
        for gen_id in sorted(registered):
            print(f'  - {gen_id}')
        env.close()
        sys.exit(1)

    # Accumulate tracker timesteps with safe filler actions (failures are fine;
    # the tracker records the state either way). Keep stepping even after the
    # env reports done — generators need timesteps, not a finished task.
    for i in range(args.steps):
        cmd = FILLER_ACTIONS[i % len(FILLER_ACTIONS)]
        try:
            _obs, _reward, done, info = env.step(cmd)
        except Exception as e:
            print(f'step {i + 1}: {cmd} -> env refused further steps ({e})')
            break
        print(f'step {i + 1}: {cmd} -> '
              f'success={info["last_action_success"]}, '
              f'feedback={info["env_feedback"][:60]}')

    generated, failed = [], 0
    for gen_id in selected:
        for _ in range(args.num_questions):
            question = env.eqa_generator.generate_specific_question(
                question_id=gen_id, timestep=None, seed=None)
            if question:
                generated.append(question)
            else:
                failed += 1
                print(f'[{gen_id}] generator returned None '
                      f'(insufficient tracker data for this scene?)')

    print(f'\nGenerated {len(generated)} question(s), {failed} failed '
          f'across {len(selected)} generator(s), '
          f'{env._current_step} tracker timesteps.')
    for i, q in enumerate(generated):
        print(f'\n--- question {i + 1} ---')
        print(json.dumps(q, indent=2, ensure_ascii=False))

    if generated:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(generated, f, indent=2, ensure_ascii=False)
        print(f'\nSaved to {args.out}')

    env.close()


if __name__ == '__main__':
    main()
