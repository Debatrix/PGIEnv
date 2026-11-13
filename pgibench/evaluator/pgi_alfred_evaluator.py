import os
import numpy as np
from tqdm import tqdm
import time
import json
from glob import glob
from pgibench.envs.pgi_alfred.PGIAlfEnv import PGIAlfEnv
from pgibench.planner.vlm_planner import VLMPlanner
from pgibench.evaluator.summarize_result import average_json_values
from pgibench.evaluator.evaluator_utils import load_saved_data, update_config_with_args, make_json_serializable
from pgibench.evaluator.config.system_prompts import alfred_system_prompt
from pgibench.main import logger

system_prompt = alfred_system_prompt


class PGI_AlfredEvaluator():

    def __init__(self, config):
        self.config = config
        self.model_name = config.get('model_name')
        self.max_steps = config.get('max_steps', 50)
        self.env = None
        self.planner = None

    def check_config_valid(self):
        if self.config['multistep'] + self.config['chat_history'] > 1:
            raise ValueError(
                "Only one of multistep, chat_history can be enabled at a time."
            )

        if self.config['language_only']:
            if self.config['multistep']:
                logger.warning(
                    "Language only mode should not have multistep enabled. Setting these arguments to False ..."
                )
                self.config['multistep'] = 0

    def save_episode_metric(self, episode_info):
        episode_idx = self.env._current_episode_num if not len(
            self.env.selected_indexes
        ) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1

        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        episode_info = make_json_serializable(episode_info)
        episode_messages = episode_info.pop('episode_messages', [])
        # Strip image_url content blocks (large base64 data URLs) before saving,
        # keeping the textual content so the message history stays readable.
        episode_messages = [
            {
                **msg,
                "content": [
                    block for block in msg.get("content", [])
                    if not (isinstance(block, dict)
                            and block.get("type") == "image_url")
                ] if isinstance(msg.get("content"), list) else msg.get("content"),
            }
            for msg in episode_messages
        ]
        with open(os.path.join(res_path,
                               f'episode_{episode_idx}_final_res.json'),
                  'w',
                  encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)

        with open(os.path.join(res_path,
                               f'episode_{episode_idx}_messages.json'),
                  'w',
                  encoding='utf-8') as f:
            json.dump(episode_messages, f, ensure_ascii=False)

        self.env.save_thor_tracking(res_path)

    def evaluate_eqa_questions(self, questions, img_path):
        """
        Evaluate a batch of EQA questions using the planner.

        Args:
            questions: List of EQA question dicts
            img_path: Path to current observation image

        Returns:
            List of evaluation results with answers and correctness
        """
        results = []

        for idx, question in enumerate(questions):
            # Get correct answer
            correct_answer = question.get('correct_answer', '')
            question_type = question.get('metadata',
                                         {}).get('question_type',
                                                 'single_choice')

            # Ask the question
            answer, reasoning = self.planner.ask_eqa_question(
                question, img_path)

            # Compare with correct answer
            is_correct = self._compare_eqa_answer(answer, correct_answer,
                                                  question_type)

            results.append({
                'question_id': question.get('id', ''),
                'question_text': question['question_text'],
                'options': question['options'],
                'correct_answer': correct_answer,
                'model_answer': answer,
                'reasoning': reasoning,
                'is_correct': is_correct,
                'evidence': question['evidence'],
                'metadata': question.get('metadata', {})
            })

            print(
                f"[E{self.env._current_episode_num}|Q{idx + 1}/{len(questions)}] {question['question_text']}. Options: {question['options']}. Correct Answer: {correct_answer}. Model Answer: {answer}. Reasoning: {reasoning}. Is Correct: {is_correct}"
            )

        return results

    def _compare_eqa_answer(self, model_answer: str, correct_answer: str,
                            question_type: str) -> bool:
        """
        Compare model answer with correct answer, handling both single and multi-choice.

        Args:
            model_answer: The answer provided by the model
            correct_answer: The correct answer(s), comma-separated for multi-choice
            question_type: 'single_choice' or 'multi_choice'

        Returns:
            True if answer is correct, False otherwise
        """
        if not model_answer or not correct_answer:
            return False

        # Normalize answers: remove whitespace, convert to uppercase
        model_normalized = model_answer.replace(' ', '').upper()
        correct_normalized = correct_answer.replace(' ', '').upper()

        if question_type == 'multi_choice':
            # For multi-choice: split by comma and check if all correct options are present
            # and no incorrect options are included
            model_options = set(model_normalized.split(','))
            correct_options = set(correct_normalized.split(','))

            # Check if model selected all correct options (exact match)
            return model_options == correct_options
        else:
            # For single-choice: direct string comparison
            return model_normalized == correct_normalized

    def evaluate_main(self):
        valid_eval_sets = self.config.get('eval_sets', None)
        valid_eval_sets = list(valid_eval_sets) if valid_eval_sets else []
        if len(valid_eval_sets) == 0:
            # No eval_sets specified: discover all eval_sets present on disk.
            valid_eval_sets = PGIAlfEnv.available_eval_sets()

        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(
                self.config['exp_name']
            ) else f"{self.model_name.split('/')[-1]}/{eval_set}"
            self.env = PGIAlfEnv(
                eval_set=self.eval_set,
                down_sample_ratio=self.config['down_sample_ratio'],
                exp_name=exp_name,
                selected_indexes=self.config.get('selected_indexes', []),
                detection_box=self.config.get('detection_box', False),
                action_display_mode=self.config.get('action_display_mode',
                                                    'visible'),
                visual_prompt=self.config.get('visual_prompt', False),
                resolution=self.config.get('resolution', 500),
                enable_eqa=self.config.get('enable_eqa', True),
                nav_mode=self.config.get('nav_mode', ['default_waypoint']),
                object_reference_mode=self.config.get('object_reference_mode',
                                                      'id'),
                bbox_iou_threshold=self.config.get('bbox_iou_threshold', 0.3),
                point_dist_ratio=self.config.get('point_dist_ratio', 0.3),
                instruction_mode=self.config.get('instruction_mode') or 'micro',
                x_display=self.config.get('x_display', '1'))
            model_type = self.config.get('model_type', 'remote')
            self.planner = VLMPlanner(
                self.model_name,
                model_type,
                self.env.language_skill_set,
                system_prompt,
                obs_key='head_rgb',
                chat_history=self.config['chat_history'],
                language_only=self.config['language_only'],
                use_feedback=self.config.get('env_feedback', True),
                multistep=self.config.get('multistep', 0),
                action_display_mode=self.config.get('action_display_mode',
                                                    'visible'),
                object_reference_mode=self.config.get('object_reference_mode',
                                                      'id'),
                nav_mode=self.config.get('nav_mode', ['default_waypoint']),
                image_history_keep=self.config.get('image_history_keep', 5),
                eqa_image_history_keep=self.config.get('eqa_image_history_keep', 5))

            self.evaluate()
            # average_json_values(os.path.join(self.env.log_path, 'results'),
            #                     output_file='summary.json')
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def evaluate(self):
        if self.env is None:
            self.env = PGIAlfEnv(eval_set='base',
                                down_sample_ratio=1.0,
                                selected_indexes=[],
                                visual_prompt=False,
                                action_display_mode='visible',
                                enable_eqa=True)

        if self.planner is None:
            raise RuntimeError(
                "VLMPlanner not initialized; call evaluate_main() first.")
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        exists_episode = [
            int(os.path.basename(x).split('_')[1]) for x in glob(
                os.path.join(self.env.log_path, 'results',
                             'episode_*_final_res.json'))
        ]
        exists_episode_num = max(exists_episode) if len(
            exists_episode) > 0 else 0
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(
                f"Evaluating episode {self.env._current_episode_num} ...")
            if self.env._current_episode_num < exists_episode_num:
                logger.info(
                    f"Episode {self.env._current_episode_num} already exists, skipping ..."
                )
                self.env._current_episode_num += 1
                progress_bar.update()
                continue

            episode_info = {
                'reward': [],
                'num_invalid_actions': 0,
                'empty_plan': 0
            }
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")

            self.planner.reset()
            done = False
            failed_step_num = 0
            for step_num in range(self.max_steps):
                if failed_step_num > 0.15 * self.max_steps:
                    logger.info(
                        f"Too many failed steps, terminating episode early at step {step_num} ..."
                    )
                    break
                try:
                    # Get current agent position
                    agent_position = self.env.get_agent_position()

                    # update the action space for alfred due to dynamic objects
                    self.planner.set_actions(
                        self.env.language_skill_set,
                        object_catalog=self.env.get_object_catalog(
                            mode=self.env.object_reference_mode),
                        nav_waypoints=self.env.get_available_waypoints())
                    action, reasoning = self.planner.act(
                        img_path,
                        user_instruction,
                        agent_position=agent_position)
                    print(
                        f"[E{self.env._current_episode_num}|S{step_num}] Planner Output: {reasoning}"
                    )
                    if action == -1:  # empty plan stop here
                        episode_info['empty_plan'] = 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -1,
                            'action_description': 'Done(empty plan)',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success':
                            episode_info.get('task_success', 0),
                            'task_progress':
                            episode_info.get("task_progress", 0),
                            'num_subtasks':
                            episode_info.get('num_subtasks', 0),
                            'total_goal_conditions':
                            episode_info.get('total_goal_conditions', 0),
                            'subtask_completion':
                            episode_info.get('subtask_completion', 0.0),
                            'subtask_details':
                            episode_info.get('subtask_details', []),
                            'env_step': self.env._current_step,
                        }
                        break

                    # Add agent position to info for planner
                    agent_position = self.env.get_agent_position()

                    if isinstance(action, dict):  # command mode: {'command', 'target_bbox', 'target_point'}
                        action_str = action['command']
                        obs, reward, done, info = self.env.step(
                            action_str,
                            reasoning=reasoning,
                            target_bbox=action.get('target_bbox'),
                            target_point=action.get('target_point'))
                    else:
                        action_str = action if type(
                            action) == str else self.env.language_skill_set[action]
                        obs, reward, done, info = self.env.step(
                            action, reasoning=reasoning)
                    print(
                        f"[E{self.env._current_episode_num}|S{step_num}] Executed action: {action_str}, Task success: {info['task_success']}, Subtask: {info.get('subtask_completion', 0.0):.2f}, Env feedback: {info['env_feedback']}"
                    )
                    logger.debug(f"reward: {reward}")
                    logger.debug(f"terminate: {done}\n")

                    info['agent_position'] = agent_position

                    self.planner.update_info(info)
                    img_path = self.env.save_image(obs)
                    episode_info['reward'].append(reward)
                    episode_info['num_invalid_actions'] += (
                        info['last_action_success'] == 0)

                    # Generate and evaluate EQA questions at intervals
                    if self.config.get('enable_eqa', False):
                        eqa_interval = self.config.get('eqa_interval_steps',
                                                       10)
                        eqa_per_interval = self.config.get(
                            'eqa_questions_per_interval', 5)

                        if self.env._current_step > 0 and self.env._current_step % eqa_interval == 0 and eqa_interval > 0:
                            logger.info(
                                f"Generating {eqa_per_interval} EQA questions at step {self.env._current_step}"
                            )
                            questions = self.env.eqa_generator.generate_batch_questions(
                                eqa_per_interval, specific=False)

                            if questions:
                                eqa_results = self.evaluate_eqa_questions(
                                    questions, img_path)

                                # Store results in episode_info (initialize list if needed)
                                if 'eqa_results' not in episode_info:
                                    episode_info['eqa_results'] = []
                                episode_info['eqa_results'].extend(eqa_results)

                                # Log accuracy so far
                                total_eqa = len(episode_info['eqa_results'])
                                correct_eqa = sum(
                                    1 for r in episode_info['eqa_results']
                                    if r['is_correct'])
                                logger.info(
                                    f"EQA accuracy so far: {correct_eqa}/{total_eqa} = {correct_eqa/total_eqa if total_eqa > 0 else 0:.2%}"
                                )
                    if done:
                        break

                except Exception as e:
                    print(e)
                    failed_step_num += 1
                    time.sleep(30)

            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward'])
            episode_info['task_success'] = info['task_success']
            episode_info["task_progress"] = info['task_progress']
            episode_info["num_subtasks"] = info.get("num_subtasks", 0)
            episode_info["total_goal_conditions"] = info.get(
                "total_goal_conditions", 0)
            episode_info["subtask_completion"] = info.get(
                "subtask_completion", 0.0)
            episode_info["subtask_details"] = info.get("subtask_details", [])
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info[
                'planner_output_error'] = self.planner.output_json_error
            episode_info["num_invalid_actions"] = episode_info[
                'num_invalid_actions']
            episode_info["num_invalid_action_ratio"] = episode_info[
                'num_invalid_actions'] / info["env_step"] if info[
                    'env_step'] > 0 else 0
            episode_info["episode_elapsed_seconds"] = info.get(
                "episode_elapsed_seconds",
                time.time() - self.env._episode_start_time)
            # Placeholder, will be updated below if EQA results exist
            episode_info['eqa_accuracy'] = -1
            episode_info['eqa_total'] = -1
            episode_info['eqa_correct'] = -1
            episode_info["episode_log"] = self.env.episode_log
            episode_info["episode_messages"] = self.planner.episode_messages

            # Generate and evaluate final EQA questions at episode end
            if self.config.get('enable_eqa', False):
                eqa_end_count = self.config.get('eqa_end_episode_questions', 2)

                if eqa_end_count > 0:
                    logger.info(
                        f"Generating {eqa_end_count} per categories final EQA questions at episode end"
                    )
                    questions = self.env.eqa_generator.generate_batch_questions(
                        eqa_end_count, specific=True)
                elif eqa_end_count < 0:
                    eqa_end_count = -eqa_end_count
                    logger.info(
                        f"Generating {eqa_end_count} final EQA questions at episode end"
                    )
                    questions = self.env.eqa_generator.generate_batch_questions(
                        eqa_end_count, specific=False)
                else:
                    questions = []
                    logger.info("No EQA questions generated")

                if questions:
                    eqa_results = self.evaluate_eqa_questions(
                        questions, img_path)

                    if 'eqa_results' not in episode_info:
                        episode_info['eqa_results'] = []
                    episode_info['eqa_results'].extend(eqa_results)

            # Calculate EQA metrics
            if episode_info.get('eqa_results'):
                total_eqa = len(episode_info['eqa_results'])
                correct_eqa = sum(1 for r in episode_info['eqa_results']
                                  if r['is_correct'])
                episode_info[
                    'eqa_accuracy'] = correct_eqa / total_eqa if total_eqa > 0 else 0
                episode_info['eqa_total'] = total_eqa
                episode_info['eqa_correct'] = correct_eqa
                logger.info(
                    f"Final EQA accuracy: {correct_eqa}/{total_eqa} = {episode_info['eqa_accuracy']:.2%}"
                )

            self.save_episode_metric(episode_info)
            progress_bar.update()


